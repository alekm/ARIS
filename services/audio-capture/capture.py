import asyncio
import logging
import os
import json
import redis
import threading
import time
import yaml

from kiwi_client import KiwiSDRClient

# Logging Setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CaptureService")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
STREAM_CONTROL = "control:audio-capture"
STREAM_AUDIO = "audio:chunks"


class CaptureThread(threading.Thread):
    """Manages a single KiwiSDR connection in a separate thread/loop"""
    def __init__(self, slot_id, config):
        super().__init__()
        self.slot_id = str(slot_id)
        self.config = config
        self.running = False
        self.client = None
        self.loop = None
        
        # Redis (Per thread) - Use Sync Client for stability
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

    def run(self):
        logger.info(f"Slot-{self.slot_id} - Starting Capture Thread")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.running = True
        self.loop.run_until_complete(self._lifecycle())
        self.loop.close()
        logger.info(f"Slot-{self.slot_id} - Thread Stopped")

    async def _lifecycle(self):
        while self.running:
            try:
                # Initialize Client
                self.client = KiwiSDRClient(
                    host=self.config['host'],
                    port=self.config['port'],
                    password=self.config.get('password', ''),
                    frequency_hz=self.config['frequency_hz'],
                    mode=self.config['mode']
                )
                
                # Attach Audio Callback
                self.client.on_audio_data = self._handle_audio
                
                # Connect (Blocking until disconnect)
                await self.client.connect()
                
            except Exception as e:
                logger.error(f"Slot-{self.slot_id} - Error: {e}", exc_info=True)
                
            # If we are strictly stopped, break. If error, retry after delay.
            if not self.running: break
            await asyncio.sleep(5) # Retry delay

    async def _handle_audio(self, audio_bytes, seq):
        """Callback from KiwiSDRClient"""
        try:
            # Ensure audio_bytes is even length (required for 16-bit PCM)
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes[:-1]
            
            # Calculate filter cutoffs based on mode
            low_cut = 300
            high_cut = 2700
            if self.client.mode == "LSB":
                low_cut = -2700
                high_cut = -300
            elif self.client.mode == "AM":
                low_cut = -5000
                high_cut = 5000
            elif self.client.mode == "CW":
                low_cut = 400
                high_cut = 800
            
            # Match Shared Model (AudioChunk) Protocol
            payload = {
                "source_id": self.slot_id,
                "timestamp": str(time.time()),
                "frequency_hz": str(self.client.frequency_hz),
                "mode": self.client.mode,
                "sample_rate": "12000",
                "duration_ms": str(int((len(audio_bytes) / 2 / 12000) * 1000)),
                "seq": str(seq),
                "data": audio_bytes.hex(),  # Convert to hex string as expected by models.py
                "s_meter": "0.0",
                "signal_strength_db": "-150.0",
                "squelch_open": "true",  # Boolean as string for Redis
                "low_cut": str(low_cut),
                "high_cut": str(high_cut),
                "rssi": ""  # Optional, empty if not available
            }
            
            self.redis.xadd(STREAM_AUDIO, payload, maxlen=1000)
            
        except Exception as e:
            logger.error(f"Slot-{self.slot_id} - Audio Publish Error: {e}", exc_info=True)

    def stop(self):
        self.running = False
        if self.client and self.loop:
            # Schedule stop in the event loop
            try:
                asyncio.run_coroutine_threadsafe(self.client.stop(), self.loop)
            except Exception as e:
                logger.warning(f"Slot-{self.slot_id} - Error stopping client: {e}")


class SlotManager:
    def __init__(self):
        self.slots = {}  # {slot_id: CaptureThread}
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    def start(self):
        logger.info("SlotManager starting...")
        
        # Auto-start from environment variables (backward compatibility)
        self._auto_start_from_env()
        
        logger.info("SlotManager running. Waiting for commands...")
        last_id = '$'
        while True:
            try:
                # Read Control Stream
                messages = self.redis.xread({STREAM_CONTROL: last_id}, count=1, block=5000)
                if messages:
                    for stream, msgs in messages:
                        for msg_id, data in msgs:
                            last_id = msg_id
                            self.handle_command(data)
            except Exception as e:
                logger.error(f"Control Loop Error: {e}", exc_info=True)
                time.sleep(1)

    def _auto_start_from_env(self):
        """Auto-start a slot from environment variables (backward compatibility)"""
        mode = os.getenv('MODE', '').lower()
        if mode not in ['kiwi', 'kiwisdr']:
            logger.info("MODE not set to 'kiwi', skipping auto-start")
            return
        
        # Read KiwiSDR config from environment
        host = os.getenv('KIWI_HOST', '')
        port = int(os.getenv('KIWI_PORT', 8073))
        password = os.getenv('KIWI_PASSWORD', '')
        frequency_hz = int(os.getenv('FREQUENCY_HZ', 7200000))
        demod_mode = os.getenv('DEMOD_MODE', os.getenv('MODE', 'USB')).upper()
        
        if not host:
            logger.warning("KIWI_HOST not set, cannot auto-start")
            return
        
        # Auto-start slot 0
        config = {
            'host': host,
            'port': port,
            'password': password,
            'frequency_hz': frequency_hz,
            'mode': demod_mode
        }
        
        logger.info(f"Auto-starting slot 0 from environment: {host}:{port}, {frequency_hz} Hz, {demod_mode}")
        self.start_slot('0', config)

    def handle_command(self, data):
        cmd = data.get('command')
        slot_id = data.get('slot_id')
        
        logger.info(f"Received command: {cmd} for slot {slot_id}")
        
        if cmd == "START":
            config_str = data.get('config')
            try:
                config = json.loads(config_str)
                self.start_slot(slot_id, config)
            except Exception as e:
                logger.error(f"Invalid START config: {e}", exc_info=True)
                
        elif cmd == "STOP":
            self.stop_slot(slot_id)

    def start_slot(self, slot_id, config):
        self.stop_slot(slot_id)  # Ensure clean slate
        
        thread = CaptureThread(slot_id, config)
        thread.daemon = True
        thread.start()
        self.slots[slot_id] = thread
        logger.info(f"Started slot {slot_id}")

    def stop_slot(self, slot_id):
        if slot_id in self.slots:
            logger.info(f"Stopping slot {slot_id}")
            thread = self.slots[slot_id]
            thread.stop()
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning(f"Slot {slot_id} thread did not stop within timeout")
            del self.slots[slot_id]
            logger.info(f"Stopped slot {slot_id}")


if __name__ == "__main__":
    manager = SlotManager()
    try:
        manager.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        # Stop all slots
        for slot_id in list(manager.slots.keys()):
            manager.stop_slot(slot_id)
