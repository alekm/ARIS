import asyncio
import logging
import os
import json
import redis
import threading
import time
import yaml
import numpy as np

from kiwi_client import KiwiSDRClient

# Logging Setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CaptureService")

# Reduce noise from websockets library
logging.getLogger("websockets.client").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)
logging.getLogger("websockets.protocol").setLevel(logging.WARNING)

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
        self._retry_count = 0  # Track retry attempts for exponential backoff
        
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
        # Start control command checker task
        control_task = None
        
        while self.running:
            try:
                # Reset retry count on successful connection attempt
                self._retry_count = 0
                
                # Update Heartbeat immediately
                self._update_heartbeat()
                
                # Initialize Client
                # Read audio endianness from config (default: big, as per KiwiSDR spec)
                audio_endian = self.config.get('audio_endian', 'big')
                self.client = KiwiSDRClient(
                    host=self.config['host'],
                    port=self.config['port'],
                    password=self.config.get('password', ''),
                    frequency_hz=self.config['frequency_hz'],
                    mode=self.config['mode'],
                    audio_endian=audio_endian
                )
                
                # Start Heartbeat
                last_heartbeat = 0
                
                # Attach Audio Callback
                self.client.on_audio_data = self._handle_audio
                
                # Start control command checker if not already running
                if control_task is None or control_task.done():
                    control_task = asyncio.create_task(self._check_control_commands())
                
                # Connect (Blocking until disconnect)
                await self.client.connect()
                
            except TimeoutError as e:
                logger.warning(f"Slot-{self.slot_id} - Connection timeout: {e}")
                logger.info(f"Slot-{self.slot_id} - This may indicate network issues or KiwiSDR is unreachable")
            except Exception as e:
                logger.error(f"Slot-{self.slot_id} - Error: {e}", exc_info=True)
                
            # If we are strictly stopped, break. If error, retry after delay.
            if not self.running: break
            
            # Exponential backoff for retries (5s, 10s, 20s, max 60s)
            retry_delay = min(5 * (2 ** min(self._retry_count, 3)), 60)
            self._retry_count = getattr(self, '_retry_count', 0) + 1
            logger.info(f"Slot-{self.slot_id} - Retrying in {retry_delay}s (attempt {self._retry_count})...")
            await asyncio.sleep(retry_delay)

    async def _handle_audio(self, audio_bytes, seq):
        """Callback from KiwiSDRClient"""
        try:
            # Ensure audio_bytes is even length (required for 16-bit PCM)
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes[:-1]
            
            # Calculate RMS for debugging (especially for CW)
            if len(audio_bytes) >= 2:
                samples = np.frombuffer(audio_bytes, dtype=np.int16)
                rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0
                if self.client.mode == "CW":
                    logger.info(f"Slot-{self.slot_id} Audio RMS: {rms:.6f} (samples: {len(samples)})")
            
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
                low_cut = 300  # Asymmetric, positive side only (matches kiwiclient default)
                high_cut = 700  # Asymmetric, positive side only (matches kiwiclient default)
            
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
            
            # Update heartbeat on audio activity too (at most once per sec)
            self._update_heartbeat()
            
        except Exception as e:
            logger.error(f"Slot-{self.slot_id} - Audio Publish Error: {e}", exc_info=True)

    async def _check_control_commands(self):
        """Check for frequency/mode update commands for this slot"""
        last_id = '$'
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                # Run blocking Redis call in executor to avoid blocking event loop
                messages = await loop.run_in_executor(
                    None,
                    lambda: self.redis.xread({STREAM_CONTROL: last_id}, count=10, block=1000)
                )
                
                if messages:
                    for stream, msgs in messages:
                        for msg_id, data in msgs:
                            last_id = msg_id
                            
                            # Only process commands for this slot
                            slot_id = data.get(b'slot_id', data.get('slot_id'))
                            if slot_id and str(slot_id) != self.slot_id:
                                continue
                            
                            cmd = data.get(b'command', data.get('command'))
                            if isinstance(cmd, bytes):
                                cmd = cmd.decode('utf-8', errors='ignore')
                            
                            if cmd == 'set_frequency' and self.client and self.client.running:
                                try:
                                    freq_hz_str = data.get(b'frequency_hz', data.get('frequency_hz'))
                                    if isinstance(freq_hz_str, bytes):
                                        freq_hz_str = freq_hz_str.decode('utf-8', errors='ignore')
                                    frequency_hz = int(freq_hz_str)
                                    
                                    logger.info(f"Slot-{self.slot_id} - Received frequency update: {frequency_hz} Hz")
                                    self.config['frequency_hz'] = frequency_hz
                                    await self.client.set_frequency(frequency_hz)
                                except Exception as e:
                                    logger.error(f"Slot-{self.slot_id} - Error updating frequency: {e}")
                            
                            elif cmd == 'set_mode' and self.client and self.client.running:
                                try:
                                    mode_str = data.get(b'mode', data.get('mode'))
                                    if isinstance(mode_str, bytes):
                                        mode_str = mode_str.decode('utf-8', errors='ignore')
                                    
                                    logger.info(f"Slot-{self.slot_id} - Received mode update: {mode_str}")
                                    self.config['mode'] = mode_str
                                    await self.client.set_mode(mode_str)
                                except Exception as e:
                                    logger.error(f"Slot-{self.slot_id} - Error updating mode: {e}")
                            
            except Exception as e:
                # Don't log timeout errors (normal when no messages)
                if "timeout" not in str(e).lower():
                    logger.debug(f"Slot-{self.slot_id} - Control check error: {e}")
                await asyncio.sleep(1)

    def _update_heartbeat(self):
        """Update Redis with current status"""
        try:
            # Rate limit to 1s
            now = time.time()
            if hasattr(self, '_last_hb') and now - self._last_hb < 1.0:
                return
            self._last_hb = now
            
            key = f"slot:{self.slot_id}:activity"
            data = {
                "frequency_hz": self.config['frequency_hz'],
                "mode": self.config.get('mode', 'USB'),
                "host": self.config.get('host'),
                "port": self.config.get('port'),
                "last_seen": now
            }
            self.redis.set(key, json.dumps(data), ex=30)
        except Exception as e:
            logger.error(f"Slot-{self.slot_id} - Heartbeat Error: {e}")

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
        # Only auto-start if explicitly enabled via MODE=kiwi and KIWI_HOST is set
        # Otherwise, slots should be configured via Web UI or API
        auto_start_enabled = os.getenv('MODE', '').lower() in ['kiwi', 'kiwisdr']
        if auto_start_enabled:
            self._auto_start_from_env()
        else:
            logger.info("Auto-start disabled. Configure slots via Web UI or API.")
        
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
        # Read KiwiSDR config from environment
        host = os.getenv('KIWI_HOST', '').strip()
        port_str = os.getenv('KIWI_PORT', '').strip()
        password = os.getenv('KIWI_PASSWORD', '').strip()
        frequency_hz_str = os.getenv('FREQUENCY_HZ', '').strip()
        demod_mode = os.getenv('DEMOD_MODE', 'USB').strip().upper()
        
        # Validate required fields
        if not host:
            logger.info("KIWI_HOST not set, skipping auto-start (configure slots via Web UI instead)")
            return
        
        if not port_str:
            logger.warning("KIWI_PORT not set, using default 8073")
            port = 8073
        else:
            try:
                port = int(port_str)
            except ValueError:
                logger.error(f"Invalid KIWI_PORT value: {port_str}, using default 8073")
                port = 8073
        
        if not frequency_hz_str:
            logger.warning("FREQUENCY_HZ not set, using default 7200000")
            frequency_hz = 7200000
        else:
            try:
                frequency_hz = int(frequency_hz_str)
            except ValueError:
                logger.error(f"Invalid FREQUENCY_HZ value: {frequency_hz_str}, using default 7200000")
                frequency_hz = 7200000
        
        # Auto-start slot 1 (Matches UI/Server indexing 1-4)
        config = {
            'host': host,
            'port': port,
            'password': password,
            'frequency_hz': frequency_hz,
            'mode': demod_mode
        }
        
        logger.info(f"Auto-starting slot 1 from environment variables: {host}:{port}, {frequency_hz} Hz, {demod_mode}")
        logger.info("Note: To disable auto-start, remove MODE=kiwi from environment or unset KIWI_HOST")
        self.start_slot('1', config)

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
