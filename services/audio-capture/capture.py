import asyncio
import logging
import os
import signal
import json
import redis
import threading
import time
from datetime import datetime
import yaml

from kiwi_client import KiwiSDRClient

# Logging Setup
# Logging Setup
logging.basicConfig(
    level=logging.INFO,
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
        


# ... (logging setup) ...

class CaptureThread(threading.Thread):
    """Manages a single KiwiSDR connection in a separate thread/loop"""
    def __init__(self, slot_id, config):
        super().__init__()
        self.slot_id = str(slot_id)
        self.config = config
        self.running = False
import redis

# ... (logging setup) ...

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
                logger.error(f"Slot-{self.slot_id} - Error: {e}")
                
            # If we are strictly stopped, break. If error, retry after delay.
            if not self.running: break
            await asyncio.sleep(5) # Retry delay

    async def _handle_audio(self, audio_bytes, seq):
        """Callback from KiwiSDRClient"""
        try:
            # Ensure audio_bytes is even length (required for 16-bit PCM)
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes[:-1]
                
            # Create AudioChunk metadata
            metadata = {
                "source_id": self.slot_id,
                "timestamp": time.time(),
                "frequency_hz": self.client.frequency_hz,
                "mode": self.client.mode,
                "sample_rate": 12000,
                "duration_ms": (len(audio_bytes) / 2 / 12000) * 1000,
                "seq": seq
            }
            
            # Match Shared Model (AudioChunk) Protocol
            # We must write individual fields, not a binary blob
            
            payload = {
                "source_id": self.slot_id,
                "timestamp": time.time(),
                "frequency_hz": self.client.frequency_hz,
                "mode": self.client.mode,
                "sample_rate": 12000,
                "duration_ms": int((len(audio_bytes) / 2 / 12000) * 1000),
                "seq": seq,
                "data": audio_bytes.hex(), # Convert to hex string as expected by models.py
                # Add optional fields if needed
                "s_meter": 0.0,
                "signal_strength_db": -150.0,
                "squelch_open": "true" 
            }
            
            self.redis.xadd(STREAM_AUDIO, payload, maxlen=1000)
            
        except Exception as e:
            logger.error(f"Slot-{self.slot_id} - Audio Publish Error: {e}")

    def stop(self):
        self.running = False
        if self.client:
            # Schedule stop? Client logic is blocking await.
            # We need to signal the client to stop.
            # Since we are in a different thread, we can't await.
            # Best way: Client checks `self.running` or we cancel connection.
            # For now, simplistic approach:
            asyncio.run_coroutine_threadsafe(self.client.stop(), self.loop)

import struct

class SlotManager:
    def __init__(self):
        self.slots = {} # {slot_id: CaptureThread}
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    def start(self):
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
                logger.error(f"Control Loop Error: {e}")
                time.sleep(1)

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
                logger.error(f"Invalid START config: {e}")
                
        elif cmd == "STOP":
            self.stop_slot(slot_id)

    def start_slot(self, slot_id, config):
        self.stop_slot(slot_id) # Ensure clean slate
        
        thread = CaptureThread(slot_id, config)
        thread.start()
        self.slots[slot_id] = thread
        logger.info(f"Started slot {slot_id}")

    def stop_slot(self, slot_id):
        if slot_id in self.slots:
            logger.info(f"Stopping slot {slot_id}")
            thread = self.slots[slot_id]
            thread.stop()
            thread.join(timeout=2)
            del self.slots[slot_id]

if __name__ == "__main__":
    manager = SlotManager()
    manager.start()
