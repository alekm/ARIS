#!/usr/bin/env python3
"""
Audio Capture Service
Captures audio from KiwiSDR (or mock source) and streams to Redis
"""
import os
import sys
import time
import logging
import numpy as np
import soundfile as sf
import redis
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/app')
from shared.models import AudioChunk, STREAM_AUDIO, RedisMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockAudioSource:
    """Mock audio source for testing without KiwiSDR"""

    def __init__(self, config):
        self.frequency = config.get('frequency_hz', 7200000)
        self.mode = config.get('mode', 'USB')
        self.sample_rate = config.get('sample_rate', 16000)
        self.chunk_duration = config.get('chunk_duration_ms', 1000)
        self.chunk_samples = int(self.sample_rate * self.chunk_duration / 1000)
        logger.info(f"Mock source: {self.frequency} Hz, {self.mode}, {self.sample_rate} Hz")

    def generate_test_tone(self, duration_ms):
        """Generate a test tone (simulated voice frequencies)"""
        samples = int(self.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, samples)

        # Mix of tones in voice range (300-3000 Hz)
        signal = (
            0.3 * np.sin(2 * np.pi * 500 * t) +
            0.2 * np.sin(2 * np.pi * 1200 * t) +
            0.1 * np.sin(2 * np.pi * 2500 * t)
        )

        # Add some noise to simulate RF
        noise = np.random.normal(0, 0.05, samples)
        signal = signal + noise

        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8

        # Convert to int16 PCM
        pcm = (signal * 32767).astype(np.int16)
        return pcm.tobytes()

    def read_chunk(self):
        """Return a chunk of mock audio"""
        audio_data = self.generate_test_tone(self.chunk_duration)

        return AudioChunk(
            timestamp=time.time(),
            frequency_hz=self.frequency,
            mode=self.mode,
            sample_rate=self.sample_rate,
            duration_ms=self.chunk_duration,
            data=audio_data
        )


class KiwiSDRAudioSource:
    """Real KiwiSDR audio source"""

    def __init__(self, config):
        self.host = config['kiwi_host']
        self.port = config.get('kiwi_port', 8073)
        self.frequency = config['frequency_hz']
        self.mode = config.get('mode', 'USB')
        self.sample_rate = config.get('sample_rate', 12000)
        self.chunk_duration = config.get('chunk_duration_ms', 1000)

        logger.info(f"KiwiSDR source: {self.host}:{self.port}, {self.frequency} Hz, {self.mode}")

        # TODO: Implement actual KiwiSDR connection
        # Will use kiwirecorder.py or custom WebSocket client
        raise NotImplementedError("KiwiSDR connection not yet implemented - use MODE=mock")

    def read_chunk(self):
        """Return a chunk from KiwiSDR"""
        # TODO: Read from KiwiSDR WebSocket stream
        pass


class AudioCaptureService:
    """Main audio capture service"""

    def __init__(self, config_path='config.yaml'):
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Connect to Redis
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

        # Initialize audio source
        mode = os.getenv('MODE', self.config.get('mode', 'mock'))

        if mode == 'mock':
            self.source = MockAudioSource(self.config)
        elif mode == 'kiwi':
            self.source = KiwiSDRAudioSource(self.config)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.running = False

    def publish_chunk(self, chunk: AudioChunk):
        """Publish audio chunk to Redis stream"""
        try:
            msg = RedisMessage.encode(chunk)
            self.redis.xadd(STREAM_AUDIO, msg, maxlen=1000)  # Keep last 1000 chunks
            logger.debug(f"Published chunk: {chunk.timestamp}, {len(chunk.data)} bytes")
        except Exception as e:
            logger.error(f"Failed to publish chunk: {e}")

    def run(self):
        """Main capture loop"""
        self.running = True
        logger.info("Starting audio capture service...")

        chunk_count = 0
        start_time = time.time()

        try:
            while self.running:
                chunk = self.source.read_chunk()
                self.publish_chunk(chunk)

                chunk_count += 1
                if chunk_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = chunk_count / elapsed
                    logger.info(f"Captured {chunk_count} chunks ({rate:.1f}/sec)")

                # Sleep to maintain real-time rate
                time.sleep(chunk.duration_ms / 1000 * 0.9)  # Slight underrun

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            self.running = False

    def stop(self):
        self.running = False


if __name__ == '__main__':
    service = AudioCaptureService()
    service.run()
