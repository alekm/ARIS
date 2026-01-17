#!/usr/bin/env python3
"""
Speech-to-Text Service
Uses faster-whisper to transcribe audio chunks from Redis stream
"""
import os
import sys
import time
import logging
import numpy as np
import redis
from faster_whisper import WhisperModel

sys.path.insert(0, '/app')
from shared.models import AudioChunk, Transcript, STREAM_AUDIO, STREAM_TRANSCRIPTS, RedisMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class STTService:
    """Speech-to-Text service using faster-whisper"""

    def __init__(self):
        # Connect to Redis
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

        # Initialize Whisper model
        model_size = os.getenv('MODEL_SIZE', 'medium')
        device = os.getenv('DEVICE', 'cuda')
        compute_type = "float16" if device == "cuda" else "int8"

        logger.info(f"Loading Whisper model: {model_size} on {device}")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root="/data/models"
        )
        logger.info("Model loaded successfully")

        # VAD parameters
        self.vad_threshold = float(os.getenv('VAD_THRESHOLD', '0.5'))

        # Buffer for accumulating audio
        self.audio_buffer = []
        self.buffer_duration_ms = 0
        self.max_buffer_ms = int(os.getenv('MAX_BUFFER_MS', '30000'))  # 30 seconds max
        self.min_buffer_ms = int(os.getenv('MIN_BUFFER_MS', '1000'))  # 1 second min

        self.running = False
        self.consumer_group = 'stt-service'
        self.consumer_name = f'stt-{os.getpid()}'

        # Create consumer group if it doesn't exist
        try:
            self.redis.xgroup_create(STREAM_AUDIO, self.consumer_group, id='0', mkstream=True)
            logger.info(f"Created consumer group: {self.consumer_group}")
        except redis.exceptions.ResponseError as e:
            if 'BUSYGROUP' not in str(e):
                raise
            logger.info(f"Consumer group already exists: {self.consumer_group}")

    def bytes_to_float32(self, audio_bytes, sample_rate):
        """Convert int16 PCM bytes to float32 numpy array"""
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        float_audio = pcm.astype(np.float32) / 32768.0
        return float_audio

    def transcribe_buffer(self):
        """Transcribe accumulated audio buffer"""
        if self.buffer_duration_ms < self.min_buffer_ms:
            logger.debug(f"Buffer too short: {self.buffer_duration_ms}ms")
            return None

        # Concatenate all chunks
        audio_data = np.concatenate(self.audio_buffer)

        logger.info(f"Transcribing {self.buffer_duration_ms}ms of audio ({len(audio_data)} samples)")

        try:
            # Transcribe
            segments, info = self.model.transcribe(
                audio_data,
                language="en",
                vad_filter=True,
                vad_parameters=dict(threshold=self.vad_threshold),
                beam_size=5
            )

            # Collect all segments
            full_text = ""
            segment_count = 0
            for segment in segments:
                full_text += segment.text + " "
                segment_count += 1

            full_text = full_text.strip()

            if full_text:
                logger.info(f"Transcribed ({segment_count} segments): {full_text[:100]}...")
                return full_text, info.language_probability
            else:
                logger.debug("No speech detected in buffer")
                return None

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def publish_transcript(self, text, confidence, chunk_info):
        """Publish transcript to Redis stream"""
        transcript = Transcript(
            timestamp=time.time(),
            frequency_hz=chunk_info['frequency_hz'],
            mode=chunk_info['mode'],
            text=text,
            confidence=confidence,
            duration_ms=self.buffer_duration_ms,
            language="en"
        )

        msg = RedisMessage.encode(transcript)
        self.redis.xadd(STREAM_TRANSCRIPTS, msg, maxlen=10000)
        logger.info(f"Published transcript: {text[:50]}...")

    def process_chunk(self, chunk_data):
        """Process a single audio chunk"""
        try:
            chunk = RedisMessage.decode(chunk_data, AudioChunk)

            # Convert to float32
            audio_float = self.bytes_to_float32(chunk.data, chunk.sample_rate)

            # Add to buffer
            self.audio_buffer.append(audio_float)
            self.buffer_duration_ms += chunk.duration_ms

            # Store chunk info for transcript metadata
            self.last_chunk_info = {
                'frequency_hz': chunk.frequency_hz,
                'mode': chunk.mode
            }

            # Transcribe if buffer is getting full
            if self.buffer_duration_ms >= self.max_buffer_ms:
                result = self.transcribe_buffer()
                if result:
                    text, confidence = result
                    self.publish_transcript(text, confidence, self.last_chunk_info)

                # Clear buffer
                self.audio_buffer = []
                self.buffer_duration_ms = 0

        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)

    def run(self):
        """Main processing loop"""
        self.running = True
        logger.info("Starting STT service...")

        last_id = '>'  # Read only new messages

        try:
            while self.running:
                # Read from stream
                messages = self.redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {STREAM_AUDIO: last_id},
                    count=1,
                    block=1000  # 1 second timeout
                )

                if not messages:
                    continue

                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        self.process_chunk(msg_data)

                        # Acknowledge message
                        self.redis.xack(STREAM_AUDIO, self.consumer_group, msg_id)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            self.running = False

    def stop(self):
        self.running = False


if __name__ == '__main__':
    service = STTService()
    service.run()
