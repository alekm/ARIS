#!/usr/bin/env python3
"""
Speech-to-Text Service
Uses faster-whisper to transcribe audio chunks from Redis stream.
Supports multiple concurrent audio sources (slots) via source_id tagging.
"""
import os
import sys
import time
import logging
import numpy as np
import redis
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from faster_whisper import WhisperModel
from scipy import signal as scipy_signal

sys.path.insert(0, '/app')
from shared.models import AudioChunk, Transcript, STREAM_AUDIO, STREAM_TRANSCRIPTS, RedisMessage

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SlotState:
    """State for a single audio source slot"""
    source_id: str
    audio_buffer: List[np.ndarray] = field(default_factory=list)
    buffer_duration_ms: int = 0
    in_speech: bool = False
    silence_chunks_count: int = 0
    silence_chunks_needed: int = 0
    last_chunk_info: Dict = field(default_factory=dict)

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

        # Translation configuration
        self.enable_translation = os.getenv('ENABLE_TRANSLATION', 'false').lower() == 'true'
        if self.enable_translation:
            logger.info("Translation mode enabled: will auto-detect language and translate to English")
        
        # VAD parameters
        self.vad_threshold = float(os.getenv('VAD_THRESHOLD', '0.5'))
        self.energy_threshold = float(os.getenv('ENERGY_THRESHOLD', '0.01'))
        self.silence_duration_ms = int(os.getenv('SILENCE_DURATION_MS', '2000'))
        self.max_buffer_ms = int(os.getenv('MAX_BUFFER_MS', '30000'))
        self.min_buffer_ms = int(os.getenv('MIN_BUFFER_MS', '1000'))

        self.running = False
        self.consumer_group = 'stt-service'
        self.consumer_name = f'stt-{os.getpid()}'
        
        # State per slot
        self.slots: Dict[str, SlotState] = {}

        # Create consumer group if it doesn't exist
        try:
            self.redis.xgroup_create(STREAM_AUDIO, self.consumer_group, id='0', mkstream=True)
            logger.info(f"Created consumer group: {self.consumer_group}")
        except redis.exceptions.ResponseError as e:
            if 'BUSYGROUP' not in str(e):
                raise
            logger.info(f"Consumer group already exists: {self.consumer_group}")

    def bytes_to_float32(self, audio_bytes, sample_rate):
        """Convert int16 PCM bytes to float32 numpy array, resampling to 16kHz if needed"""
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        float_audio = pcm.astype(np.float32) / 32768.0
        
        # Whisper expects 16kHz, resample if needed
        target_rate = 16000
        if sample_rate != target_rate:
            # Calculate resampling ratio
            num_samples = int(len(float_audio) * target_rate / sample_rate)
            # Use scipy's resample for high-quality resampling
            float_audio = scipy_signal.resample(float_audio, num_samples)
            logger.debug(f"Resampled audio from {sample_rate}Hz to {target_rate}Hz ({len(pcm)} -> {num_samples} samples)")
        
        return float_audio

    def detect_speech(self, audio_float):
        """Detect if audio chunk contains speech using RMS energy"""
        rms = np.sqrt(np.mean(audio_float ** 2))
        has_speech = rms > self.energy_threshold
        # logger.debug(f"Audio RMS: {rms:.4f}, threshold: {self.energy_threshold:.4f}, speech: {has_speech}")
        return has_speech

    def transcribe_buffer(self, state: SlotState):
        """Transcribe accumulated audio buffer for a slot"""
        if state.buffer_duration_ms < self.min_buffer_ms:
            return None

        # Concatenate all chunks
        audio_data = np.concatenate(state.audio_buffer)

        logger.info(f"[Slot {state.source_id}] Transcribing {state.buffer_duration_ms}ms ({len(audio_data)} samples)")

        try:
            # Whisper expects 16kHz audio
            # Audio should already be resampled to 16kHz in bytes_to_float32()
            # faster-whisper auto-detects sample rate from the audio data
            segments, info = self.model.transcribe(
                audio_data,
                language=None if self.enable_translation else "en",
                task="translate" if self.enable_translation else "transcribe",
                vad_filter=True,
                vad_parameters=dict(threshold=self.vad_threshold),
                beam_size=5
            )

            full_text = ""
            for segment in segments:
                full_text += segment.text + " "
            full_text = full_text.strip()
            logger.info(f"[Slot {state.source_id}] RAW WHISPER OUTPUT: '{full_text}'")

            if full_text:
                # Filter out transcripts that are just punctuation/whitespace
                if self.is_noise_or_punctuation_only(full_text):
                    logger.info(f"[Slot {state.source_id}] Rejecting noise/punctuation-only transcript: '{full_text}'")
                    return None
                
                if self.is_hallucination(full_text):
                    return None
                    
                logger.info(f"[Slot {state.source_id}] Transcribed: {full_text[:100]}...")
                return full_text, info.language_probability
            else:
                return None

        except Exception as e:
            logger.error(f"[Slot {state.source_id}] Transcription error: {e}")
            return None

    def is_noise_or_punctuation_only(self, text):
        """Check if text is only punctuation, whitespace, or noise patterns"""
        # Remove all whitespace and check if only punctuation remains
        text_no_ws = text.replace(" ", "").replace("\t", "").replace("\n", "").strip()
        if not text_no_ws:
            return True
        
        # Check if it's only periods/dots (common noise pattern like ". . . ." or "....")
        # After removing whitespace, check if all characters are periods
        if len(text_no_ws) > 0 and all(c == '.' for c in text_no_ws):
            return True
        
        # Check if it's only punctuation marks (no alphanumeric characters)
        import string
        if all(c in string.punctuation or c.isspace() for c in text):
            return True
        
        # Check for patterns with spaces like ".  .  .  ." - normalize and check
        normalized = text.replace(" ", "").replace("\t", "").replace("\n", "")
        if normalized and all(c == '.' for c in normalized):
            return True
        
        return False

    def is_hallucination(self, text):
        hallucinations = ["Thanks for watching", "subscribe", "MBC", "www.", ".com", "Amara.org"]
        text_lower = text.lower()
        for h in hallucinations:
            if h.lower() in text_lower: return True
            
        # Check for single-word noise hallucinations (common in Whisper)
        # Remove punctuation for the check
        normalized = text.strip().lower().rstrip('.!?')
        noise_words = ["you", "bye", "goodbye"]
        if normalized in noise_words:
            return True
            
        return False

    def publish_transcript(self, text, confidence, chunk_info, source_id):
        """Publish transcript to Redis stream with source_id"""
        transcript = Transcript(
            timestamp=time.time(),
            frequency_hz=chunk_info['frequency_hz'],
            mode=chunk_info['mode'],
            text=text,
            confidence=confidence,
            duration_ms=chunk_info.get('duration_ms', 0), # This is kinda wrong, currently buffer duration isn't passed here
            # But Transcript.duration_ms is used for display. 
            # I should pass state.buffer_duration_ms.
            source_id=source_id,
            language="en"
        )
        # Fix: pass buffer duration
        transcript.duration_ms = chunk_info['buffer_duration_ms']

        msg = RedisMessage.encode(transcript)
        self.redis.xadd(STREAM_TRANSCRIPTS, msg, maxlen=10000)
        logger.info(f"[Slot {source_id}] Published transcript")

    def process_chunk(self, chunk_data):
        """Process a single audio chunk with VAD-triggered transcription"""
        try:
            chunk = RedisMessage.decode(chunk_data, AudioChunk)
            source_id = chunk.source_id
            
            # Get or create state
            if source_id not in self.slots:
                self.slots[source_id] = SlotState(source_id=source_id)
            state = self.slots[source_id]

            audio_float = self.bytes_to_float32(chunk.data, chunk.sample_rate)

            # Silence calc init
            if state.silence_chunks_needed == 0 and chunk.duration_ms > 0:
                state.silence_chunks_needed = max(1, self.silence_duration_ms // chunk.duration_ms)

            # Detect speech
            has_speech = self.detect_speech(audio_float)

            # Add to buffer
            state.audio_buffer.append(audio_float)
            state.buffer_duration_ms += chunk.duration_ms
            
            # Update info
            state.last_chunk_info = {
                'frequency_hz': chunk.frequency_hz,
                'mode': chunk.mode,
                'buffer_duration_ms': state.buffer_duration_ms # For transcript duration
            }

            should_transcribe = False
            transcribe_reason = ""

            if has_speech:
                if not state.in_speech:
                    state.in_speech = True
                state.silence_chunks_count = 0
            else:
                if state.in_speech:
                    state.silence_chunks_count += 1
                    if state.silence_chunks_count >= state.silence_chunks_needed:
                        should_transcribe = True
                        transcribe_reason = "speech_ended"
                        state.in_speech = False
                        state.silence_chunks_count = 0

            if state.buffer_duration_ms >= self.max_buffer_ms:
                should_transcribe = True
                transcribe_reason = "buffer_full"
                state.in_speech = False
                state.silence_chunks_count = 0

            if should_transcribe:
                result = self.transcribe_buffer(state)
                if result:
                    text, confidence = result
                    self.publish_transcript(text, confidence, state.last_chunk_info, source_id)

                # Clear buffer
                state.audio_buffer = []
                state.buffer_duration_ms = 0

        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)

    def run(self):
        self.running = True
        logger.info("Starting STT Multi-Slot Service...")
        last_id = '>'

        try:
            # First, process any pending messages (PEL)
            pending_id = '0'
            logger.info("Checking for pending messages...")
            while self.running:
                messages = self.redis.xreadgroup(
                    self.consumer_group, self.consumer_name,
                    {STREAM_AUDIO: pending_id}, count=20, block=None
                )
                
                if not messages or not messages[0][1]:
                    break # No more pending messages

                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        self.process_chunk(msg_data)
                        self.redis.xack(STREAM_AUDIO, self.consumer_group, msg_id)
            
            logger.info("Processed all pending messages. Listening for new data...")

            # Now listen for new messages
            while self.running:
                messages = self.redis.xreadgroup(
                    self.consumer_group, self.consumer_name,
                    {STREAM_AUDIO: last_id}, count=5, block=1000
                )

                if not messages: continue

                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        self.process_chunk(msg_data)
                        self.redis.xack(STREAM_AUDIO, self.consumer_group, msg_id)
        except KeyboardInterrupt:
            self.running = False

if __name__ == '__main__':
    service = STTService()
    service.run()
