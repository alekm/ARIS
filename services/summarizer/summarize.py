#!/usr/bin/env python3
"""
Summarization Service
Groups transcripts into QSO sessions and generates summaries using LLM
"""
import os
import sys
import time
import logging
import redis
import json
from typing import List, Dict
from datetime import datetime
import ollama
from openai import OpenAI

sys.path.insert(0, '/app')
from shared.models import Transcript, Callsign, QSO, STREAM_TRANSCRIPTS, STREAM_CALLSIGNS, STREAM_QSOS, STREAM_CONTROL, RedisMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SessionManager:
    """Manages QSO sessions and grouping"""

    def __init__(self, gap_threshold_sec=30):
        self.gap_threshold = gap_threshold_sec
        self.active_sessions = {}  # frequency -> session data
        self.transcript_buffer = {}  # frequency -> list of transcripts

    def add_transcript(self, transcript: Transcript) -> bool:
        """
        Add transcript to appropriate session
        Returns: True if session is ready for summarization
        """
        freq = transcript.frequency_hz

        if freq not in self.transcript_buffer:
            self.transcript_buffer[freq] = []

        buffer = self.transcript_buffer[freq]

        # Check if this starts a new session (gap detection)
        if buffer and (transcript.timestamp - buffer[-1].timestamp) > self.gap_threshold:
            # Gap detected - current session is complete
            logger.info(f"Session complete on {freq} Hz ({len(buffer)} transcripts)")
            return True

        buffer.append(transcript)
        return False

    def get_session_transcripts(self, frequency_hz: int) -> List[Transcript]:
        """Get and clear transcripts for a frequency"""
        if frequency_hz in self.transcript_buffer:
            transcripts = self.transcript_buffer[frequency_hz]
            self.transcript_buffer[frequency_hz] = []
            return transcripts
        return []


class LLMSummarizer:
    """Interface to LLM for generating summaries"""

    def __init__(self):
        self.backend = os.getenv('LLM_BACKEND', 'ollama')
        self.model = os.getenv('LLM_MODEL', 'llama3.2:latest')
        self.host = os.getenv('LLM_HOST', 'localhost:11434')
        self.api_key = os.getenv('LLM_API_KEY', 'not-needed')

        logger.info(f"LLM backend: {self.backend}, model: {self.model}, host: {self.host}")

        if self.backend == 'openai':
            self.client = OpenAI(
                base_url=f"http://{self.host}/v1",
                api_key=self.api_key
            )

    def summarize_qso(self, transcripts: List[Transcript], callsigns: List[str]) -> str:
        """Generate a summary of a QSO session"""

        # Build context
        full_text = "\n".join([f"[{datetime.fromtimestamp(t.timestamp).strftime('%H:%M:%S')}] {t.text}"
                               for t in transcripts])

        callsigns_str = ", ".join(callsigns) if callsigns else "unknown stations"

        prompt = f"""You are analyzing amateur radio (ham radio) communications. Below is a transcript from a QSO (conversation) or net.

Callsigns detected: {callsigns_str}

Transcript:
{full_text}

Please provide a concise 2-3 sentence summary covering:
1. Who was involved (callsigns)
2. Main topics discussed
3. Any notable information (antenna issues, band conditions, locations, weather, emergencies, etc.)

Keep it brief and factual. Focus on what was actually discussed."""

        try:
            if self.backend == 'ollama':
                response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.3}
                )
                return response['message']['content'].strip()
            elif self.backend == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            else:
                return f"Summary not available (unknown backend: {self.backend})"

        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return f"[Summary failed: {str(e)}]"


class SummarizerService:
    """Main summarization service"""

    def __init__(self):
        # Connect to Redis
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

        self.session_manager = SessionManager(gap_threshold_sec=30)
        self.summarizer = LLMSummarizer()

        # Cache callsigns by frequency
        self.callsign_cache = {}  # frequency -> list of callsigns

        self.running = False
        self.consumer_group = 'summarizer'
        self.consumer_name = f'summarizer-{os.getpid()}'

        # Create consumer groups
        for stream in [STREAM_TRANSCRIPTS, STREAM_CALLSIGNS, STREAM_CONTROL]:
            try:
                self.redis.xgroup_create(stream, self.consumer_group, id='0', mkstream=True)
            except redis.exceptions.ResponseError as e:
                if 'BUSYGROUP' not in str(e):
                    raise

    def process_transcript(self, transcript_data):
        """Process incoming transcript"""
        try:
            transcript = RedisMessage.decode(transcript_data, Transcript)

            # Add to session manager
            session_ready = self.session_manager.add_transcript(transcript)

            if session_ready:
                # Generate summary for completed session
                self.summarize_session(transcript.frequency_hz, transcript.mode)

        except Exception as e:
            logger.error(f"Error processing transcript: {e}", exc_info=True)

    def process_callsign(self, callsign_data):
        """Cache callsigns by frequency"""
        try:
            callsign = RedisMessage.decode(callsign_data, Callsign)

            freq = callsign.frequency_hz
            if freq not in self.callsign_cache:
                self.callsign_cache[freq] = set()

            self.callsign_cache[freq].add(callsign.callsign)

        except Exception as e:
            logger.error(f"Error processing callsign: {e}", exc_info=True)

    def process_control_command(self, control_data):
        """Process control commands"""
        try:
            # Decode control message
            command_dict = {}
            for key, value in control_data.items():
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                val_str = value.decode('utf-8') if isinstance(value, bytes) else value
                command_dict[key_str] = val_str

            command = command_dict.get('command', '')

            if command == 'trigger_summarize':
                logger.info("Manual summarization trigger received")
                # Summarize all active sessions immediately
                for frequency_hz in list(self.session_manager.transcript_buffer.keys()):
                    transcripts = self.session_manager.transcript_buffer.get(frequency_hz, [])
                    if transcripts:
                        mode = transcripts[0].mode
                        logger.info(f"Force summarizing {frequency_hz} Hz ({len(transcripts)} transcripts)")
                        self.summarize_session(frequency_hz, mode)

        except Exception as e:
            logger.error(f"Error processing control command: {e}", exc_info=True)

    def summarize_session(self, frequency_hz: int, mode: str):
        """Generate summary for a completed session"""
        transcripts = self.session_manager.get_session_transcripts(frequency_hz)

        if not transcripts:
            return

        # Get callsigns for this frequency
        callsigns = list(self.callsign_cache.get(frequency_hz, []))

        logger.info(f"Summarizing session: {len(transcripts)} transcripts, {len(callsigns)} callsigns")

        # Generate summary
        summary = self.summarizer.summarize_qso(transcripts, callsigns)

        # Create QSO object
        qso = QSO(
            session_id=f"{frequency_hz}_{int(transcripts[0].timestamp)}",
            start_time=transcripts[0].timestamp,
            end_time=transcripts[-1].timestamp,
            frequency_hz=frequency_hz,
            mode=mode,
            callsigns=callsigns,
            transcript_ids=[],  # Would store DB IDs in production
            summary=summary
        )

        # Publish to Redis
        msg = RedisMessage.encode(qso)
        self.redis.xadd(STREAM_QSOS, msg, maxlen=1000)

        logger.info(f"Published QSO summary: {summary[:100]}...")

        # Clear callsign cache for this frequency
        if frequency_hz in self.callsign_cache:
            del self.callsign_cache[frequency_hz]

    def run(self):
        """Main processing loop"""
        self.running = True
        logger.info("Starting summarization service...")

        last_transcript_id = '>'
        last_callsign_id = '>'
        last_control_id = '>'

        try:
            while self.running:
                # Read from all streams
                messages = self.redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {
                        STREAM_TRANSCRIPTS: last_transcript_id,
                        STREAM_CALLSIGNS: last_callsign_id,
                        STREAM_CONTROL: last_control_id
                    },
                    count=5,
                    block=1000
                )

                if not messages:
                    continue

                for stream_name, stream_messages in messages:
                    stream_name = stream_name.decode('utf-8') if isinstance(stream_name, bytes) else stream_name

                    for msg_id, msg_data in stream_messages:
                        if stream_name == STREAM_TRANSCRIPTS:
                            self.process_transcript(msg_data)
                        elif stream_name == STREAM_CALLSIGNS:
                            self.process_callsign(msg_data)
                        elif stream_name == STREAM_CONTROL:
                            self.process_control_command(msg_data)

                        self.redis.xack(stream_name, self.consumer_group, msg_id)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            self.running = False

    def stop(self):
        self.running = False


if __name__ == '__main__':
    service = SummarizerService()
    service.run()
