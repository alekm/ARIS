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

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logging.basicConfig(
    level=log_level,
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

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approx 4 chars per token)"""
        return len(text) // 4

    def summarize_qso(self, transcripts: List[Transcript], callsigns: List[str]) -> str:
        """
        Generate a summary of a QSO session.
        Uses recursive summarization for long sessions to fit context window.
        """
        # Constants for context management (assuming ~4k-8k context window)
        # 3000 tokens * 4 chars = 12000 chars safe limit per chunk
        MAX_CHARS_PER_CHUNK = 12000 
        
        # Calculate total size
        full_text_combined = "".join([t.text for t in transcripts])
        total_len = len(full_text_combined)

        # Base case: fit within limit
        if total_len <= MAX_CHARS_PER_CHUNK:
            return self._generate_summary(transcripts, callsigns)

        # Recursive case: Chunking
        logger.info(f"Session too long ({total_len} chars), performing recursive summarization...")
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for t in transcripts:
            t_len = len(t.text)
            if current_len + t_len > MAX_CHARS_PER_CHUNK and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_len = 0
            
            current_chunk.append(t)
            current_len += t_len
            
        if current_chunk:
            chunks.append(current_chunk)
            
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Map: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}...")
            # Detect callsigns local to this chunk (optional optimization, passing all for now)
            summary = self._generate_summary(chunk, callsigns, is_partial=True)
            chunk_summaries.append(f"[Part {i+1}] {summary}")
            
        # Reduce: Final summary of summaries
        logger.info("Generating final summary from chunks...")
        return self._generate_final_summary(chunk_summaries, callsigns)

    def _generate_summary(self, transcripts: List[Transcript], callsigns: List[str], is_partial: bool = False) -> str:
        """Internal method to call LLM for a specific batch of transcripts"""
        
        full_text = "\n".join([f"[{datetime.fromtimestamp(t.timestamp).strftime('%H:%M:%S')}] {t.text}"
                               for t in transcripts])

        callsigns_str = ", ".join(callsigns) if callsigns else "unknown stations"
        
        context_type = "segment of a" if is_partial else ""

        prompt = f"""Summarize this amateur radio QSO transcript in 3-5 sentences. Write directly, without an introduction.

Callsigns: {callsigns_str}

Transcript:
{full_text}

Summary (3-5 sentences covering participants, main topics, and notable details like signal reports or locations):"""

        return self._call_llm(prompt)

    def _generate_final_summary(self, summaries: List[str], callsigns: List[str]) -> str:
        """Generate final summary from a list of partial summaries"""
        
        combined_summaries = "\n\n".join(summaries)
        callsigns_str = ", ".join(callsigns)

        prompt = f"""Synthesize these partial summaries into a single 4-6 sentence summary of the entire amateur radio session. Write directly without an introduction.

Callsigns: {callsigns_str}

Partial Summaries:
{combined_summaries}

Consolidated Summary (4-6 sentences covering main participants, key discussion points, and operational details):"""
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        """Helper to call the configured LLM backend"""
        start_time = time.time()
        try:
            # Use system message to encourage direct, natural summaries
            system_msg = "You are a concise technical writer. Write summaries directly without introductions or meta-commentary."
            timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "60"))

            if self.backend == 'ollama':
                # Set host for ollama client if not default
                if self.host and self.host != 'localhost:11434':
                    os.environ['OLLAMA_HOST'] = f"http://{self.host}"
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': system_msg},
                        {'role': 'user', 'content': prompt}
                    ],
                    options={'temperature': 0.3},
                    timeout=timeout_sec,
                )
                result = response['message']['content'].strip()
            elif self.backend == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': system_msg},
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=0.3,
                    timeout=timeout_sec,
                )
                result = response.choices[0].message.content.strip()
            else:
                return f"Summary not available (unknown backend: {self.backend})"

            latency_ms = (time.time() - start_time) * 1000.0
            logger.info(f"LLM summarization completed in {latency_ms:.1f}ms using backend={self.backend}")

            # Persist simple latency metrics to Redis for external monitoring
            try:
                # Avoid circular import at module load time
                r_host = os.getenv('REDIS_HOST', 'localhost')
                r_port = int(os.getenv('REDIS_PORT', 6379))
                r = redis.Redis(host=r_host, port=r_port, decode_responses=False)
                metrics = {
                    "last_latency_ms": f"{latency_ms:.1f}",
                    "last_backend": self.backend,
                    "last_model": self.model,
                    "last_updated": f"{time.time():.3f}",
                }
                r.hset("metrics:summarizer", mapping=metrics)
            except Exception as metrics_err:
                logger.debug(f"Failed to update summarizer metrics: {metrics_err}")

            # Post-process: Remove common boilerplate prefixes
            boilerplate_prefixes = [
                "Here's a summary",
                "Here is a summary",
                "Summary:",
                "The summary is:",
                "This is a summary"
            ]
            for prefix in boilerplate_prefixes:
                if result.startswith(prefix):
                    # Remove prefix and any following punctuation/whitespace
                    result = result[len(prefix):].lstrip(' :\n-')
                    break

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000.0
            logger.error(f"LLM summarization failed after {latency_ms:.1f}ms: {e}")
            return f"[Summary failed: {str(e)}]"


class SummarizerService:
    """Main summarization service"""

    def __init__(self):
        # Connect to Redis
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

        self.session_manager = SessionManager(gap_threshold_sec=60)
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
            # If add_transcript returns True, it means a gap was detected and the CURRENT transcript was NOT added.
            session_ready = self.session_manager.add_transcript(transcript)

            if session_ready:
                # Generate summary for completed (OLD) session
                self.summarize_session(transcript.frequency_hz, transcript.mode)
                
                # Add the CURRENT transcript to the new (now empty) buffer
                self.session_manager.add_transcript(transcript)

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

            if command == 'regenerate_qso':
                session_id = command_dict.get('session_id')
                if session_id:
                    logger.info(f"Regenerate QSO command received for session {session_id}")
                    self.regenerate_qso_from_db(session_id)
                else:
                    logger.warning("regenerate_qso command missing session_id")
            elif command == 'trigger_summarize':
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

    def regenerate_qso_from_db(self, session_id: str):
        """Regenerate summary for an existing QSO from database"""
        try:
            # Import database models
            from shared.db import init_db, QSOModel, TranscriptModel
            import os
            database_url = os.getenv('DATABASE_URL', 'sqlite:////data/db/aris.db')
            SessionLocal = init_db(database_url)
            session = SessionLocal()
            
            try:
                # Get QSO from database
                qso = session.query(QSOModel).filter(QSOModel.session_id == session_id).first()
                if not qso:
                    logger.error(f"QSO {session_id} not found in database")
                    return
                
                # Get associated transcripts
                buffer_sec = 2.0
                end_time = qso.end_time if qso.end_time else time.time()
                
                transcripts_db = session.query(TranscriptModel).filter(
                    TranscriptModel.frequency_hz == qso.frequency_hz,
                    TranscriptModel.timestamp >= qso.start_time - buffer_sec,
                    TranscriptModel.timestamp <= end_time + buffer_sec
                ).order_by(TranscriptModel.timestamp.asc()).all()
                
                if not transcripts_db:
                    logger.error(f"No transcripts found for QSO {session_id}")
                    return
                
                # Convert to Transcript objects
                transcripts = [
                    Transcript(
                        timestamp=t.timestamp,
                        frequency_hz=t.frequency_hz,
                        mode=t.mode,
                        text=t.text,
                        confidence=t.confidence,
                        duration_ms=t.duration_ms,
                        source_id=str(t.frequency_hz),
                        language=t.language
                    )
                    for t in transcripts_db
                ]
                
                # Get callsigns
                callsigns_list = qso.callsigns_list.split(',') if qso.callsigns_list else []
                callsigns = [c.strip() for c in callsigns_list if c.strip()]
                
                # Regenerate summary
                logger.info(f"Regenerating summary for QSO {session_id} with {len(transcripts)} transcripts")
                new_summary = self.summarizer.summarize_qso(transcripts, callsigns)
                
                # Update QSO in database
                qso.summary = new_summary
                session.commit()
                
                logger.info(f"Summary regenerated for QSO {session_id}: {new_summary[:100]}...")
                
                # Also publish updated QSO to Redis stream
                updated_qso = QSO(
                    session_id=qso.session_id,
                    start_time=qso.start_time,
                    end_time=qso.end_time,
                    frequency_hz=qso.frequency_hz,
                    mode=qso.mode,
                    callsigns=callsigns,
                    transcript_ids=[],
                    summary=new_summary
                )
                msg = RedisMessage.encode(updated_qso)
                self.redis.xadd(STREAM_QSOS, msg, maxlen=1000)
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error regenerating QSO from database: {e}", exc_info=True)

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
