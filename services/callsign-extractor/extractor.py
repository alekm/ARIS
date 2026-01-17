#!/usr/bin/env python3
"""
Callsign Extraction Service
Extracts amateur radio callsigns from transcripts using regex and phonetics
"""
import os
import sys
import re
import logging
import redis
from typing import List, Tuple

sys.path.insert(0, '/app')
from shared.models import Transcript, Callsign, STREAM_TRANSCRIPTS, STREAM_CALLSIGNS, RedisMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CallsignExtractor:
    """Extract amateur radio callsigns from text"""

    # Phonetic alphabet mapping
    PHONETICS = {
        'alpha': 'A', 'alfa': 'A',
        'bravo': 'B',
        'charlie': 'C',
        'delta': 'D',
        'echo': 'E',
        'foxtrot': 'F',
        'golf': 'G',
        'hotel': 'H',
        'india': 'I',
        'juliet': 'J', 'juliett': 'J',
        'kilo': 'K',
        'lima': 'L',
        'mike': 'M',
        'november': 'N',
        'oscar': 'O',
        'papa': 'P',
        'quebec': 'Q',
        'romeo': 'R',
        'sierra': 'S',
        'tango': 'T',
        'uniform': 'U',
        'victor': 'V',
        'whiskey': 'W',
        'xray': 'X', 'x-ray': 'X',
        'yankee': 'Y',
        'zulu': 'Z',
        'zero': '0',
        'one': '1', 'wun': '1',
        'two': '2', 'too': '2',
        'three': '3', 'tree': '3',
        'four': '4', 'fower': '4',
        'five': '5', 'fife': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9', 'niner': '9'
    }

    def __init__(self):
        # US/Canadian callsign patterns
        # Format: 1-2 letter prefix, digit, 1-3 letter suffix
        self.callsign_pattern = re.compile(
            r'\b([A-Z]{1,2}\d[A-Z]{1,3})\b',
            re.IGNORECASE
        )

        # Common false positives to filter
        self.blacklist = {
            'A1', 'B2', 'C3', 'D4', 'E5',  # Too short
            'TEST', 'QSO', 'CQ', 'FM', 'AM', 'USB', 'LSB'  # Common ham terms
        }

    def convert_phonetics(self, text: str) -> str:
        """Convert phonetic alphabet to letters"""
        words = text.lower().split()
        converted = []

        for word in words:
            if word in self.PHONETICS:
                converted.append(self.PHONETICS[word])
            else:
                converted.append(word)

        return ' '.join(converted)

    def extract_callsigns(self, text: str) -> List[Tuple[str, float, str]]:
        """
        Extract callsigns from text
        Returns: List of (callsign, confidence, context)
        """
        results = []

        # First, try direct regex on text
        matches = self.callsign_pattern.finditer(text.upper())
        for match in matches:
            callsign = match.group(1)
            if self.is_valid_callsign(callsign):
                context = text[max(0, match.start()-20):min(len(text), match.end()+20)]
                results.append((callsign, 0.9, context))

        # Convert phonetics and try again
        phonetic_text = self.convert_phonetics(text)
        phonetic_matches = self.callsign_pattern.finditer(phonetic_text.upper())
        for match in phonetic_matches:
            callsign = match.group(1)
            if self.is_valid_callsign(callsign):
                # Check if we already found this one
                if not any(r[0] == callsign for r in results):
                    context = text[max(0, match.start()-20):min(len(text), match.end()+20)]
                    results.append((callsign, 0.7, context))  # Lower confidence for phonetic

        return results

    def is_valid_callsign(self, callsign: str) -> bool:
        """Validate callsign against rules and blacklist"""
        if callsign in self.blacklist:
            return False

        # Must be 3-6 characters
        if len(callsign) < 3 or len(callsign) > 6:
            return False

        # Must contain at least one letter and one digit
        has_letter = any(c.isalpha() for c in callsign)
        has_digit = any(c.isdigit() for c in callsign)

        if not (has_letter and has_digit):
            return False

        # US/Canadian callsigns start with K, W, N, A, or VE
        if not callsign[0] in ['K', 'W', 'N', 'A', 'V']:
            return False

        return True


class CallsignExtractionService:
    """Service that reads transcripts and extracts callsigns"""

    def __init__(self):
        # Connect to Redis
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

        self.extractor = CallsignExtractor()

        self.running = False
        self.consumer_group = 'callsign-extractor'
        self.consumer_name = f'extractor-{os.getpid()}'

        # Create consumer group
        try:
            self.redis.xgroup_create(STREAM_TRANSCRIPTS, self.consumer_group, id='0', mkstream=True)
            logger.info(f"Created consumer group: {self.consumer_group}")
        except redis.exceptions.ResponseError as e:
            if 'BUSYGROUP' not in str(e):
                raise
            logger.info(f"Consumer group already exists: {self.consumer_group}")

    def process_transcript(self, transcript_data):
        """Process a single transcript and extract callsigns"""
        try:
            transcript = RedisMessage.decode(transcript_data, Transcript)

            logger.debug(f"Processing transcript: {transcript.text[:50]}...")

            # Extract callsigns
            callsigns = self.extractor.extract_callsigns(transcript.text)

            if callsigns:
                logger.info(f"Found {len(callsigns)} callsign(s): {[c[0] for c in callsigns]}")

                # Publish each callsign
                for callsign_str, confidence, context in callsigns:
                    callsign = Callsign(
                        callsign=callsign_str,
                        timestamp=transcript.timestamp,
                        frequency_hz=transcript.frequency_hz,
                        confidence=confidence,
                        context=context
                    )

                    msg = RedisMessage.encode(callsign)
                    self.redis.xadd(STREAM_CALLSIGNS, msg, maxlen=10000)

                    logger.info(f"Published callsign: {callsign_str} (confidence: {confidence:.2f})")

        except Exception as e:
            logger.error(f"Error processing transcript: {e}", exc_info=True)

    def run(self):
        """Main processing loop"""
        self.running = True
        logger.info("Starting callsign extraction service...")

        last_id = '>'

        try:
            while self.running:
                messages = self.redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {STREAM_TRANSCRIPTS: last_id},
                    count=1,
                    block=1000
                )

                if not messages:
                    continue

                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        self.process_transcript(msg_data)
                        self.redis.xack(STREAM_TRANSCRIPTS, self.consumer_group, msg_id)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            self.running = False

    def stop(self):
        self.running = False


if __name__ == '__main__':
    service = CallsignExtractionService()
    service.run()
