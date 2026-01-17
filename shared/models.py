"""
Shared data models for HamEars pipeline
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import json


@dataclass
class AudioChunk:
    """Raw audio data from capture service"""
    timestamp: float
    frequency_hz: int
    mode: str  # USB, LSB, FM, AM, CW
    sample_rate: int
    duration_ms: int
    data: bytes  # PCM audio data

    def to_dict(self):
        d = asdict(self)
        d['data'] = self.data.hex()  # Convert bytes to hex for JSON
        return d

    @classmethod
    def from_dict(cls, d):
        d['data'] = bytes.fromhex(d['data'])
        return cls(**d)


@dataclass
class Transcript:
    """STT output"""
    timestamp: float
    frequency_hz: int
    mode: str
    text: str
    confidence: float
    duration_ms: int
    language: str = "en"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class Callsign:
    """Extracted callsign"""
    callsign: str
    timestamp: float
    frequency_hz: int
    confidence: float
    context: str  # Surrounding text

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class QSO:
    """QSO/conversation session"""
    session_id: str
    start_time: float
    end_time: Optional[float]
    frequency_hz: int
    mode: str
    callsigns: List[str]
    transcript_ids: List[int]
    summary: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class RedisMessage:
    """Helper for Redis stream messages"""

    @staticmethod
    def encode(obj) -> dict:
        """Convert dataclass to Redis-compatible dict"""
        if hasattr(obj, 'to_dict'):
            return {k: json.dumps(v) if not isinstance(v, (str, int, float, bytes)) else v
                    for k, v in obj.to_dict().items()}
        return obj

    @staticmethod
    def decode(data: dict, cls):
        """Convert Redis dict back to dataclass"""
        decoded = {}
        for k, v in data.items():
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            try:
                decoded[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                decoded[k] = v
        return cls.from_dict(decoded)


# Redis stream names
STREAM_AUDIO = "audio:chunks"
STREAM_TRANSCRIPTS = "transcripts"
STREAM_CALLSIGNS = "callsigns"
STREAM_QSOS = "qsos"
