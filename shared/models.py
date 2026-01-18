"""
Shared data models for ARIS (Amateur Radio Intelligence System) pipeline
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
    s_meter: float = 0.0  # S-meter reading (0.0 to 9.9)
    signal_strength_db: float = -150.0  # Signal strength in dB
    squelch_open: bool = True  # Whether squelch threshold was met
    rssi: Optional[float] = None  # RSSI from KiwiSDR if available
    low_cut: Optional[int] = None  # Filter low cutoff in Hz
    high_cut: Optional[int] = None  # Filter high cutoff in Hz

    def to_dict(self):
        d = asdict(self)
        d['data'] = self.data.hex()  # Convert bytes to hex for JSON
        return d

    @classmethod
    def from_dict(cls, d):
        if 'data' not in d:
            raise ValueError(f"Missing 'data' field in dict. Available keys: {list(d.keys())}")
        # Convert data from hex string to bytes
        d['data'] = bytes.fromhex(d['data'])
        # Convert numeric fields from strings to proper types
        d['timestamp'] = float(d['timestamp']) if isinstance(d['timestamp'], str) else d['timestamp']
        d['frequency_hz'] = int(d['frequency_hz']) if isinstance(d['frequency_hz'], str) else d['frequency_hz']
        d['sample_rate'] = int(d['sample_rate']) if isinstance(d['sample_rate'], str) else d['sample_rate']
        d['duration_ms'] = int(d['duration_ms']) if isinstance(d['duration_ms'], str) else d['duration_ms']
        # Optional fields with defaults
        d['s_meter'] = float(d.get('s_meter', 0.0)) if isinstance(d.get('s_meter'), str) else d.get('s_meter', 0.0)
        d['signal_strength_db'] = float(d.get('signal_strength_db', -150.0)) if isinstance(d.get('signal_strength_db'), str) else d.get('signal_strength_db', -150.0)
        d['squelch_open'] = d.get('squelch_open', True) if not isinstance(d.get('squelch_open'), str) else d.get('squelch_open', 'true').lower() == 'true'
        # Handle rssi - convert from string if non-empty, otherwise use None
        rssi_val = d.get('rssi')
        if rssi_val is not None and isinstance(rssi_val, str):
            d['rssi'] = float(rssi_val) if rssi_val.strip() else None
        else:
            d['rssi'] = rssi_val
        # Handle low_cut and high_cut
        low_cut_val = d.get('low_cut')
        d['low_cut'] = int(low_cut_val) if low_cut_val is not None and isinstance(low_cut_val, str) and low_cut_val.strip() else low_cut_val
        high_cut_val = d.get('high_cut')
        d['high_cut'] = int(high_cut_val) if high_cut_val is not None and isinstance(high_cut_val, str) and high_cut_val.strip() else high_cut_val
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
        # Convert numeric fields from strings to proper types
        d['timestamp'] = float(d['timestamp']) if isinstance(d['timestamp'], str) else d['timestamp']
        d['frequency_hz'] = int(d['frequency_hz']) if isinstance(d['frequency_hz'], str) else d['frequency_hz']
        d['confidence'] = float(d['confidence']) if isinstance(d['confidence'], str) else d['confidence']
        d['duration_ms'] = int(d['duration_ms']) if isinstance(d['duration_ms'], str) else d['duration_ms']
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
        # Convert numeric fields from strings to proper types
        d['timestamp'] = float(d['timestamp']) if isinstance(d['timestamp'], str) else d['timestamp']
        d['frequency_hz'] = int(d['frequency_hz']) if isinstance(d['frequency_hz'], str) else d['frequency_hz']
        d['confidence'] = float(d['confidence']) if isinstance(d['confidence'], str) else d['confidence']
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
        # Convert numeric fields from strings to proper types
        d['start_time'] = float(d['start_time']) if isinstance(d['start_time'], str) else d['start_time']
        if d.get('end_time') is not None:
            d['end_time'] = float(d['end_time']) if isinstance(d['end_time'], str) else d['end_time']
        d['frequency_hz'] = int(d['frequency_hz']) if isinstance(d['frequency_hz'], str) else d['frequency_hz']
        # Convert transcript_ids if they're strings
        if 'transcript_ids' in d and d['transcript_ids']:
            if isinstance(d['transcript_ids'], list) and len(d['transcript_ids']) > 0:
                if isinstance(d['transcript_ids'][0], str):
                    d['transcript_ids'] = [int(x) for x in d['transcript_ids']]
        return cls(**d)


class RedisMessage:
    """Helper for Redis stream messages"""

    @staticmethod
    def encode(obj) -> dict:
        """Convert dataclass to Redis-compatible dict"""
        if hasattr(obj, 'to_dict'):
            result = {}
            for k, v in obj.to_dict().items():
                # Check bool first (since bool is a subclass of int)
                if isinstance(v, bool):
                    result[k] = 'true' if v else 'false'  # Convert bool to string
                elif isinstance(v, (str, int, float)):
                    result[k] = v
                elif isinstance(v, bytes):
                    result[k] = v.hex()  # Convert bytes to hex string
                elif v is None:
                    result[k] = ''  # Convert None to empty string
                else:
                    result[k] = json.dumps(v)  # JSON encode other types
            return result
        return obj

    @staticmethod
    def decode(data: dict, cls):
        """Convert Redis dict back to dataclass"""
        import logging
        logger = logging.getLogger(__name__)
        
        decoded = {}
        for k, v in data.items():
            try:
                # Convert key to string if it's bytes (Redis returns bytes keys)
                key = k.decode('utf-8') if isinstance(k, bytes) else k
                
                # Convert value to string if it's bytes
                if isinstance(v, bytes):
                    v = v.decode('utf-8')
                
                # Try to JSON parse, but if it fails, use the value as-is
                # This handles both JSON-encoded values and plain strings (like hex)
                if isinstance(v, str):
                    # Hex strings (like audio data) are very long and not valid JSON
                    # Only try JSON parsing for short strings that might be JSON
                    # For very long strings, skip the strip() check to avoid performance issues
                    if len(v) > 1000:
                        decoded[key] = v
                    elif not v.strip().startswith(('{', '[', '"')):
                        decoded[key] = v
                    else:
                        try:
                            decoded[key] = json.loads(v)
                        except (json.JSONDecodeError, ValueError):
                            decoded[key] = v
                else:
                    decoded[key] = v
            except Exception as e:
                logger.error(f"Error processing key {k}: {e}")
                # Still try to include it
                key = k.decode('utf-8') if isinstance(k, bytes) else k
                decoded[key] = v if not isinstance(v, bytes) else v.decode('utf-8', errors='ignore')
        
        return cls.from_dict(decoded)


# Redis stream names
STREAM_AUDIO = "audio:chunks"
STREAM_TRANSCRIPTS = "transcripts"
STREAM_CALLSIGNS = "callsigns"
STREAM_QSOS = "qsos"
STREAM_CONTROL = "control:audio-capture"  # Control commands for audio capture service
