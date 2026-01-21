"""
Database models and initialization for ARIS
"""
import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import StaticPool

Base = declarative_base()
logger = logging.getLogger(__name__)

class TranscriptModel(Base):
    """DB model for Transcript"""
    __tablename__ = 'transcripts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float, index=True)
    datetime = Column(DateTime, default=datetime.utcnow, index=True)
    frequency_hz = Column(Integer, index=True)
    mode = Column(String(10))
    text = Column(Text)
    confidence = Column(Float)
    duration_ms = Column(Integer)
    language = Column(String(10), default='en')
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "datetime": self.timestamp, # Legacy compat
            "frequency_hz": self.frequency_hz,
            "mode": self.mode,
            "text": self.text,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "language": self.language
        }

class CallsignModel(Base):
    """DB model for Callsign"""
    __tablename__ = 'callsigns'

    id = Column(Integer, primary_key=True, autoincrement=True)
    callsign = Column(String(20), index=True)
    timestamp = Column(Float, index=True)
    datetime = Column(DateTime, default=datetime.utcnow)
    frequency_hz = Column(Integer)
    confidence = Column(Float)
    context = Column(Text)

class QSOModel(Base):
    """DB model for QSO Session"""
    __tablename__ = 'qsos'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, index=True)
    start_time = Column(Float, index=True)
    end_time = Column(Float)
    frequency_hz = Column(Integer)
    mode = Column(String(10))
    summary = Column(Text)
    # Storing lists as JSON-like strings or referenced tables is better, but for simplicity:
    callsigns_list = Column(Text) # Comma separated
    transcript_ids_list = Column(Text) # Comma separated IDs (optional reference)
    transcript_ids_list = Column(Text) # Comma separated IDs (optional reference)
    created_at = Column(DateTime, default=datetime.utcnow)


# Additional indexes for common query patterns
Index('idx_transcripts_freq_time', TranscriptModel.frequency_hz, TranscriptModel.timestamp)
Index('idx_callsigns_freq_time', CallsignModel.frequency_hz, CallsignModel.timestamp)
Index('idx_qsos_freq_start', QSOModel.frequency_hz, QSOModel.start_time)

class SlotModel(Base):
    """DB model for Receiver Slot Configuration"""
    __tablename__ = 'slots'

    id = Column(Integer, primary_key=True, autoincrement=False) # 1-4 determined by user/system
    enabled = Column(Boolean, default=False)
    config_json = Column(Text) # JSON string of full configuration
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def init_db(database_url):
    """Initialize database and create tables"""
    try:
        if database_url.startswith('sqlite'):
            # SQLite optimization for write performance
             engine = create_engine(
                database_url, 
                connect_args={"check_same_thread": False}, 
                poolclass=StaticPool
            )
        else:
            engine = create_engine(database_url)
            
        Base.metadata.create_all(engine)
        logger.info(f"Database initialized: {database_url}")
        return sessionmaker(bind=engine)
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
