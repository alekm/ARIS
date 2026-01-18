
"""
Background persistence worker
"""
import time
import logging
import threading
import json
import redis
from datetime import datetime
from sqlalchemy.orm import Session
from contextlib import contextmanager

from shared.models import (
    Transcript, Callsign, QSO,
    STREAM_TRANSCRIPTS, STREAM_CALLSIGNS, STREAM_QSOS, RedisMessage
)
from shared.db import TranscriptModel, CallsignModel, QSOModel

logger = logging.getLogger(__name__)

class PersistenceWorker(threading.Thread):
    """
    Background thread that consumes from Redis streams and persists to SQLite.
    Uses a separate consumer group 'persistence'.
    """
    def __init__(self, redis_client, db_session_factory):
        super().__init__()
        self.redis = redis_client
        self.SessionLocal = db_session_factory
        self.running = False
        self.consumer_group = 'persistence_service'
        self.consumer_name = f'worker-{int(time.time())}'
        
        # Create consumer group if not exists
        for stream in [STREAM_TRANSCRIPTS, STREAM_CALLSIGNS, STREAM_QSOS]:
            try:
                self.redis.xgroup_create(stream, self.consumer_group, id='0', mkstream=True)
            except redis.exceptions.ResponseError as e:
                if 'BUSYGROUP' not in str(e):
                    logger.warning(f"Error creating group for {stream}: {e}")

    def run(self):
        self.running = True
        logger.info("Persistence worker started")
        
        last_ids = {
            STREAM_TRANSCRIPTS: '>',
            STREAM_CALLSIGNS: '>',
            STREAM_QSOS: '>'
        }

        while self.running:
            try:
                messages = self.redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    last_ids,
                    count=20,
                    block=2000
                )

                if not messages:
                    continue

                with self.get_db() as db:
                    for stream, msg_list in messages:
                        stream = stream.decode('utf-8') if isinstance(stream, bytes) else stream
                        for msg_id, msg_data in msg_list:
                            self.process_message(db, stream, msg_data)
                            self.redis.xack(stream, self.consumer_group, msg_id)
                            
                    db.commit()

            except redis.exceptions.ResponseError as e:
                if "NOGROUP" in str(e):
                    logger.warning(f"Consumer group missing, recreating...")
                    for stream in [STREAM_TRANSCRIPTS, STREAM_CALLSIGNS, STREAM_QSOS]:
                        try:
                            self.redis.xgroup_create(stream, self.consumer_group, id='0', mkstream=True)
                        except Exception:
                            pass
                else:
                    logger.error(f"Persistence loop error: {e}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Persistence loop error: {e}")
                time.sleep(1)

    @contextmanager
    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def process_message(self, db: Session, stream: str, msg_data: dict):
        try:
            if stream == STREAM_TRANSCRIPTS:
                obj = RedisMessage.decode(msg_data, Transcript)
                db_obj = TranscriptModel(
                    timestamp=obj.timestamp,
                    datetime=datetime.fromtimestamp(obj.timestamp),
                    frequency_hz=obj.frequency_hz,
                    mode=obj.mode,
                    text=obj.text,
                    confidence=obj.confidence,
                    duration_ms=obj.duration_ms,
                    language=obj.language
                )
                db.add(db_obj)
                
            elif stream == STREAM_CALLSIGNS:
                obj = RedisMessage.decode(msg_data, Callsign)
                db_obj = CallsignModel(
                    callsign=obj.callsign,
                    timestamp=obj.timestamp,
                    datetime=datetime.fromtimestamp(obj.timestamp),
                    frequency_hz=obj.frequency_hz,
                    confidence=obj.confidence,
                    context=obj.context
                )
                db.add(db_obj)
                
            elif stream == STREAM_QSOS:
                obj = RedisMessage.decode(msg_data, QSO)
                # Check for existing (update logic) or just insert
                # Since session_id is unique, we merge
                existing = db.query(QSOModel).filter(QSOModel.session_id == obj.session_id).first()
                if existing:
                    existing.summary = obj.summary
                    existing.end_time = obj.end_time
                else:
                    db_obj = QSOModel(
                        session_id=obj.session_id,
                        start_time=obj.start_time,
                        end_time=obj.end_time,
                        frequency_hz=obj.frequency_hz,
                        mode=obj.mode,
                        summary=obj.summary,
                        callsigns_list=",".join(obj.callsigns),
                        transcript_ids_list=",".join(map(str, obj.transcript_ids))
                    )
                    db.add(db_obj)

        except Exception as e:
            logger.error(f"Failed to persist message from {stream}: {e}")

    def stop(self):
        self.running = False
