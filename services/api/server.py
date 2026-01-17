#!/usr/bin/env python3
"""
HamEars API Server
Provides REST API and web UI for browsing transcripts, callsigns, and QSOs
"""
import os
import sys
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import redis
from pydantic import BaseModel

sys.path.insert(0, '/app')
from shared.models import (
    Transcript, Callsign, QSO,
    STREAM_TRANSCRIPTS, STREAM_CALLSIGNS, STREAM_QSOS,
    RedisMessage
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HamEars API",
    description="Amateur Radio Monitoring and Intelligence System",
    version="0.1.0"
)

# Connect to Redis
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

logger.info(f"Connected to Redis at {redis_host}:{redis_port}")


# Pydantic models for API responses
class TranscriptResponse(BaseModel):
    timestamp: float
    datetime: str
    frequency_hz: int
    mode: str
    text: str
    confidence: float
    duration_ms: int


class CallsignResponse(BaseModel):
    callsign: str
    timestamp: float
    datetime: str
    frequency_hz: int
    confidence: float
    context: str


class QSOResponse(BaseModel):
    session_id: str
    start_time: float
    end_time: Optional[float]
    start_datetime: str
    end_datetime: Optional[str]
    frequency_hz: int
    mode: str
    callsigns: List[str]
    summary: Optional[str]


@app.get("/")
async def root():
    """Simple web UI"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HamEars - Ham Radio Intelligence</title>
        <style>
            body {
                font-family: 'Courier New', monospace;
                background: #1a1a1a;
                color: #00ff00;
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 { color: #00ff00; border-bottom: 2px solid #00ff00; }
            h2 { color: #00aa00; }
            .card {
                background: #2a2a2a;
                border: 1px solid #00ff00;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            .callsign { color: #ffaa00; font-weight: bold; }
            .frequency { color: #00aaff; }
            .timestamp { color: #888; }
            a { color: #00ff00; }
            button {
                background: #00ff00;
                color: #1a1a1a;
                border: none;
                padding: 10px 20px;
                font-family: 'Courier New', monospace;
                cursor: pointer;
                margin: 5px;
            }
            button:hover { background: #00aa00; }
        </style>
    </head>
    <body>
        <h1>📻 HamEars - Ham Radio Intelligence System</h1>

        <div class="card">
            <h2>System Status</h2>
            <p>API is running. Use the endpoints below to access data.</p>
        </div>

        <div class="card">
            <h2>API Endpoints</h2>
            <ul>
                <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                <li><a href="/api/transcripts">/api/transcripts</a> - Recent transcripts</li>
                <li><a href="/api/callsigns">/api/callsigns</a> - Recent callsigns detected</li>
                <li><a href="/api/qsos">/api/qsos</a> - Recent QSO summaries</li>
                <li><a href="/api/stats">/api/stats</a> - System statistics</li>
            </ul>
        </div>

        <div class="card">
            <h2>Quick Actions</h2>
            <button onclick="loadTranscripts()">Load Recent Transcripts</button>
            <button onclick="loadCallsigns()">Load Recent Callsigns</button>
            <button onclick="loadQSOs()">Load QSO Summaries</button>
        </div>

        <div id="results"></div>

        <script>
            async function loadTranscripts() {
                const response = await fetch('/api/transcripts?limit=10');
                const data = await response.json();
                displayResults('Transcripts', data);
            }

            async function loadCallsigns() {
                const response = await fetch('/api/callsigns?limit=20');
                const data = await response.json();
                displayResults('Callsigns', data);
            }

            async function loadQSOs() {
                const response = await fetch('/api/qsos?limit=10');
                const data = await response.json();
                displayResults('QSO Summaries', data);
            }

            function displayResults(title, data) {
                const div = document.getElementById('results');
                div.innerHTML = `<div class="card"><h2>${title}</h2><pre>${JSON.stringify(data, null, 2)}</pre></div>`;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = {
            "transcripts_count": redis_client.xlen(STREAM_TRANSCRIPTS),
            "callsigns_count": redis_client.xlen(STREAM_CALLSIGNS),
            "qsos_count": redis_client.xlen(STREAM_QSOS),
            "uptime": "N/A"  # TODO: Track service uptime
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transcripts", response_model=List[TranscriptResponse])
async def get_transcripts(
    limit: int = Query(default=50, le=500),
    frequency: Optional[int] = None
):
    """Get recent transcripts"""
    try:
        # Read from Redis stream (most recent first)
        messages = redis_client.xrevrange(STREAM_TRANSCRIPTS, count=limit)

        results = []
        for msg_id, msg_data in messages:
            transcript = RedisMessage.decode(msg_data, Transcript)

            # Filter by frequency if specified
            if frequency and transcript.frequency_hz != frequency:
                continue

            results.append(TranscriptResponse(
                timestamp=transcript.timestamp,
                datetime=datetime.fromtimestamp(transcript.timestamp).isoformat(),
                frequency_hz=transcript.frequency_hz,
                mode=transcript.mode,
                text=transcript.text,
                confidence=transcript.confidence,
                duration_ms=transcript.duration_ms
            ))

        return results

    except Exception as e:
        logger.error(f"Error getting transcripts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/callsigns", response_model=List[CallsignResponse])
async def get_callsigns(
    limit: int = Query(default=50, le=500),
    callsign: Optional[str] = None
):
    """Get recent callsigns"""
    try:
        messages = redis_client.xrevrange(STREAM_CALLSIGNS, count=limit)

        results = []
        for msg_id, msg_data in messages:
            cs = RedisMessage.decode(msg_data, Callsign)

            # Filter by callsign if specified
            if callsign and cs.callsign != callsign.upper():
                continue

            results.append(CallsignResponse(
                callsign=cs.callsign,
                timestamp=cs.timestamp,
                datetime=datetime.fromtimestamp(cs.timestamp).isoformat(),
                frequency_hz=cs.frequency_hz,
                confidence=cs.confidence,
                context=cs.context
            ))

        return results

    except Exception as e:
        logger.error(f"Error getting callsigns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qsos", response_model=List[QSOResponse])
async def get_qsos(
    limit: int = Query(default=20, le=100)
):
    """Get recent QSO summaries"""
    try:
        messages = redis_client.xrevrange(STREAM_QSOS, count=limit)

        results = []
        for msg_id, msg_data in messages:
            qso = RedisMessage.decode(msg_data, QSO)

            results.append(QSOResponse(
                session_id=qso.session_id,
                start_time=qso.start_time,
                end_time=qso.end_time,
                start_datetime=datetime.fromtimestamp(qso.start_time).isoformat(),
                end_datetime=datetime.fromtimestamp(qso.end_time).isoformat() if qso.end_time else None,
                frequency_hz=qso.frequency_hz,
                mode=qso.mode,
                callsigns=qso.callsigns,
                summary=qso.summary
            ))

        return results

    except Exception as e:
        logger.error(f"Error getting QSOs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search/callsign/{callsign}")
async def search_by_callsign(callsign: str):
    """Search all data for a specific callsign"""
    callsign = callsign.upper()

    # Get all callsign detections
    callsign_detections = await get_callsigns(limit=500, callsign=callsign)

    # Get related QSOs
    qsos = await get_qsos(limit=100)
    related_qsos = [qso for qso in qsos if callsign in qso.callsigns]

    return {
        "callsign": callsign,
        "detections": callsign_detections,
        "qsos": related_qsos
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
