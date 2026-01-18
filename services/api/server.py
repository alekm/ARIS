#!/usr/bin/env python3
"""
ARIS API Server
Provides REST API and web UI for browsing transcripts, callsigns, and QSOs
"""
import os
import sys
import time
import logging
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
from pydantic import BaseModel
import io
import numpy as np
import wave
import struct
import re
from collections import defaultdict

sys.path.insert(0, '/app')
from shared.models import (
    Transcript, Callsign, QSO, AudioChunk,
    STREAM_AUDIO, STREAM_TRANSCRIPTS, STREAM_CALLSIGNS, STREAM_QSOS,
    STREAM_CONTROL, RedisMessage
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ARIS API",
    description="Amateur Radio Intelligence System",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication (optional)
API_KEY = os.getenv('API_KEY', '')
security = HTTPBearer(auto_error=False)

# Rate limiting
rate_limit_store = defaultdict(list)
RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 10))
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 60))  # seconds

def check_rate_limit(request: Request):
    """Simple rate limiting by IP address"""
    if not API_KEY:  # Only enforce if API key is set
        return True
    
    client_ip = request.client.host if request.client else "unknown"
    now = datetime.now()
    
    # Clean old entries
    rate_limit_store[client_ip] = [
        ts for ts in rate_limit_store[client_ip]
        if (now - ts).total_seconds() < RATE_LIMIT_WINDOW
    ]
    
    # Check limit
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds"
        )
    
    rate_limit_store[client_ip].append(now)
    return True

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if configured"""
    if not API_KEY:
        return True  # No API key required
    
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


from background import PersistenceWorker
from shared.db import init_db, TranscriptModel, CallsignModel, QSOModel

# Connect to Redis
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

# Initialize Database
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:////data/db/aris.db')
SessionLocal = init_db(DATABASE_URL)

# Persistence Worker (Global ref)
persistence_worker = None

@app.on_event("startup")
async def startup_event():
    global persistence_worker
    logger.info("Starting persistence worker...")
    persistence_worker = PersistenceWorker(redis_client, SessionLocal)
    persistence_worker.start()

@app.on_event("shutdown")
async def shutdown_event():
    if persistence_worker:
        logger.info("Stopping persistence worker...")
        persistence_worker.stop()
        persistence_worker.join()

logger.info(f"Connected to Redis at {redis_host}:{redis_port}")


# Pydantic models for API responses
class TranscriptResponse(BaseModel):
    id: Optional[int] = None
    timestamp: float
    datetime: str
    frequency_hz: int
    mode: str
    text: str
    confidence: float
    duration_ms: int

# ... existing code ...

@app.delete("/api/transcripts", dependencies=[Depends(check_rate_limit)])
async def clear_transcripts():
    """Clear all transcripts from Database and Redis"""
    session = SessionLocal()
    try:
        # Clear DB
        session.query(TranscriptModel).delete()
        session.commit()
        
        # Clear Redis stream too (optional but good for consistency)
        # Use xtrim instead of delete to preserve consumer groups
        try:
            redis_client.xtrim(STREAM_TRANSCRIPTS, maxlen=0)
        except Exception:
            pass # Ignore if stream doesn't exist
        
        logger.info("Cleared all transcripts from DB and Redis")
        return {"status": "success", "message": "All transcripts cleared"}
    except Exception as e:
        session.rollback()
        logger.error(f"Error clearing transcripts: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@app.delete("/api/transcripts/{transcript_id}", dependencies=[Depends(check_rate_limit)])
async def delete_transcript(transcript_id: int):
    """Delete a specific transcript"""
    session = SessionLocal()
    try:
        transcript = session.query(TranscriptModel).filter(TranscriptModel.id == transcript_id).first()
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found")
            
        session.delete(transcript)
        session.commit()
        return {"status": "success", "message": f"Transcript {transcript_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting transcript {transcript_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.get("/api/transcripts", response_model=List[TranscriptResponse])
async def get_transcripts(
    limit: int = Query(default=50, le=500),
    frequency: Optional[int] = None
):
    """Get recent transcripts from Database"""
    session = SessionLocal()
    try:
        query = session.query(TranscriptModel)
        
        if frequency:
            query = query.filter(TranscriptModel.frequency_hz == frequency)
            
        # Get most recent
        transcripts = query.order_by(TranscriptModel.timestamp.desc()).limit(limit).all()

        results = []
        for t in transcripts:
            results.append(TranscriptResponse(
                id=t.id,
                timestamp=t.timestamp,
                datetime=t.datetime.isoformat(),
                frequency_hz=t.frequency_hz,
                mode=t.mode,
                text=t.text,
                confidence=t.confidence,
                duration_ms=t.duration_ms
            ))
            
        return results
    except Exception as e:
        logger.error(f"Error reading transcripts: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


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
        <title>ARIS - Amateur Radio Intelligence System</title>
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
        <h1>📻 ARIS - Amateur Radio Intelligence System</h1>

        <div class="card">
            <h2>System Status</h2>
            <p>API is running. Use the endpoints below to access data.</p>
        </div>

        <div class="card">
            <h2>API Endpoints</h2>
            <h3 style="color: #00aaff; margin-top: 10px;">Documentation & UI</h3>
            <ul>
                <li><a href="/docs">/docs</a> - Interactive API documentation (Swagger UI)</li>
                <li><a href="/api/monitor">/api/monitor</a> - Real-time monitoring dashboard</li>
            </ul>
            
            <h3 style="color: #00aaff; margin-top: 10px;">Data Endpoints (GET)</h3>
            <ul>
                <li><a href="/api/stats">/api/stats</a> - System statistics</li>
                <li><a href="/api/transcripts">/api/transcripts</a> - Recent transcripts (query: ?limit=50&frequency=7200000)</li>
                <li><a href="/api/callsigns">/api/callsigns</a> - Recent callsigns detected (query: ?limit=50&callsign=W1AW)</li>
                <li><a href="/api/qsos">/api/qsos</a> - Recent QSO summaries (query: ?limit=20)</li>
            </ul>
            
            <h3 style="color: #00aaff; margin-top: 10px;">Search Endpoints (GET)</h3>
            <ul>
                <li><a href="/api/search/callsign/W1AW">/api/search/callsign/{callsign}</a> - Search by callsign</li>
            </ul>
            
            <h3 style="color: #00aaff; margin-top: 10px;">Control Endpoints (POST)</h3>
            <ul>
                <li><code>POST /api/control/start</code> - Start/resume audio capture</li>
                <li><code>POST /api/control/stop</code> - Stop/pause audio capture</li>
                <li><code>POST /api/control/frequency</code> - Change receiver frequency (body: {"frequency_hz": 14313000})</li>
                <li><code>POST /api/control/mode</code> - Change demodulation mode (body: {"mode": "USB"})</li>
                <li><code>POST /api/control/summarize</code> - Manually trigger QSO summarization</li>
            </ul>
            
            <h3 style="color: #00aaff; margin-top: 10px;">Audio Endpoints (GET)</h3>
            <ul>
                <li><a href="/api/audio/latest">/api/audio/latest</a> - Download latest audio chunk as WAV</li>
                <li><code>/api/audio/chunk/{chunk_id}</code> - Download specific audio chunk as WAV</li>
            </ul>
        </div>

        <div class="card">
            <h2>Receiver Control</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                <div>
                    <label for="frequency" style="display: block; margin-bottom: 5px;">Frequency (kHz):</label>
                    <input type="number" id="frequency" placeholder="e.g., 7188" step="0.001" style="width: 100%; padding: 8px; background: #1a1a1a; border: 1px solid #00ff00; color: #00ff00; font-family: 'Courier New', monospace;">
                    <button onclick="changeFrequency()" style="margin-top: 10px; width: 100%;">Change Frequency</button>
                    <div id="frequency-status" style="margin-top: 5px; font-size: 0.9em;"></div>
                </div>
                <div>
                    <label for="mode" style="display: block; margin-bottom: 5px;">Mode:</label>
                    <select id="mode" style="width: 100%; padding: 8px; background: #1a1a1a; border: 1px solid #00ff00; color: #00ff00; font-family: 'Courier New', monospace;">
                        <option value="LSB">LSB</option>
                        <option value="USB">USB</option>
                        <option value="AM">AM</option>
                        <option value="FM">FM</option>
                    </select>
                    <button onclick="changeMode()" style="margin-top: 10px; width: 100%;">Change Mode</button>
                    <div id="mode-status" style="margin-top: 5px; font-size: 0.9em;"></div>
                </div>
            </div>
            <div style="margin-top: 15px; padding: 10px; background: #1a1a1a; border: 1px solid #00aa00; border-radius: 3px;">
                <strong>Current Settings:</strong> <span id="current-settings">Loading...</span>
            </div>
        </div>

        <div class="card">
            <h2>Capture Control</h2>
            <button onclick="startCapture()" style="background: #00ff00;">▶️ Start Capture</button>
            <button onclick="stopCapture()" style="background: #ffaa00;">⏸️ Stop Capture</button>
            <div id="capture-status" style="margin-top: 10px; font-size: 0.9em;"></div>
        </div>

        <div class="card">
            <h2>Quick Actions</h2>
            <button onclick="loadTranscripts()">Load Recent Transcripts</button>
            <button onclick="loadCallsigns()">Load Recent Callsigns</button>
            <button onclick="loadQSOs()">Load QSO Summaries</button>
            <button onclick="triggerSummarize()">🤖 Trigger QSO Summarization</button>
            <button onclick="clearTranscripts()" style="background: #ff0000; color: #ffffff;">⚠️ Clear All Transcripts</button>
            <button onclick="window.location.href='/api/monitor'">📊 Real-Time Monitor</button>
            <button onclick="window.location.href='/docs'">📖 API Docs</button>
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
                displayResults('Recent Callsigns', data);
            }

            async function loadQSOs() {
                const response = await fetch('/api/qsos?limit=10');
                const data = await response.json();
                displayResults('QSO Summaries', data);
            }

            async function clearTranscripts() {
                if (!confirm('Are you sure you want to clear ALL transcripts? This cannot be undone.')) {
                    return;
                }

                try {
                    const response = await fetch('/api/transcripts', {
                        method: 'DELETE'
                    });
                    const result = await response.json();
                    alert(result.message);
                    // Reload if transcripts are currently displayed
                    const resultsDiv = document.getElementById('results');
                    if (resultsDiv.innerHTML.includes('Transcripts')) {
                         loadTranscripts();
                    }
                } catch (error) {
                    alert('Error clearing transcripts: ' + error.message);
                }
            }

            async function triggerSummarize() {
                try {
                    const response = await fetch('/api/control/summarize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });

                    const result = await response.json();

                    if (response.ok) {
                        alert('✓ ' + result.message);
                        // Reload QSOs if they're currently displayed
                        const resultsDiv = document.getElementById('results');
                        if (resultsDiv.innerHTML.includes('QSO Summaries')) {
                            setTimeout(loadQSOs, 2000); // Wait 2s for summarization
                        }
                    } else {
                        alert('✗ Error: ' + (result.detail || 'Unknown error'));
                    }
                } catch (error) {
                    alert('✗ Error: ' + error.message);
                }
            }

            async function startCapture() {
                const statusDiv = document.getElementById('capture-status');
                statusDiv.innerHTML = '<span style="color: #ffaa00;">Starting capture...</span>';

                try {
                    const response = await fetch('/api/control/start', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });

                    const result = await response.json();

                    if (response.ok) {
                        statusDiv.innerHTML = '<span style="color: #00ff00;">✓ ' + result.message + '</span>';
                        setTimeout(() => statusDiv.innerHTML = '', 3000);
                    } else {
                        statusDiv.innerHTML = '<span style="color: #ff0000;">✗ Error: ' + (result.detail || 'Unknown error') + '</span>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<span style="color: #ff0000;">✗ Error: ' + error.message + '</span>';
                }
            }

            async function stopCapture() {
                const statusDiv = document.getElementById('capture-status');
                statusDiv.innerHTML = '<span style="color: #ffaa00;">Stopping capture...</span>';

                try {
                    const response = await fetch('/api/control/stop', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });

                    const result = await response.json();

                    if (response.ok) {
                        statusDiv.innerHTML = '<span style="color: #ffaa00;">⏸️ ' + result.message + '</span>';
                        setTimeout(() => statusDiv.innerHTML = '', 3000);
                    } else {
                        statusDiv.innerHTML = '<span style="color: #ff0000;">✗ Error: ' + (result.detail || 'Unknown error') + '</span>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<span style="color: #ff0000;">✗ Error: ' + error.message + '</span>';
                }
            }

            async function deleteTranscript(id) {
                if (!confirm('Delete this transcript?')) return;
                try {
                    const response = await fetch(`/api/transcripts/${id}`, { method: 'DELETE' });
                    if (response.ok) {
                        loadTranscripts(); // Reload table
                    } else {
                        alert('Failed to delete');
                    }
                } catch (e) {
                    alert('Error: ' + e.message);
                }
            }

            function displayResults(title, data) {
                const div = document.getElementById('results');
                
                if (title === 'Transcripts' && Array.isArray(data)) {
                    let html = `<div class="card"><h2>${title}</h2>`;
                    html += '<table style="width: 100%; border-collapse: collapse; margin-top: 10px;">';
                    html += '<tr style="background: #333; color: #00ff00; text-align: left;">';
                    html += '<th style="padding: 8px;">Time</th><th style="padding: 8px;">Freq</th><th style="padding: 8px;">Mode</th><th style="padding: 8px;">Conf</th><th style="padding: 8px;">Text</th><th style="padding: 8px;">Actions</th></tr>';
                    
                    data.forEach(item => {
                        html += '<tr style="border-bottom: 1px solid #333;">';
                        html += `<td style="padding: 8px; white-space: nowrap;">${item.datetime.split('T')[1].split('.')[0]}</td>`;
                        html += `<td style="padding: 8px;">${(item.frequency_hz / 1000).toFixed(1)}</td>`;
                        html += `<td style="padding: 8px;">${item.mode}</td>`;
                        html += `<td style="padding: 8px;">${(item.confidence * 100).toFixed(0)}%</td>`;
                        html += `<td style="padding: 8px; color: #fff; white-space: normal; word-wrap: break-word;">${item.text}</td>`;
                        html += `<td style="padding: 8px;"><button onclick="deleteTranscript(${item.id})" style="background: #cc0000; padding: 2px 6px; font-size: 0.8em;">Del</button></td>`;
                        html += '</tr>';
                    });
                    
                    html += '</table></div>';
                    div.innerHTML = html;
                } else if (title === 'QSO Summaries' && Array.isArray(data)) {
                    let html = `<div class="card"><h2>${title}</h2>`;
                    html += '<table style="width: 100%; border-collapse: collapse; margin-top: 10px;">';
                    html += '<tr style="background: #333; color: #00ff00; text-align: left;">';
                    html += '<th style="padding: 8px;">Time</th><th style="padding: 8px;">Freq</th><th style="padding: 8px;">Mode</th><th style="padding: 8px;">Callsigns</th><th style="padding: 8px;">Summary</th></tr>';
                    
                    data.forEach(item => {
                        html += '<tr style="border-bottom: 1px solid #333;">';
                        // Handle start_time being float timestamp or ISO string depending on API
                        // API usually returns processed objects. Let's assume standardized format or check if it needs conversion.
                        // Assuming item has .datetime or .start_time. Let's check model.
                        // QSO model usually has start_time (float). But API might serialize it.
                        // Let's assume standard 'datetime' field added by API or format start_time.
                        // Actually, let's check what API returns for QSOS. 
                        // It returns list of QSO. QSO has start_time (float). 
                        // I'll format assuming it's available as ISO or I can convert.
                        // Wait, transcript used item.datetime.
                        // I'll assume I need to handle timestamp if datetime field isn't present.
                        // But wait, the python API usually converts to Pydantic models.
                        // checking get_qsos implementation might be wise, but I'll use robust formatting.
                        
                        let timeStr = item.datetime || new Date(item.start_time * 1000).toISOString();
                        timeStr = timeStr.split('T')[1].split('.')[0];

                        html += `<td style="padding: 8px; white-space: nowrap;">${timeStr}</td>`;
                        html += `<td style="padding: 8px;">${(item.frequency_hz / 1000).toFixed(1)}</td>`;
                        html += `<td style="padding: 8px;">${item.mode}</td>`;
                        html += `<td style="padding: 8px; color: #aaa;">${item.callsigns.join(', ')}</td>`;
                        html += `<td style="padding: 8px; color: #fff; white-space: normal; word-wrap: break-word;">${item.summary || 'No summary'}</td>`;
                        html += '</tr>';
                    });
                    
                    html += '</table></div>';
                    div.innerHTML = html;
                } else if (title === 'Recent Callsigns' && Array.isArray(data)) {
                    let html = `<div class="card"><h2>${title}</h2>`;
                    html += '<table style="width: 100%; border-collapse: collapse; margin-top: 10px;">';
                    html += '<tr style="background: #333; color: #00ff00; text-align: left;">';
                    html += '<th style="padding: 8px;">Time</th><th style="padding: 8px;">Freq</th><th style="padding: 8px;">Callsign</th><th style="padding: 8px;">Context</th><th style="padding: 8px;">Conf</th></tr>';
                    
                    data.forEach(item => {
                        html += '<tr style="border-bottom: 1px solid #333;">';
                        
                        // Handle timestamp
                        let timeStr = item.datetime || new Date(item.timestamp * 1000).toISOString();
                        timeStr = timeStr.split('T')[1].split('.')[0];

                        html += `<td style="padding: 8px; white-space: nowrap;">${timeStr}</td>`;
                        html += `<td style="padding: 8px;">${(item.frequency_hz / 1000).toFixed(1)}</td>`;
                        html += `<td style="padding: 8px; font-weight: bold; color: #00ffff;">${item.callsign}</td>`;
                        html += `<td style="padding: 8px; color: #ccc; font-style: italic;">"${item.context || ''}"</td>`;
                        html += `<td style="padding: 8px;">${(item.confidence * 100).toFixed(0)}%</td>`;
                        html += '</tr>';
                    });
                    
                    html += '</table></div>';
                    div.innerHTML = html;
                } else {
                    div.innerHTML = `<div class="card"><h2>${title}</h2><pre>${JSON.stringify(data, null, 2)}</pre></div>`;
                }
            }

            async function changeFrequency() {
                const frequencyInput = document.getElementById('frequency');
                const frequency = parseFloat(frequencyInput.value);
                const statusDiv = document.getElementById('frequency-status');

                if (!frequency || frequency < 100 || frequency > 6000000) {
                    statusDiv.innerHTML = '<span style="color: #ff0000;">Invalid frequency (100 kHz - 6000 MHz)</span>';
                    return;
                }

                statusDiv.innerHTML = '<span style="color: #ffaa00;">Changing frequency...</span>';

                try {
                    const response = await fetch('/api/control/frequency', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ frequency_khz: frequency })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        statusDiv.innerHTML = `<span style="color: #00ff00;">✓ ${result.message}</span>`;
                        // Wait a bit for new chunks to be published, then update settings
                        setTimeout(() => {
                            updateCurrentSettings();
                            // Check again after another second to ensure we got the new mode
                            setTimeout(updateCurrentSettings, 1000);
                        }, 1500);
                    } else {
                        statusDiv.innerHTML = `<span style="color: #ff0000;">✗ Error: ${result.detail || 'Unknown error'}</span>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<span style="color: #ff0000;">✗ Error: ${error.message}</span>`;
                }
            }

            async function changeMode() {
                const modeSelect = document.getElementById('mode');
                const mode = modeSelect.value;
                const statusDiv = document.getElementById('mode-status');
                
                statusDiv.innerHTML = '<span style="color: #ffaa00;">Changing mode...</span>';
                
                try {
                    const response = await fetch('/api/control/mode', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ mode: mode })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        statusDiv.innerHTML = `<span style="color: #00ff00;">✓ ${result.message}</span>`;
                        // Wait a bit for new chunks to be published, then update settings
                        setTimeout(() => {
                            updateCurrentSettings();
                            // Check again after another second to ensure we got the new mode
                            setTimeout(updateCurrentSettings, 1000);
                        }, 1500);
                    } else {
                        statusDiv.innerHTML = `<span style="color: #ff0000;">✗ Error: ${result.detail || 'Unknown error'}</span>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<span style="color: #ff0000;">✗ Error: ${error.message}</span>`;
                }
            }

            async function updateCurrentSettings() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    
                    if (stats.recent_audio) {
                        const freq = stats.recent_audio.frequency_hz;
                        const mode = stats.recent_audio.mode;
                        const freqMHz = (freq / 1000000).toFixed(3);
                        document.getElementById('current-settings').textContent = `${freqMHz} MHz (${mode})`;

                        // Update the frequency input with current value (in kHz)
                        document.getElementById('frequency').value = (freq / 1000).toFixed(3);
                        document.getElementById('mode').value = mode;
                    } else {
                        document.getElementById('current-settings').textContent = 'No audio data available';
                    }
                } catch (error) {
                    document.getElementById('current-settings').textContent = 'Error loading settings';
                }
            }

            // Load current settings on page load
            updateCurrentSettings();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        # Get stream lengths
        audio_count = redis_client.xlen(STREAM_AUDIO)
        transcripts_count = redis_client.xlen(STREAM_TRANSCRIPTS)
        callsigns_count = redis_client.xlen(STREAM_CALLSIGNS)
        qsos_count = redis_client.xlen(STREAM_QSOS)
        
        # Get most recent audio chunk timestamp to check if audio is flowing
        recent_audio = None
        if audio_count > 0:
            try:
                messages = redis_client.xrevrange(STREAM_AUDIO, count=1)
                if messages:
                    msg_id, msg_data = messages[0]
                    chunk = RedisMessage.decode(msg_data, AudioChunk)
                    recent_audio = {
                        "last_chunk_time": chunk.timestamp,
                        "last_chunk_datetime": datetime.fromtimestamp(chunk.timestamp).isoformat(),
                        "frequency_hz": chunk.frequency_hz,
                        "mode": chunk.mode,
                        "sample_rate": chunk.sample_rate,
                        "duration_ms": chunk.duration_ms,
                        "s_meter": getattr(chunk, 's_meter', 0.0),
                        "signal_strength_db": getattr(chunk, 'signal_strength_db', -150.0),
                        "squelch_open": getattr(chunk, 'squelch_open', True)
                    }
            except Exception as e:
                logger.warning(f"Could not decode recent audio chunk: {e}")
        
        stats = {
            "audio_chunks_count": audio_count,
            "transcripts_count": transcripts_count,
            "callsigns_count": callsigns_count,
            "qsos_count": qsos_count,
            "recent_audio": recent_audio,
            "audio_flowing": recent_audio is not None and (time.time() - recent_audio["last_chunk_time"]) < 5.0 if recent_audio else False,
            "uptime": "N/A"  # TODO: Track service uptime
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# Transcripts endpoints moved to top of file



@app.get("/api/callsigns", response_model=List[CallsignResponse])
async def get_callsigns(
    limit: int = Query(default=50, le=500),
    callsign: Optional[str] = None
):
    """Get recent callsigns from Database"""
    session = SessionLocal()
    try:
        query = session.query(CallsignModel)
        
        if callsign:
            query = query.filter(CallsignModel.callsign == callsign.upper())
            
        results_db = query.order_by(CallsignModel.timestamp.desc()).limit(limit).all()

        results = []
        for cs in results_db:
            results.append(CallsignResponse(
                callsign=cs.callsign,
                timestamp=cs.timestamp,
                datetime=cs.datetime.isoformat(),
                frequency_hz=cs.frequency_hz,
                confidence=cs.confidence,
                context=cs.context
            ))
            
        return results
    except Exception as e:
        logger.error(f"Error getting callsigns: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.get("/api/qsos", response_model=List[QSOResponse])
async def get_qsos(
    limit: int = Query(default=20, le=100)
):
    """Get recent QSO summaries from Database"""
    session = SessionLocal()
    try:
        results_db = session.query(QSOModel).order_by(QSOModel.start_time.desc()).limit(limit).all()

        results = []
        for qso in results_db:
            callsigns_list = qso.callsigns_list.split(',') if qso.callsigns_list else []
            
            results.append(QSOResponse(
                session_id=qso.session_id,
                start_time=qso.start_time,
                end_time=qso.end_time,
                start_datetime=datetime.fromtimestamp(qso.start_time).isoformat(),
                end_datetime=datetime.fromtimestamp(qso.end_time).isoformat() if qso.end_time else None,
                frequency_hz=qso.frequency_hz,
                mode=qso.mode,
                callsigns=callsigns_list,
                summary=qso.summary
            ))

        return results
    except Exception as e:
        logger.error(f"Error getting QSOs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


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


class FrequencyRequest(BaseModel):
    frequency_khz: float

class ModeRequest(BaseModel):
    mode: str

class FilterRequest(BaseModel):
    low_cut: Optional[int] = None
    high_cut: Optional[int] = None

class AGCRequest(BaseModel):
    agc_mode: Optional[int] = None  # 0=manual, 1=auto
    manual_gain: Optional[int] = None
    threshold: Optional[int] = None
    slope: Optional[int] = None
    decay: Optional[int] = None

class NoiseBlankerRequest(BaseModel):
    enabled: bool

def validate_frequency(frequency_khz: float) -> int:
    """Validate frequency is within KiwiSDR limits (0-30 MHz)"""
    if frequency_khz < 0 or frequency_khz > 30000:
        raise HTTPException(
            status_code=400, 
            detail=f"Frequency must be between 0 and 30000 kHz for KiwiSDR, got {frequency_khz} kHz"
        )
    return int(frequency_khz * 1000)

def validate_mode(mode: str) -> str:
    """Validate and sanitize mode"""
    valid_modes = ['USB', 'LSB', 'AM', 'FM', 'CW']
    mode_upper = str(mode).upper()
    if mode_upper not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Mode must be one of {valid_modes}, got {mode}"
        )
    return mode_upper

@app.post("/api/control/frequency", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def set_frequency(request: FrequencyRequest, req: Request = None):
    """
    Change the receiver frequency

    - **frequency_khz**: Frequency in kHz (0 to 30000 for KiwiSDR)
    """
    frequency_hz = validate_frequency(request.frequency_khz)

    try:
        # Send control command to audio-capture service via Redis
        command = {
            'command': 'set_frequency',
            'frequency_hz': str(frequency_hz),
            'timestamp': str(time.time())
        }

        # Add to control stream
        redis_client.xadd(STREAM_CONTROL, command, maxlen=100)  # Keep last 100 commands

        logger.info(f"Frequency change command sent: {frequency_hz} Hz ({request.frequency_khz} kHz)")

        return {
            "status": "success",
            "message": f"Frequency change command sent: {request.frequency_khz} kHz",
            "frequency_khz": request.frequency_khz,
            "frequency_hz": frequency_hz
        }
    except Exception as e:
        logger.error(f"Error sending frequency change command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/control/mode", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def set_mode(request: ModeRequest, req: Request = None):
    """
    Change the demodulation mode
    
    - **mode**: Demodulation mode (USB, LSB, AM, FM, CW)
    """
    mode = validate_mode(request.mode)
    
    try:
        # Send control command to audio-capture service via Redis
        command = {
            'command': 'set_mode',
            'mode': mode,
            'timestamp': str(time.time())
        }
        
        # Add to control stream
        redis_client.xadd(STREAM_CONTROL, command, maxlen=100)  # Keep last 100 commands
        
        logger.info(f"Mode change command sent: {mode}")
        
        return {
            "status": "success",
            "message": f"Mode change command sent: {mode}",
            "mode": mode
        }
    except Exception as e:
        logger.error(f"Error sending mode change command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/control/filter", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def set_filter(request: FilterRequest, req: Request = None):
    """
    Change filter bandwidth
    
    - **low_cut**: Low cutoff frequency in Hz (optional)
    - **high_cut**: High cutoff frequency in Hz (optional)
    """
    if request.low_cut is not None and (request.low_cut < 0 or request.low_cut > 20000):
        raise HTTPException(status_code=400, detail="low_cut must be between 0 and 20000 Hz")
    if request.high_cut is not None and (request.high_cut < 0 or request.high_cut > 20000):
        raise HTTPException(status_code=400, detail="high_cut must be between 0 and 20000 Hz")
    if request.low_cut is not None and request.high_cut is not None and request.low_cut >= request.high_cut:
        raise HTTPException(status_code=400, detail="low_cut must be less than high_cut")
    
    try:
        command = {
            'command': 'set_filter',
            'timestamp': str(time.time())
        }
        if request.low_cut is not None:
            command['low_cut'] = str(request.low_cut)
        if request.high_cut is not None:
            command['high_cut'] = str(request.high_cut)
        
        redis_client.xadd(STREAM_CONTROL, command, maxlen=100)
        logger.info(f"Filter change command sent: low_cut={request.low_cut}, high_cut={request.high_cut}")
        
        return {
            "status": "success",
            "message": "Filter change command sent",
            "low_cut": request.low_cut,
            "high_cut": request.high_cut
        }
    except Exception as e:
        logger.error(f"Error sending filter change command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/control/agc", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def set_agc(request: AGCRequest, req: Request = None):
    """
    Change AGC settings
    
    - **agc_mode**: 0=manual, 1=auto (optional)
    - **manual_gain**: Manual gain value (optional)
    - **threshold**: AGC threshold (optional)
    - **slope**: AGC slope (optional)
    - **decay**: AGC decay (optional)
    """
    try:
        command = {
            'command': 'set_agc',
            'timestamp': str(time.time())
        }
        if request.agc_mode is not None:
            command['agc_mode'] = str(request.agc_mode)
        if request.manual_gain is not None:
            command['manual_gain'] = str(request.manual_gain)
        if request.threshold is not None:
            command['threshold'] = str(request.threshold)
        if request.slope is not None:
            command['slope'] = str(request.slope)
        if request.decay is not None:
            command['decay'] = str(request.decay)
        
        redis_client.xadd(STREAM_CONTROL, command, maxlen=100)
        logger.info(f"AGC change command sent")
        
        return {
            "status": "success",
            "message": "AGC change command sent"
        }
    except Exception as e:
        logger.error(f"Error sending AGC change command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/control/noise-blanker", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def set_noise_blanker(request: NoiseBlankerRequest, req: Request = None):
    """
    Enable/disable noise blanker

    - **enabled**: True to enable, False to disable
    """
    try:
        command = {
            'command': 'set_noise_blanker',
            'enabled': 'true' if request.enabled else 'false',
            'timestamp': str(time.time())
        }

        redis_client.xadd(STREAM_CONTROL, command, maxlen=100)
        logger.info(f"Noise blanker change command sent: {request.enabled}")

        return {
            "status": "success",
            "message": f"Noise blanker {'enabled' if request.enabled else 'disabled'}",
            "enabled": request.enabled
        }
    except Exception as e:
        logger.error(f"Error sending noise blanker command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/control/summarize", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def trigger_summarize(req: Request = None):
    """
    Manually trigger QSO summarization for pending transcripts

    Forces the summarizer to process all transcripts since the last QSO summary,
    regardless of time gap.
    """
    try:
        command = {
            'command': 'trigger_summarize',
            'timestamp': str(time.time())
        }

        redis_client.xadd(STREAM_CONTROL, command, maxlen=100)
        logger.info("Manual summarization trigger sent")

        return {
            "status": "success",
            "message": "Summarization triggered - check /api/qsos for results"
        }
    except Exception as e:
        logger.error(f"Error sending summarization trigger command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/control/start", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def start_capture(req: Request = None):
    """
    Start/resume audio capture and processing
    """
    try:
        command = {
            'command': 'start_capture',
            'timestamp': str(time.time())
        }

        redis_client.xadd(STREAM_CONTROL, command, maxlen=100)
        logger.info("Audio capture start command sent")

        return {
            "status": "success",
            "message": "Audio capture started"
        }
    except Exception as e:
        logger.error(f"Error sending start command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/control/stop", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def stop_capture(req: Request = None):
    """
    Stop/pause audio capture and processing
    """
    try:
        command = {
            'command': 'stop_capture',
            'timestamp': str(time.time())
        }

        redis_client.xadd(STREAM_CONTROL, command, maxlen=100)
        logger.info("Audio capture stop command sent")

        return {
            "status": "success",
            "message": "Audio capture stopped"
        }
    except Exception as e:
        logger.error(f"Error sending stop command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kiwi/status")
async def get_kiwi_status():
    """Get KiwiSDR connection status and metrics"""
    # This would need to be implemented in audio-capture service
    # For now, return basic info
    return {
        "status": "unknown",
        "message": "KiwiSDR status endpoint - requires implementation in audio-capture service"
    }

@app.get("/api/kiwi/config")
async def get_kiwi_config():
    """Get current KiwiSDR configuration"""
    # Read from recent audio chunk
    try:
        messages = redis_client.xrevrange(STREAM_AUDIO, count=1)
        if messages:
            msg_id, msg_data = messages[0]
            chunk = RedisMessage.decode(msg_data, AudioChunk)
            return {
                "frequency_hz": chunk.frequency_hz,
                "mode": chunk.mode,
                "low_cut": chunk.low_cut,
                "high_cut": chunk.high_cut
            }
        return {"message": "No audio chunks available"}
    except Exception as e:
        logger.error(f"Error getting KiwiSDR config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audio/chunk/{chunk_id}")
async def get_audio_chunk(chunk_id: str):
    """
    Download an audio chunk as WAV file for playback/verification
    
    - **chunk_id**: Redis stream message ID (e.g., "1768622820-0")
    """
    try:
        # Read the specific message from Redis stream
        messages = redis_client.xrange(STREAM_AUDIO, min=chunk_id, max=chunk_id, count=1)
        
        if not messages:
            raise HTTPException(status_code=404, detail="Audio chunk not found")
        
        msg_id, msg_data = messages[0]
        chunk = RedisMessage.decode(msg_data, AudioChunk)
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(chunk.data, dtype=np.int16)

        # Reduce volume for comfortable playback (50% of original)
        PLAYBACK_VOLUME = 0.5
        audio_data = (audio_data * PLAYBACK_VOLUME).astype(np.int16)

        # Create WAV file in memory
        import wave
        import struct

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(chunk.sample_rate)
            wav_file.setcomptype('NONE', 'not compressed')

            # Write audio data
            for sample in audio_data:
                wav_file.writeframes(struct.pack('<h', int(sample)))
        
        wav_buffer.seek(0)
        
        return Response(
            content=wav_buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=chunk_{chunk_id}_{chunk.frequency_hz}Hz_{chunk.mode}.wav"
            }
        )
    except Exception as e:
        logger.error(f"Error getting audio chunk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audio/latest")
async def get_latest_audio():
    """Get the most recent audio chunk as WAV file"""
    try:
        # Get the most recent message
        messages = redis_client.xrevrange(STREAM_AUDIO, count=1)
        
        if not messages:
            raise HTTPException(status_code=404, detail="No audio chunks available")
        
        msg_id, msg_data = messages[0]
        chunk = RedisMessage.decode(msg_data, AudioChunk)
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(chunk.data, dtype=np.int16)

        # Reduce volume for comfortable playback (50% of original)
        PLAYBACK_VOLUME = 0.5
        audio_data = (audio_data * PLAYBACK_VOLUME).astype(np.int16)

        # Create WAV file in memory
        import wave
        import struct

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(chunk.sample_rate)
            wav_file.setcomptype('NONE', 'not compressed')

            # Write audio data
            for sample in audio_data:
                wav_file.writeframes(struct.pack('<h', int(sample)))
        
        wav_buffer.seek(0)
        
        return Response(
            content=wav_buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=latest_{chunk.frequency_hz}Hz_{chunk.mode}.wav"
            }
        )
    except Exception as e:
        logger.error(f"Error getting latest audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/monitor")
async def monitor_dashboard():
    """Real-time monitoring dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ARIS - Real-Time Monitor</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {
                font-family: 'Courier New', monospace;
                background: #1a1a1a;
                color: #00ff00;
                padding: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 { color: #00ff00; border-bottom: 2px solid #00ff00; }
            h2 { color: #00aa00; margin-top: 20px; }
            .card {
                background: #2a2a2a;
                border: 1px solid #00ff00;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            .status-good { color: #00ff00; }
            .status-warning { color: #ffaa00; }
            .status-error { color: #ff0000; }
            .transcript-item {
                background: #1a1a1a;
                border-left: 3px solid #00ff00;
                padding: 10px;
                margin: 5px 0;
            }
            .transcript-text { margin: 5px 0; }
            .transcript-meta { color: #888; font-size: 0.9em; }
            .audio-player { margin: 10px 0; }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin: 10px 0;
            }
            .stat-box {
                background: #1a1a1a;
                border: 1px solid #00ff00;
                padding: 10px;
                text-align: center;
            }
            .stat-value {
                font-size: 2em;
                font-weight: bold;
                color: #00ff00;
            }
            .stat-label {
                color: #888;
                font-size: 0.9em;
            }
            button {
                background: #00ff00;
                color: #1a1a1a;
                border: none;
                padding: 8px 16px;
                font-family: 'Courier New', monospace;
                cursor: pointer;
                margin: 5px;
            }
            button:hover { background: #00aa00; }
            .refresh-info { color: #888; font-size: 0.8em; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>📻 ARIS - Real-Time Monitor</h1>
        
        <div class="card">
            <h2>System Status</h2>
            <div id="stats"></div>
        </div>
        
        <div class="card">
            <h2>Latest Audio</h2>
            <div id="latest-audio">
                <p>Loading...</p>
            </div>
        </div>
        
        <div class="card">
            <h2>Recent Transcripts (Last 10)</h2>
            <div id="transcripts"></div>
        </div>
        
        <div class="card">
            <h2>Quick Actions</h2>
            <button onclick="playLatestAudio()">▶️ Play Latest Audio</button>
            <button onclick="downloadLatestAudio()">💾 Download Latest Audio</button>
            <button onclick="location.reload()">🔄 Refresh</button>
        </div>
        
        <div class="refresh-info">Auto-refreshing every 5 seconds...</div>
        
        <script>
            async function loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    
                    const statsHtml = `
                        <div class="stats-grid">
                            <div class="stat-box">
                                <div class="stat-value">${stats.audio_chunks_count || 0}</div>
                                <div class="stat-label">Audio Chunks</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-value">${stats.transcripts_count || 0}</div>
                                <div class="stat-label">Transcripts</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-value">${stats.callsigns_count || 0}</div>
                                <div class="stat-label">Callsigns</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-value ${stats.audio_flowing ? 'status-good' : 'status-error'}">${stats.audio_flowing ? '✓' : '✗'}</div>
                                <div class="stat-label">Audio Flowing</div>
                            </div>
                        </div>
                        ${stats.recent_audio ? `
                            <p><strong>Current Frequency:</strong> ${(stats.recent_audio.frequency_hz / 1000000).toFixed(3)} MHz (${stats.recent_audio.mode})</p>
                            <p><strong>Last Chunk:</strong> ${stats.recent_audio.last_chunk_datetime}</p>
                            <p><strong>Sample Rate:</strong> ${stats.recent_audio.sample_rate} Hz</p>
                        ` : '<p class="status-warning">No recent audio data</p>'}
                    `;
                    
                    document.getElementById('stats').innerHTML = statsHtml;
                } catch (e) {
                    document.getElementById('stats').innerHTML = '<p class="status-error">Error loading stats</p>';
                }
            }
            
            async function loadLatestAudio() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    
                    if (stats.recent_audio) {
                        const audioHtml = `
                            <p><strong>Frequency:</strong> ${(stats.recent_audio.frequency_hz / 1000000).toFixed(3)} MHz (${stats.recent_audio.mode})</p>
                            <p><strong>Last Update:</strong> ${stats.recent_audio.last_chunk_datetime}</p>
                            <div class="audio-player">
                                <audio controls>
                                    <source src="/api/audio/latest" type="audio/wav">
                                    Your browser does not support the audio element.
                                </audio>
                            </div>
                            <p><a href="/api/audio/latest" download>Download Latest Audio (WAV)</a></p>
                        `;
                        document.getElementById('latest-audio').innerHTML = audioHtml;
                    } else {
                        document.getElementById('latest-audio').innerHTML = '<p class="status-warning">No audio available</p>';
                    }
                } catch (e) {
                    document.getElementById('latest-audio').innerHTML = '<p class="status-error">Error loading audio</p>';
                }
            }
            
            async function loadTranscripts() {
                try {
                    const response = await fetch('/api/transcripts?limit=10');
                    const transcripts = await response.json();
                    
                    if (transcripts.length === 0) {
                        document.getElementById('transcripts').innerHTML = '<p class="status-warning">No transcripts yet</p>';
                        return;
                    }
                    
                    const transcriptsHtml = transcripts.map(t => `
                        <div class="transcript-item">
                            <div class="transcript-text">${escapeHtml(t.text)}</div>
                            <div class="transcript-meta">
                                ${new Date(t.datetime).toLocaleString()} | 
                                ${(t.frequency_hz / 1000000).toFixed(3)} MHz ${t.mode} | 
                                Confidence: ${(t.confidence * 100).toFixed(0)}% | 
                                <a href="/api/audio/chunk/${t.timestamp.toString().replace('.', '-')}" target="_blank">🎵 Audio</a>
                            </div>
                        </div>
                    `).join('');
                    
                    document.getElementById('transcripts').innerHTML = transcriptsHtml;
                } catch (e) {
                    document.getElementById('transcripts').innerHTML = '<p class="status-error">Error loading transcripts</p>';
                }
            }
            
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            function playLatestAudio() {
                const audio = document.querySelector('audio');
                if (audio) {
                    audio.play();
                } else {
                    loadLatestAudio().then(() => {
                        const audio = document.querySelector('audio');
                        if (audio) audio.play();
                    });
                }
            }
            
            function downloadLatestAudio() {
                window.open('/api/audio/latest', '_blank');
            }
            
            // Load data on page load
            loadStats();
            loadLatestAudio();
            loadTranscripts();
            
            // Auto-refresh every 5 seconds
            setInterval(() => {
                loadStats();
                loadLatestAudio();
                loadTranscripts();
            }, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
