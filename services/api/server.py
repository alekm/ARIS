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
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
import redis
from pydantic import BaseModel
import io
import numpy as np
import wave
import struct

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
                <li><code>POST /api/control/frequency</code> - Change receiver frequency (body: {"frequency_hz": 14313000})</li>
                <li><code>POST /api/control/mode</code> - Change demodulation mode (body: {"mode": "USB"})</li>
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
                    <label for="frequency" style="display: block; margin-bottom: 5px;">Frequency (Hz):</label>
                    <input type="number" id="frequency" placeholder="e.g., 7188000" style="width: 100%; padding: 8px; background: #1a1a1a; border: 1px solid #00ff00; color: #00ff00; font-family: 'Courier New', monospace;">
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
            <h2>Quick Actions</h2>
            <button onclick="loadTranscripts()">Load Recent Transcripts</button>
            <button onclick="loadCallsigns()">Load Recent Callsigns</button>
            <button onclick="loadQSOs()">Load QSO Summaries</button>
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

            async function changeFrequency() {
                const frequencyInput = document.getElementById('frequency');
                const frequency = parseInt(frequencyInput.value);
                const statusDiv = document.getElementById('frequency-status');
                
                if (!frequency || frequency < 100000 || frequency > 6000000000) {
                    statusDiv.innerHTML = '<span style="color: #ff0000;">Invalid frequency (100 kHz - 6 GHz)</span>';
                    return;
                }
                
                statusDiv.innerHTML = '<span style="color: #ffaa00;">Changing frequency...</span>';
                
                try {
                    const response = await fetch('/api/control/frequency', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ frequency_hz: frequency })
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
                        
                        // Update the frequency input with current value
                        document.getElementById('frequency').value = freq;
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
                        "duration_ms": chunk.duration_ms
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


class FrequencyRequest(BaseModel):
    frequency_hz: int

class ModeRequest(BaseModel):
    mode: str

@app.post("/api/control/frequency")
async def set_frequency(request: FrequencyRequest):
    """
    Change the receiver frequency
    
    - **frequency_hz**: Frequency in Hz (100 kHz to 6 GHz)
    """
    if request.frequency_hz < 100000 or request.frequency_hz > 6000000000:
        raise HTTPException(status_code=400, detail="Frequency must be between 100 kHz and 6 GHz")
    
    try:
        # Send control command to audio-capture service via Redis
        command = {
            'command': 'set_frequency',
            'frequency_hz': str(request.frequency_hz),
            'timestamp': str(time.time())
        }
        
        # Add to control stream
        redis_client.xadd(STREAM_CONTROL, command, maxlen=100)  # Keep last 100 commands
        
        logger.info(f"Frequency change command sent: {request.frequency_hz} Hz")
        
        return {
            "status": "success",
            "message": f"Frequency change command sent: {request.frequency_hz} Hz",
            "frequency_hz": request.frequency_hz
        }
    except Exception as e:
        logger.error(f"Error sending frequency change command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/control/mode")
async def set_mode(request: ModeRequest):
    """
    Change the demodulation mode
    
    - **mode**: Demodulation mode (USB, LSB, AM, FM)
    """
    if request.mode.upper() not in ['USB', 'LSB', 'AM', 'FM']:
        raise HTTPException(status_code=400, detail="Mode must be USB, LSB, AM, or FM")
    
    try:
        # Send control command to audio-capture service via Redis
        command = {
            'command': 'set_mode',
            'mode': request.mode.upper(),
            'timestamp': str(time.time())
        }
        
        # Add to control stream
        redis_client.xadd(STREAM_CONTROL, command, maxlen=100)  # Keep last 100 commands
        
        logger.info(f"Mode change command sent: {request.mode}")
        
        return {
            "status": "success",
            "message": f"Mode change command sent: {request.mode}",
            "mode": request.mode.upper()
        }
    except Exception as e:
        logger.error(f"Error sending mode change command: {e}")
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
        
        # Convert to float32 for WAV (normalize to -1.0 to 1.0)
        audio_float = audio_data.astype(np.float32) / 32768.0
        
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
