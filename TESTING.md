# ARIS Testing Guide

This guide covers how to test the ARIS system end-to-end, from audio capture through transcription, callsign extraction, and summarization.

## Quick Start Testing

### 1. Verify All Services Are Running

```bash
docker compose ps
```

All services should show "Up" status. Expected services:
- `aris-redis` (healthy)
- `aris-audio-capture` 
- `aris-stt`
- `aris-callsign-extractor`
- `aris-summarizer`
- `aris-api` (healthy)

### 2. Check Audio Capture (HackRF)

**Using the monitoring script:**
```bash
make monitor-audio
# or
./monitor-audio.sh
```

**Manual checks:**
```bash
# Verify HackRF is connected
lsusb | grep -i hackrf

# Check audio capture logs
docker compose logs -f audio-capture

# Check Redis stream for audio chunks
docker compose exec redis redis-cli XLEN audio:chunks
```

**What to look for:**
- ✅ Audio chunks count increasing: `XLEN audio:chunks` should show growing numbers
- ✅ No errors in logs (some `-4` timeout warnings are normal)
- ✅ Log messages showing "Captured X chunks" periodically
- ❌ If chunks aren't increasing, check HackRF connection and USB passthrough

### 3. Check STT Service (Speech-to-Text)

```bash
# Watch STT logs
docker compose logs -f stt

# Check for transcriptions
docker compose exec redis redis-cli XLEN transcripts
```

**What to look for:**
- ✅ No `KeyError` or `TypeError` errors
- ✅ Messages like "Transcribing Xms of audio"
- ✅ "Published transcript" messages
- ✅ Transcript count increasing in Redis stream

**Common issues:**
- If you see "Buffer too short" - this is normal, STT waits for enough audio
- If you see decode errors - check that `shared/models.py` has the latest fixes

### 4. Check API Endpoints

**System Statistics:**
```bash
curl http://localhost:8000/api/stats | python3 -m json.tool
```

Expected response:
```json
{
    "audio_chunks_count": 123,
    "transcripts_count": 5,
    "callsigns_count": 2,
    "qsos_count": 1,
    "recent_audio": {
        "last_chunk_time": 1768621169.098,
        "last_chunk_datetime": "2026-01-17T03:39:29",
        "frequency_hz": 7188000,
        "mode": "LSB",
        "sample_rate": 16000,
        "duration_ms": 1000
    },
    "audio_flowing": true,
    "uptime": "N/A"
}
```

**Web UI:**
```bash
# Open in browser
http://localhost:8000

# Or use curl
curl http://localhost:8000
```

**API Endpoints:**
- `/api/stats` - System statistics
- `/api/transcripts` - Recent transcripts
- `/api/callsigns` - Detected callsigns
- `/api/qsos` - QSO summaries
- `/docs` - Interactive API documentation (Swagger UI)

### 5. End-to-End Pipeline Test

**Test the full pipeline:**

1. **Audio Capture** → Check Redis stream:
   ```bash
   docker compose exec redis redis-cli XREVRANGE audio:chunks + - COUNT 1
   ```
   Should show recent audio chunk with hex data.

2. **STT Processing** → Check transcripts:
   ```bash
   docker compose exec redis redis-cli XREVRANGE transcripts + - COUNT 1
   ```
   Should show transcribed text.

3. **Callsign Extraction** → Check callsigns:
   ```bash
   docker compose exec redis redis-cli XREVRANGE callsigns + - COUNT 1
   ```
   Should show extracted callsigns if any were found.

4. **Summarization** → Check QSOs:
   ```bash
   docker compose exec redis redis-cli XREVRANGE qsos + - COUNT 1
   ```
   Should show QSO summaries if conversations were detected.

## Detailed Testing

### Service Health Checks

**Redis:**
```bash
docker compose exec redis redis-cli ping
# Should return: PONG
```

**API Health:**
```bash
curl http://localhost:8000/api/stats
# Should return JSON, not error
```

**Audio Capture:**
```bash
docker compose logs audio-capture --tail=20 | grep -E "(ERROR|WARNING|INFO.*Captured)"
```

**STT Service:**
```bash
docker compose logs stt --tail=20 | grep -E "(ERROR|Transcribed|Published transcript)"
```

### Monitoring Commands

**Using Makefile:**
```bash
make monitor-audio    # Monitor audio capture
make stats            # Show Redis stream stats
make logs-audio      # Tail audio capture logs
make logs-stt         # Tail STT logs
make logs-api         # Tail API logs
make ps               # Show service status
```

**Manual monitoring:**
```bash
# Watch all logs
docker compose logs -f

# Watch specific service
docker compose logs -f audio-capture
docker compose logs -f stt

# Check Redis streams
docker compose exec redis redis-cli XLEN audio:chunks
docker compose exec redis redis-cli XLEN transcripts
docker compose exec redis redis-cli XLEN callsigns
docker compose exec redis redis-cli XLEN qsos
```

### Testing Individual Components

#### 1. Audio Capture Service

**Test HackRF connection:**
```bash
# Check if HackRF is visible to container
docker compose exec audio-capture ls -la /dev/bus/usb/ | grep -i hackrf

# Check SoapySDR can see device
docker compose exec audio-capture python3 -c "from SoapySDR import *; print(Device.enumerate())"
```

**Test audio flow:**
```bash
# Monitor chunk rate
watch -n 1 'docker compose exec redis redis-cli XLEN audio:chunks'

# Should see count increasing every ~1 second
```

#### 2. STT Service

**Test model loading:**
```bash
docker compose logs stt | grep "Model loaded successfully"
# Should see this message on startup
```

**Test transcription:**
```bash
# Wait for audio to buffer (STT needs ~3-5 seconds of audio)
# Then check for transcripts
docker compose exec redis redis-cli XREVRANGE transcripts + - COUNT 5
```

#### 3. Callsign Extractor

**Check for extracted callsigns:**
```bash
docker compose logs callsign-extractor --tail=20
docker compose exec redis redis-cli XREVRANGE callsigns + - COUNT 5
```

#### 4. Summarizer

**Check for QSO summaries:**
```bash
docker compose logs summarizer --tail=20
docker compose exec redis redis-cli XREVRANGE qsos + - COUNT 5
```

**Note:** Summarizer requires LLM backend. Check `.env` for `LLM_HOST` configuration.

## Troubleshooting Tests

### If Audio Isn't Flowing

1. **Check HackRF connection:**
   ```bash
   lsusb | grep -i hackrf
   # Should show: Bus 00X Device 00Y: ID 1d50:6089 Great Scott Gadgets HackRF One
   ```

2. **Check USB passthrough:**
   ```bash
   docker compose exec audio-capture ls -la /dev/bus/usb/
   # Should show USB devices
   ```

3. **Check SoapySDR module:**
   ```bash
   docker compose exec audio-capture ls -la /usr/lib/x86_64-linux-gnu/SoapySDR/modules0.8/
   # Should show libHackRFSupport.so
   ```

4. **Check audio capture logs:**
   ```bash
   docker compose logs audio-capture | grep -E "(ERROR|WARNING|HackRF)"
   ```

### If STT Isn't Processing

1. **Check for decode errors:**
   ```bash
   docker compose logs stt | grep -E "(KeyError|TypeError|Error processing)"
   ```

2. **Check audio chunks are in Redis:**
   ```bash
   docker compose exec redis redis-cli XLEN audio:chunks
   # Should be > 0
   ```

3. **Check STT is receiving chunks:**
   ```bash
   docker compose logs stt | grep "Received chunk_data"
   # Should see this repeatedly
   ```

### If API Isn't Responding

1. **Check API health:**
   ```bash
   curl http://localhost:8000/api/stats
   ```

2. **Check API logs:**
   ```bash
   docker compose logs api --tail=20
   ```

3. **Check API container:**
   ```bash
   docker compose ps api
   # Should show "Up (healthy)"
   ```

## Performance Testing

### Measure Audio Capture Rate

```bash
# Count chunks over 10 seconds
START=$(docker compose exec redis redis-cli XLEN audio:chunks)
sleep 10
END=$(docker compose exec redis redis-cli XLEN audio:chunks)
echo "Chunks per second: $(( (END - START) / 10 ))"
# Should be ~1 chunk/second (1000ms chunks)
```

### Measure Transcription Latency

```bash
# Watch for transcription events
docker compose logs -f stt | grep "Published transcript"
# Note time between audio chunks and transcript publication
```

### Check System Resources

```bash
# CPU and memory usage
docker stats --no-stream

# Disk usage
docker system df
```

## Integration Test Script

Create a simple test script:

```bash
#!/bin/bash
# test-aris.sh - Quick integration test

echo "=== ARIS Integration Test ==="

# 1. Check services
echo "1. Checking services..."
docker compose ps | grep -q "Up" || { echo "❌ Services not running"; exit 1; }
echo "✅ Services running"

# 2. Check audio chunks
echo "2. Checking audio capture..."
sleep 2
CHUNKS=$(docker compose exec -T redis redis-cli XLEN audio:chunks)
if [ "$CHUNKS" -gt 0 ]; then
    echo "✅ Audio chunks detected: $CHUNKS"
else
    echo "⚠️  No audio chunks yet (may need more time)"
fi

# 3. Check API
echo "3. Checking API..."
STATS=$(curl -s http://localhost:8000/api/stats)
if echo "$STATS" | grep -q "audio_chunks_count"; then
    echo "✅ API responding"
else
    echo "❌ API not responding"
    exit 1
fi

# 4. Check transcripts (may be empty if no speech)
echo "4. Checking transcripts..."
TRANSCRIPTS=$(echo "$STATS" | grep -o '"transcripts_count":[0-9]*' | cut -d: -f2)
echo "   Transcripts: $TRANSCRIPTS"

echo ""
echo "=== Test Complete ==="
```

## Expected Behavior

### Normal Operation

1. **Audio Capture:**
   - Captures ~1 chunk per second (1000ms chunks)
   - Some `-4` timeout warnings are normal (HackRF read timeouts)
   - Chunk count in Redis should steadily increase

2. **STT Service:**
   - Buffers ~3-5 seconds of audio before transcribing
   - Produces transcripts when speech is detected
   - May show "No speech detected" if only noise

3. **Callsign Extractor:**
   - Extracts callsigns from transcripts
   - May not find callsigns if none are present in audio

4. **Summarizer:**
   - Creates QSO summaries when conversations are detected
   - Requires LLM backend to be configured

5. **API:**
   - Responds to all endpoints
   - Shows increasing counts as data flows
   - `audio_flowing` should be `true` when audio is active

### Success Criteria

✅ All services running and healthy  
✅ Audio chunks accumulating in Redis  
✅ STT processing chunks without errors  
✅ API responding with correct data  
✅ Transcripts appearing when speech is detected  
✅ No critical errors in logs  

## Next Steps

Once basic testing passes:
1. Monitor for actual ham radio traffic
2. Verify callsign extraction accuracy
3. Test summarization with real conversations
4. Check QSO tracking and session management
5. Monitor system performance over time
