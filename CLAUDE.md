# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

ARIS (Amateur Radio Intelligence System) is a GPU-accelerated amateur radio intelligence system that monitors ham radio frequencies via KiwiSDR receivers. It performs real-time speech-to-text transcription, extracts callsigns, and generates AI summaries of QSOs (conversations) and nets.

The system is designed to run locally without cloud dependencies on Ubuntu with NVIDIA GPUs (2x A4000, i9-10980XE, 96GB RAM).

## Architecture

### Service Pipeline (Redis Streams)

Data flows through 5 microservices via Redis streams:

1. **audio-capture** → `STREAM_AUDIO` ("audio:chunks")
   - Captures from KiwiSDR or mock audio source
   - Publishes `AudioChunk` dataclass (PCM int16 bytes + metadata)

2. **stt** → `STREAM_TRANSCRIPTS` ("transcripts")
   - Consumes audio chunks, buffers audio (1-30 seconds)
   - Uses faster-whisper (GPU) with VAD filtering
   - Publishes `Transcript` dataclass (text + confidence + metadata)

3. **callsign-extractor** → `STREAM_CALLSIGNS` ("callsigns")
   - Consumes transcripts
   - Regex pattern matching for US/Canadian calls (e.g., W1AW, K4XXX)
   - Converts phonetic alphabet ("Alpha Bravo" → "AB")
   - Publishes `Callsign` dataclass with context

4. **summarizer** → `STREAM_QSOS` ("qsos")
   - Consumes both transcripts and callsigns
   - Groups transcripts by frequency + time gaps (30s threshold)
   - Sends completed sessions to Ollama LLM for summary
   - Publishes `QSO` dataclass (session with summary)

5. **api** (FastAPI)
   - REST API reads from all Redis streams
   - Web UI at http://localhost:8000
   - Endpoints: `/api/transcripts`, `/api/callsigns`, `/api/qsos`, `/api/search/callsign/{call}`

### Data Models (`shared/models.py`)

All services import shared dataclasses:
- `AudioChunk`, `Transcript`, `Callsign`, `QSO`
- `RedisMessage.encode()` / `.decode()` for Redis stream serialization
- Stream names: `STREAM_AUDIO`, `STREAM_TRANSCRIPTS`, `STREAM_CALLSIGNS`, `STREAM_QSOS`

### Consumer Groups

Services use Redis consumer groups for scalability:
- Each service creates a consumer group on startup (handles `BUSYGROUP` error gracefully)
- Messages are acknowledged with `XACK` after processing
- Streams use `maxlen` for automatic cleanup (1000-10000 messages)

## Common Commands

```bash
# Initial setup
./setup.sh                    # Create data dirs, copy .env
make setup                    # Alternative setup

# Build and run
make build                    # Build all Docker images
make up                       # Start all services in background
make down                     # Stop all services

# Development
make logs                     # Tail all service logs
make logs-stt                 # Tail specific service
make logs-audio / logs-callsign / logs-summarizer / logs-api
make ps                       # Show running containers
make stats                    # Show Redis stream lengths

# Testing
make test-mock                # Start with mock audio + follow logs

# Restart
make restart                  # Restart all services
make restart-audio            # Restart audio capture only

# Cleanup
make clean                    # Stop and remove containers + volumes
```

## Configuration

### Environment Variables (`.env` from `.env.example`)

**Audio Capture:**
- `MODE=mock` or `MODE=kiwi`
- `KIWI_HOST`, `KIWI_PORT`, `FREQUENCY_HZ`

**STT:**
- `MODEL_SIZE=medium` (tiny/base/small/medium/large-v2/large-v3)
- `DEVICE=cuda` or `cpu`
- `VAD_THRESHOLD=0.5` (voice activity detection)
- `MAX_BUFFER_MS=30000`, `MIN_BUFFER_MS=1000`

**LLM:**
- `LLM_BACKEND=ollama` (or llama-cpp/openai-compatible)
- `LLM_MODEL=llama3.2:latest`
- `LLM_HOST=host.docker.internal:11434`

### Service-Specific Config

**audio-capture:** `services/audio-capture/config.yaml`
- Frequency, mode (USB/LSB/FM/AM/CW), sample rate
- KiwiSDR host/port/password

## Development Workflow

### Mock Mode (No Hardware)

The system can run entirely with mock audio:
- `MODE=mock` generates test tones (300-3000 Hz voice range + noise)
- Allows testing the entire pipeline without KiwiSDR

### Adding KiwiSDR Support

1. Implement `KiwiSDRAudioSource.read_chunk()` in `services/audio-capture/capture.py:92`
   - Connect to KiwiSDR WebSocket (port 8073)
   - Return `AudioChunk` with real audio bytes

2. Set `MODE=kiwi` in `.env`

3. Configure `services/audio-capture/config.yaml` with KiwiSDR IP/frequency

### Service Communication Pattern

All services follow this pattern:
```python
# Connect to Redis
redis_client = redis.Redis(host=os.getenv('REDIS_HOST'), port=int(os.getenv('REDIS_PORT')), decode_responses=False)

# Create consumer group
redis_client.xgroup_create(STREAM_NAME, 'service-name', id='0', mkstream=True)

# Read loop
while True:
    messages = redis_client.xreadgroup('service-name', f'consumer-{pid}',
                                       {STREAM_NAME: '>'}, count=N, block=1000)
    for stream, stream_messages in messages:
        for msg_id, msg_data in stream_messages:
            obj = RedisMessage.decode(msg_data, DataClass)
            # Process...
            redis_client.xack(STREAM_NAME, 'service-name', msg_id)
```

### Debugging

**View Redis streams directly:**
```bash
docker compose exec redis redis-cli
> XLEN audio:chunks
> XRANGE transcripts - + COUNT 5
> XINFO GROUPS transcripts
```

**Test individual services:**
```bash
docker compose up -d redis audio-capture  # Only start needed services
docker compose logs -f audio-capture      # Watch specific service
```

**Check STT GPU usage:**
```bash
nvidia-smi -l 1  # Monitor GPU while stt service runs
```

## Current Implementation Status

### ✅ Implemented
- Complete Docker microservices architecture
- Redis stream-based message bus
- Mock audio source with test tones
- STT pipeline with faster-whisper + GPU
- Callsign extraction (regex + phonetics)
- LLM summarization via Ollama
- FastAPI REST API + web UI
- Makefile shortcuts

### 🚧 In Progress
- KiwiSDR WebSocket client (placeholder exists at `services/audio-capture/capture.py:75-95`)

### 📋 Planned (Roadmap in README.md)
- Multi-frequency monitoring
- Alert system (webhooks/notifications)
- PostgreSQL for long-term storage
- DX spot detection
- Net detection and tracking

## Hardware Notes

- **GPU Required:** STT service needs NVIDIA GPU with CUDA support (set `DEVICE=cpu` for CPU fallback)
- **Ollama:** Must be running on host machine at `LLM_HOST` for summarization
- **nvidia-docker:** Required for GPU pass-through to containers

## Audio Processing Details

- **Sample Rate:** 16kHz (Whisper optimal for voice)
- **Audio Format:** int16 PCM in `AudioChunk.data` bytes
- **VAD:** Built into faster-whisper, filters non-speech segments
- **Buffering:** STT accumulates 1-30 seconds before transcribing (configurable)
- **Session Gaps:** Summarizer groups transcripts when gap > 30 seconds

## Callsign Detection

- **Pattern:** `[A-Z]{1,2}\d[A-Z]{1,3}` (e.g., W1AW, K4XXX, N2ABC)
- **Prefixes:** US/Canadian (K, W, N, A, VE)
- **Phonetics:** Converts "Whiskey One Alpha Whiskey" → "W1AW"
- **Blacklist:** Filters false positives (TEST, QSO, CQ, FM, AM, USB, LSB)
