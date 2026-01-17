# ARIS Deployment Changes

This document tracks all changes made to prepare ARIS for deployment with HackRF support.

## 2026-01-17 - Added Dynamic Frequency Control API

### Feature
Added API endpoints to dynamically change receiver frequency and demodulation mode without restarting the audio-capture service.

### Implementation
- **Control Stream**: Added `STREAM_CONTROL` to `shared/models.py` for Redis-based command communication
- **HackRF Methods**: Added `set_frequency()` and `set_mode()` methods to `HackRFAudioSource` class for dynamic changes
- **Command Handler**: Added `check_control_commands()` method to `AudioCaptureService` that listens for control commands via Redis consumer groups
- **API Endpoints**:
  - `POST /api/control/frequency?frequency_hz=<hz>` - Change frequency (100 kHz to 6 GHz)
  - `POST /api/control/mode?mode=<USB|LSB|AM|FM>` - Change demodulation mode

### Files Changed
- `shared/models.py`: Added `STREAM_CONTROL` constant
- `services/audio-capture/capture.py`: 
  - Added `set_frequency()` and `set_mode()` methods to `HackRFAudioSource`
  - Added `check_control_commands()` method to `AudioCaptureService`
  - Modified `run()` loop to check for control commands every second
- `services/api/server.py`: Added frequency and mode control endpoints

### Usage
```bash
# Change frequency to 14.313 MHz (20m band)
curl -X POST "http://localhost:8000/api/control/frequency?frequency_hz=14313000"

# Change mode to USB
curl -X POST "http://localhost:8000/api/control/mode?mode=USB"
```

### Impact
- Users can now change frequency and mode via API without restarting services
- Real-time frequency changes supported
- Commands are queued via Redis streams for reliable delivery

---

## 2026-01-17 - Fixed API Transcripts Endpoint Type Conversion

### Issue
The `/api/transcripts` endpoint was returning an error: `'str' object cannot be interpreted as an integer`. This was caused by numeric fields (timestamp, frequency_hz, confidence, duration_ms) being decoded from Redis as strings instead of their proper numeric types.

### Fix
Updated the `from_dict()` methods in `shared/models.py` for all data models to convert string values to proper numeric types:
- **Transcript**: Added type conversion for `timestamp` (float), `frequency_hz` (int), `confidence` (float), and `duration_ms` (int)
- **Callsign**: Added type conversion for `timestamp` (float), `frequency_hz` (int), and `confidence` (float)
- **QSO**: Added type conversion for `start_time` (float), `end_time` (float), `frequency_hz` (int), and `transcript_ids` (List[int])

### Files Changed
- `shared/models.py`: Enhanced `from_dict()` methods for Transcript, Callsign, and QSO classes

### Impact
- The `/api/transcripts` endpoint now correctly returns properly typed data
- All numeric fields are correctly converted from Redis string values to their expected types
- The API can now properly serialize responses without type errors

---

## Date: 2025-01-17

### Summary
Major deployment readiness improvements including environment variable configuration, security hardening, HackRF support, and Docker build fixes.

---

## 1. Environment Configuration & Deployment Readiness

### Created `.env.example`
- **File**: `.env.example`
- **Purpose**: Centralized configuration template for all deployment settings
- **Includes**:
  - API configuration (port, host)
  - Redis configuration
  - Audio capture settings (mode, KiwiSDR, HackRF)
  - STT configuration (model size, device, VAD threshold)
  - LLM configuration (backend, model, host, API key)
  - Database configuration
  - HackRF-specific settings (gains, sample rates, bandwidth)

### Updated `docker-compose.yml` for Environment Variables
- **Changed**: All services now read configuration from environment variables
- **Pattern**: `${VAR_NAME:-default_value}` syntax for all configurable values
- **Benefits**: 
  - Easy configuration via `.env` file
  - No need to edit docker-compose.yml for different deployments
  - Supports different environments (dev, staging, production)

**Services Updated**:
- `audio-capture`: MODE, KIWI_HOST, KIWI_PORT, FREQUENCY_HZ, DEMOD_MODE, HackRF settings
- `stt`: MODEL_SIZE, DEVICE, VAD_THRESHOLD
- `summarizer`: LLM_BACKEND, LLM_MODEL, LLM_HOST, LLM_API_KEY
- `api`: API_PORT, API_HOST, DATABASE_URL
- All services: REDIS_HOST, REDIS_PORT

---

## 2. Security Improvements

### Redis Port Security
- **File**: `docker-compose.yml`
- **Change**: Removed Redis port exposure to host (`6379:6379` → removed)
- **Before**: Redis was accessible from host on port 6379
- **After**: Redis is only accessible within Docker internal network
- **Impact**: Improved security - Redis no longer exposed to external access
- **Note**: Services still access Redis via internal Docker network using service name `redis`

---

## 3. Configurable Ports

### API Port Configuration
- **File**: `docker-compose.yml`, `services/api/Dockerfile`, `services/api/server.py`
- **Changes**:
  - API port now configurable via `API_PORT` environment variable (default: 8000)
  - Dockerfile updated to use environment variable for uvicorn host/port
  - Port mapping in docker-compose.yml uses `${API_PORT:-8000}:8000`
- **Benefits**: Can run multiple instances on different ports, easier port management

---

## 4. Health Checks

### Added Service Health Checks
- **File**: `docker-compose.yml`
- **Changes**:
  - **Redis**: Added health check using `redis-cli ping`
    - Interval: 10s
    - Timeout: 5s
    - Retries: 5
  - **API**: Added health check using `curl` to `/api/stats` endpoint
    - Interval: 30s
    - Timeout: 10s
    - Retries: 3
    - Start period: 40s (allows time for service to start)
- **Dependencies**: All services now use `condition: service_healthy` for Redis dependency
- **Benefits**: 
  - Services wait for Redis to be ready before starting
  - Better monitoring and automatic recovery
  - Prevents race conditions during startup

### API Dockerfile Updates
- **File**: `services/api/Dockerfile`
- **Change**: Added `curl` installation for health checks
- **Command**: `apt-get install -y curl`

---

## 5. HackRF Support Configuration

### USB Passthrough Enabled
- **File**: `docker-compose.yml`
- **Changes**:
  - Enabled `privileged: true` for audio-capture service
  - Added USB device passthrough: `/dev/bus/usb:/dev/bus/usb`
- **Purpose**: Allows container to access HackRF USB device on host
- **Security Note**: Privileged mode required for USB device access

### Default Mode Changed to HackRF
- **Files**: 
  - `docker-compose.yml`: `MODE=${MODE:-hackrf}`
  - `services/audio-capture/config.yaml`: `mode: hackrf`
  - `.env.example`: `MODE=hackrf`
- **Impact**: System defaults to HackRF mode instead of mock mode

### HackRF Environment Variables
- **File**: `docker-compose.yml`
- **Added Variables**:
  - `HACKRF_SERIAL`: Optional device serial number (for multiple devices)
  - `RF_SAMPLE_RATE`: RF sample rate in Hz (default: 2000000)
  - `LNA_GAIN`: LNA gain in dB (default: 16, range: 0-40, 8 dB steps)
  - `VGA_GAIN`: VGA gain in dB (default: 20, range: 0-62, 2 dB steps)
  - `BANDWIDTH`: RF bandwidth filter in Hz (default: 1750000)
- **Purpose**: Allow runtime configuration of HackRF settings without rebuilding

### Capture Code Updates
- **File**: `services/audio-capture/capture.py`
- **Changes**: `HackRFAudioSource.__init__()` now reads settings from environment variables first, then falls back to config file
- **Updated Settings**:
  - `frequency_hz`: From `FREQUENCY_HZ` env var or config
  - `demod_mode`: From `DEMOD_MODE` env var or config
  - `rf_sample_rate`: From `RF_SAMPLE_RATE` env var or config
  - `lna_gain`: From `LNA_GAIN` env var or config
  - `vga_gain`: From `VGA_GAIN` env var or config
  - `bandwidth`: From `BANDWIDTH` env var or config
  - `hackrf_serial`: From `HACKRF_SERIAL` env var or config
- **Benefits**: Environment variables override config file, enabling easier deployment configuration

---

## 6. Docker Build Fixes

### Removed Obsolete Version Attribute
- **File**: `docker-compose.yml`
- **Change**: Removed `version: '3.8'` line
- **Reason**: Docker Compose v2 no longer requires or uses version field
- **Impact**: Eliminates warning: "the attribute `version` is obsolete"

### Fixed Shared Code Access
- **Problem**: Dockerfiles used `COPY --from=shared` which referenced a non-existent Docker image
- **Solution**: Changed build context to project root and updated all COPY paths

#### Build Context Changes
- **File**: `docker-compose.yml`
- **Changed**: All service builds now use project root as context
- **Before**: `build: ./services/api`
- **After**: 
  ```yaml
  build:
    context: .
    dockerfile: ./services/api/Dockerfile
  ```
- **Applied to**: All services (api, stt, callsign-extractor, summarizer, audio-capture)

#### Dockerfile Path Updates
- **Files**: All service Dockerfiles
- **Changes**:
  - **Before**: `COPY requirements.txt .` → `COPY services/api/requirements.txt .`
  - **Before**: `COPY *.py /app/` → `COPY services/api/*.py /app/`
  - **Before**: `COPY --from=shared ../../shared/models.py` → `COPY shared/models.py /app/shared/`
  - **Before**: `COPY static /app/static` → `COPY services/api/static /app/static`

**Services Updated**:
- `services/api/Dockerfile`
- `services/stt/Dockerfile`
- `services/callsign-extractor/Dockerfile`
- `services/summarizer/Dockerfile`
- `services/audio-capture/Dockerfile`

---

## 7. Documentation Updates

### README.md Enhancements
- **File**: `README.md`
- **Added Sections**:
  - **Deployment Section**: Comprehensive deployment checklist
    - Environment configuration steps
    - Security considerations
    - Health checks information
    - Data persistence notes
    - GPU configuration verification
  - **Deployment Commands**: Quick reference for common operations
  - **Environment Variables**: Documentation of all available configuration options
- **Updated**: Phase 4 roadmap to reflect completed items (health checks, env vars, security)

---

## 8. Configuration Files

### config.yaml Updates
- **File**: `services/audio-capture/config.yaml`
- **Change**: Default mode changed from `mock` to `hackrf`
- **Note**: Can still be overridden via `MODE` environment variable

---

## Migration Guide

### For Existing Deployments

1. **Copy Environment File**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

2. **Update docker-compose.yml**:
   - If you have a custom docker-compose.yml, merge the environment variable changes
   - Ensure build contexts are set to project root

3. **Rebuild Images**:
   ```bash
   docker compose down
   docker compose build --no-cache
   docker compose up -d
   ```

4. **Verify HackRF Access** (if using HackRF):
   ```bash
   # On host, verify device is detected
   lsusb | grep -i hackrf
   
   # Check container logs
   docker compose logs audio-capture
   ```

5. **Check Health**:
   ```bash
   docker compose ps  # Should show all services as healthy
   ```

---

## Breaking Changes

### None
- All changes are backward compatible
- Default values maintain previous behavior where applicable
- Existing configurations will continue to work

---

## Testing Recommendations

1. **Test with Mock Mode First**:
   ```bash
   # In .env
   MODE=mock
   docker compose up -d
   ```

2. **Verify Health Checks**:
   ```bash
   docker compose ps
   # All services should show as healthy
   ```

3. **Test HackRF**:
   ```bash
   # In .env
   MODE=hackrf
   FREQUENCY_HZ=7200000
   DEMOD_MODE=USB
   docker compose restart audio-capture
   docker compose logs -f audio-capture
   ```

4. **Verify API**:
   ```bash
   curl http://localhost:8000/api/stats
   ```

---

## Files Modified

### Configuration Files
- `docker-compose.yml` - Major updates for env vars, health checks, HackRF
- `services/audio-capture/config.yaml` - Mode changed to hackrf
- `.env.example` - New file with all configuration options

### Dockerfiles
- `services/api/Dockerfile` - Build context, curl for health checks, env vars
- `services/stt/Dockerfile` - Build context fixes
- `services/callsign-extractor/Dockerfile` - Build context fixes
- `services/summarizer/Dockerfile` - Build context fixes
- `services/audio-capture/Dockerfile` - Build context fixes

### Source Code
- `services/audio-capture/capture.py` - Environment variable support for HackRF settings

### Documentation
- `README.md` - Added deployment section, updated roadmap
- `CHANGES.md` - This file (new)

---

## Next Steps

1. **Test Deployment**: Run through deployment checklist in README
2. **Monitor Logs**: Watch service logs for any issues
3. **Tune HackRF Settings**: Adjust gains and frequency based on your use case
4. **Configure LLM**: Ensure LLM_HOST points to your Ollama or compatible server
5. **Set Up Monitoring**: Consider adding external monitoring for production

---

## Notes

- All environment variables have sensible defaults
- Redis is now internal-only for security
- Health checks ensure proper service startup order
- HackRF requires privileged mode and USB passthrough
- Build context changes require rebuilding all images
- No data migration needed - existing data directories remain compatible

---

## Critical Fixes (2026-01-17)

### Redis Message Decode Fix - STT Service
- **File**: `shared/models.py`
- **Problem**: 
  - Redis returns message keys as bytes (`b'data'`, `b'timestamp'`) not strings (`'data'`, `'timestamp'`)
  - This caused `KeyError: 'data'` when trying to decode AudioChunk messages
  - Numeric fields were also strings instead of int/float, causing `TypeError: unsupported operand type(s) for +=: 'int' and 'str'`
- **Solution**: 
  - Updated `RedisMessage.decode()` to convert bytes keys to strings: `key = k.decode('utf-8') if isinstance(k, bytes) else k`
  - Added type conversion in `AudioChunk.from_dict()` to convert string values to proper types:
    - `timestamp`: string → float
    - `frequency_hz`: string → int
    - `sample_rate`: string → int
    - `duration_ms`: string → int
  - Optimized hex string handling: skip JSON parsing for strings > 1000 chars (audio data is very long hex strings)
- **Impact**: STT service can now successfully decode and process audio chunks from Redis streams
- **Errors Fixed**: 
  - `KeyError: 'data'` 
  - `TypeError: unsupported operand type(s) for +=: 'int' and 'str'`

### Audio Capture Service - Python Version Fix
- **File**: `services/audio-capture/Dockerfile`
- **Problem**: Python 3.11 from `python:3.11-slim` not compatible with apt package `python3-soapysdr` (requires Python 3.12)
- **Solution**: Changed base image from `python:3.11-slim` to `ubuntu:24.04` with system Python 3.12
- **Changes**:
  - Uses Ubuntu 24.04 system Python 3.12 (matches apt package requirements)
  - Added `--break-system-packages` flag for pip installs (PEP 668)
  - Now uses apt-installed `python3-soapysdr` directly (no need to build from source)
- **Impact**: SoapySDR Python bindings work correctly, faster builds, more reliable

### SoapySDR Module Path Fix
- **File**: `services/audio-capture/Dockerfile`, `docker-compose.yml`
- **Problem**: SoapySDR couldn't find HackRF module even though it was installed
- **Solution**: 
  - Added `SOAPY_SDR_MODULE_PATH` environment variable pointing to Ubuntu's architecture-specific path
  - Path: `/usr/lib/x86_64-linux-gnu/SoapySDR/modules0.8/`
  - Also includes fallback paths for source-built modules
- **Impact**: SoapySDR can now find and load the HackRF module

### USB Device Access Fix
- **File**: `docker-compose.yml`
- **Problem**: Container couldn't access HackRF USB device even with privileged mode
- **Solution**: 
  - Added `user: root` to audio-capture service
  - Ensures container runs as root for USB device access
  - Kept `privileged: true` and USB device passthrough
- **Impact**: HackRF device is now accessible in container

### Frequency Configuration Update
- **Files**: `.env`, `.env.example`, `services/audio-capture/config.yaml`
- **Change**: Updated frequency from 7.2 MHz to 7.188 MHz
- **Settings**: 
  - `FREQUENCY_HZ=7188000`
  - `DEMOD_MODE=LSB`

### Debug Logging Added
- **Files**: `services/stt/transcribe.py`, `shared/models.py`
- **Changes**: 
  - Added DEBUG level logging to STT service for troubleshooting
  - Added debug logging in `RedisMessage.decode()` to track key conversion
  - Helps identify decode issues in the future
- **Note**: Can be reduced to INFO level in production for less verbose logs

## Additional Fixes (2026-01-17)

### Docker Build Fixes

#### Created `.dockerignore`
- **File**: `.dockerignore`
- **Purpose**: Exclude data directories and other files from Docker build context
- **Excludes**: `data/`, logs, `.env`, git files
- **Benefit**: Prevents permission errors and reduces build context size

#### Fixed STT Service Dockerfile
- **File**: `services/stt/Dockerfile`
- **Changes**:
  - Updated CUDA base image from deprecated `12.1.0-cudnn8-runtime-ubuntu22.04` to `13.0.2-cudnn-runtime-ubuntu24.04`
  - Fixed Python installation: Changed from `python3.11` (not available in Ubuntu 24.04) to `python3` (Python 3.12)
  - Fixed PEP 668 issue: Added `--break-system-packages` flag to pip install (required for Ubuntu 24.04)
  - Removed pip upgrade step (can't upgrade apt-installed pip with pip itself)
  - Added verification step to ensure numpy is installed correctly
- **Impact**: 
  - Uses current CUDA version (no deprecation warnings)
  - Compatible with Ubuntu 24.04 and PEP 668
  - NumPy installation now works correctly

#### Fixed Audio-Capture Service Dockerfile
- **File**: `services/audio-capture/Dockerfile`
- **Changes**:
  - Fixed SoapySDR Python bindings installation
  - Strategy: Try apt package `python3-soapysdr` first (fastest), fallback to building from source
  - Added build dependencies: `cmake`, `g++`, `git`, `libsoapysdr-dev`, `python3-dev`, `swig`
  - Fixed verification: Removed `__version__` check (SoapySDR doesn't have this attribute)
  - Builds SoapySDR Python bindings from GitHub source if apt package doesn't work
- **Impact**: SoapySDR now installs correctly and works with HackRF
- **Build Time**: ~55 seconds when building from source

### Monitoring and Debugging Tools

#### Enhanced API Stats Endpoint
- **File**: `services/api/server.py`
- **Changes**:
  - Added `audio_chunks_count` to stats response
  - Added `recent_audio` object with details about most recent audio chunk
  - Added `audio_flowing` boolean (true if audio received in last 5 seconds)
  - Imports `STREAM_AUDIO` and `AudioChunk` for audio monitoring
- **Usage**: `curl http://localhost:8000/api/stats`
- **Benefit**: Easy way to verify HackRF audio capture is working on headless systems

#### Created Audio Monitoring Script
- **File**: `monitor-audio.sh`
- **Purpose**: Headless-friendly script to monitor HackRF audio capture
- **Features**:
  - Shows Redis stream statistics (audio chunks, transcripts, callsigns, QSOs)
  - Displays service status
  - Shows recent logs
  - Provides troubleshooting tips
- **Usage**: `./monitor-audio.sh` or `make monitor-audio`
- **Benefit**: Quick way to verify audio is flowing without needing a GUI

#### Added Makefile Command
- **File**: `Makefile`
- **Change**: Added `monitor-audio` target
- **Usage**: `make monitor-audio`

### Configuration Updates

#### LLM Configuration
- **File**: `.env`
- **Changes**:
  - Updated `LLM_BACKEND` from `ollama` to `openai` (for OpenAI-compatible APIs)
  - Updated `LLM_HOST` to `host.docker.internal:5000` (text-generation-webui default)
  - Updated `LLM_MODEL` to `gemma-3-12b-it-abliterated.q5_k_m.gguf`
- **Purpose**: Support for text-generation-webui running on host

#### HackRF Configuration
- **File**: `.env`, `services/audio-capture/config.yaml`
- **Changes**:
  - Set `MODE=hackrf` as default
  - Set `DEMOD_MODE=LSB` for 7.2 MHz
  - Added gain guidance comments in `.env`
- **Settings**:
  - `LNA_GAIN=16` dB (default, good for HF)
  - `VGA_GAIN=20` dB (default, conservative)
  - `FREQUENCY_HZ=7200000` (7.2 MHz, 40m band)

### Code Updates

#### Audio Capture Service
- **File**: `services/audio-capture/capture.py`
- **Changes**: Already updated to read HackRF settings from environment variables
- **Benefit**: Runtime configuration without rebuilding

### Container Management

#### Explicit Container Names
- **File**: `docker-compose.yml`
- **Change**: Added `container_name` to all services
- **Container Names**:
  - `aris-redis`
  - `aris-audio-capture`
  - `aris-stt`
  - `aris-callsign-extractor`
  - `aris-summarizer`
  - `aris-api`
- **Benefit**: 
  - Predictable container names (no more `aris-stt-1` style auto-generated names)
  - Easier to reference containers directly: `docker logs aris-stt`
  - Consistent naming across restarts

---

## Build Notes

### CUDA Image Selection
- **Current**: `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04`
- **Includes**: CUDA 13.0.2, cuDNN (for faster-whisper performance), Ubuntu 24.04
- **Python**: Uses system Python 3.12 (Ubuntu 24.04 default)
- **Why cuDNN**: Provides 2-5x speedup for faster-whisper inference operations

### SoapySDR Installation
- **Method**: Try apt package first, fallback to building from source (GitHub repository)
- **Reason**: Apt package `python3-soapysdr` may not work with Python 3.11 from python:3.11-slim
- **Process**: 
  1. Try importing apt-installed package
  2. If that fails, install build dependencies and build from source
  3. Use `make install` to install Python bindings (built via SWIG)
- **Build Time**: ~55 seconds when building from source
- **Note**: SoapySDR doesn't have `__version__` attribute, verification checks import only

### Python Version Compatibility
- **Audio Capture**: Python 3.11 (from python:3.11-slim base image)
- **STT Service**: Python 3.12 (Ubuntu 24.04 system Python)
- **Other Services**: Python 3.11 (from python:3.11-slim base image)

---

## Troubleshooting

### If SoapySDR build fails:
- Check that all build dependencies are installed
- Verify internet connection (needs to clone from GitHub)
- Check Docker has enough memory/disk space
- The build from source takes ~55 seconds, be patient
- Verify the apt package was tried first (check build logs)

### If NumPy import fails in STT:
- Verify Python version matches (should be 3.12)
- Check pip install completed successfully
- Ensure `--break-system-packages` flag is present (required for Ubuntu 24.04)
- Rebuild with `--no-cache` flag

### If audio not flowing:
- Run `make monitor-audio` to check Redis streams
- Check `docker compose logs aris-audio-capture` for errors (use container name)
- Or use: `docker logs aris-audio-capture`
- Verify HackRF is detected: `lsusb | grep -i hackrf`
- Check USB passthrough in docker-compose.yml is enabled

### Container Name Usage:
- Use explicit container names: `docker logs aris-stt` instead of auto-generated names
- Container names are now predictable: `aris-redis`, `aris-api`, `aris-stt`, etc.
- Names don't change between restarts (no more `-1`, `-2` suffixes)
