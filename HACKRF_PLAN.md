# HackRF Pro Integration Plan

## Overview

Add HackRF Pro USB SDR support to ARIS, enabling local radio monitoring via USB-connected HackRF device. This includes implementing IQ sample capture, software demodulation, and proper Docker USB device passthrough.

## Architecture Changes

The implementation will add a new `HackRFAudioSource` class following the existing pattern in [services/audio-capture/capture.py](services/audio-capture/capture.py), similar to `MockAudioSource` and `KiwiSDRAudioSource`.

## Implementation Details

### 1. Dependencies and Libraries

**Option A: SoapySDR (Recommended)**
- More flexible, supports multiple SDR devices
- Requires `SoapySDR` Python bindings
- System dependencies: `libsoapysdr0`, `soapysdr-module-hackrf`

**Option B: HackRF Python Library**
- Direct HackRF support via `hackrf` Python package
- Simpler but HackRF-specific

**Decision:** Use SoapySDR for flexibility and future extensibility.

### 2. Demodulation Implementation

HackRF outputs IQ (complex) samples at RF sample rates (typically 8-20 MS/s). Need to:
- Tune to target frequency
- Demodulate based on mode (USB/LSB/AM/FM)
- Downsample to 16 kHz audio sample rate
- Convert to int16 PCM format

**Demodulation Methods:**
- **USB/LSB:** Mix with carrier, low-pass filter, decimate
- **AM:** Envelope detection (magnitude of complex signal)
- **FM:** Frequency discriminator (phase difference)

### 3. Configuration Updates

Add HackRF-specific settings to [services/audio-capture/config.yaml](services/audio-capture/config.yaml):
- `hackrf_serial`: Device serial number (optional, for multiple devices)
- `rf_sample_rate`: RF sample rate (e.g., 8000000 Hz)
- `lna_gain`: LNA gain in dB (0-40, 8dB steps)
- `vga_gain`: VGA gain in dB (0-62, 2dB steps)
- `bandwidth`: RF bandwidth filter (optional)

### 4. Docker Configuration

Update [docker-compose.yml](docker-compose.yml) to:
- Add USB device passthrough for HackRF
- Mount udev rules if needed for device permissions
- Add environment variable for HackRF mode

### 5. Code Structure

**New Class: `HackRFAudioSource`**
- Initialize SoapySDR device
- Configure frequency, gain, sample rate
- Implement `read_chunk()` method that:
  1. Reads IQ samples from HackRF
  2. Demodulates based on mode
  3. Downsamples to 16 kHz
  4. Returns `AudioChunk` object

**Demodulation Helper Functions:**
- `demodulate_usb(iq_samples, rf_rate, audio_rate)`
- `demodulate_lsb(iq_samples, rf_rate, audio_rate)`
- `demodulate_am(iq_samples, rf_rate, audio_rate)`
- `demodulate_fm(iq_samples, rf_rate, audio_rate)`

## Files to Modify

1. **[services/audio-capture/capture.py](services/audio-capture/capture.py)**
   - Add `HackRFAudioSource` class
   - Add demodulation functions
   - Update `AudioCaptureService.__init__()` to support `mode == 'hackrf'`

2. **[services/audio-capture/config.yaml](services/audio-capture/config.yaml)**
   - Add HackRF configuration section
   - Add gain and sample rate settings

3. **[services/audio-capture/Dockerfile](services/audio-capture/Dockerfile)**
   - Install SoapySDR system packages
   - Install Python SoapySDR bindings
   - Install scipy for signal processing

4. **[services/audio-capture/requirements.txt](services/audio-capture/requirements.txt)**
   - Add `SoapySDR>=0.8.0`
   - Add `scipy>=1.10.0` (for filtering/decimation)

5. **[docker-compose.yml](docker-compose.yml)**
   - Add USB device passthrough to audio-capture service
   - Add `MODE=hackrf` option in comments

6. **[README.md](README.md)**
   - Document HackRF setup and configuration
   - Add HackRF to prerequisites

## Technical Considerations

### Sample Rate Handling
- HackRF RF sample rate: 8-20 MS/s (configurable)
- Target audio rate: 16 kHz
- Decimation ratio: ~500-1250x
- Use scipy's `decimate()` or custom FIR filter for clean downsampling

### Gain Control
- LNA gain: 0-40 dB in 8 dB steps (controls front-end amplification)
- VGA gain: 0-62 dB in 2 dB steps (controls baseband amplification)
- Auto-gain or manual configuration via config file

### Error Handling
- Device not found errors
- USB disconnection recovery
- Sample rate/bandwidth mismatches
- Gain overload detection

### Performance
- Real-time processing requirements
- Buffer management to avoid underruns
- Efficient numpy/scipy operations

## Testing Strategy

1. **Unit Tests:** Mock SoapySDR device for demodulation functions
2. **Integration:** Test with actual HackRF Pro device
3. **Mode Testing:** Verify USB/LSB/AM/FM demodulation
4. **Gain Testing:** Test various gain settings for signal quality
5. **Frequency Testing:** Verify tuning accuracy across bands

## Implementation Tasks

1. **Add SoapySDR and scipy dependencies** to requirements.txt and Dockerfile
2. **Implement demodulation functions** (USB/LSB/AM/FM) with proper downsampling to 16kHz
3. **Create HackRFAudioSource class** with SoapySDR device initialization and read_chunk() method
4. **Add HackRF configuration options** to config.yaml (gains, sample rates, device selection)
5. **Update docker-compose.yml** with USB device passthrough and HackRF mode support
6. **Integrate HackRFAudioSource** into AudioCaptureService with mode=hackrf support
7. **Update README.md** with HackRF setup instructions and configuration examples

## Future Enhancements

- Multi-frequency monitoring (switch between frequencies)
- Automatic gain control (AGC)
- Bandwidth optimization per band
- Support for other SoapySDR-compatible devices (RTL-SDR, etc.)
