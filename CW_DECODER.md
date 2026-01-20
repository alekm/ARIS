# CW (Morse Code) Decoder

ARIS now includes automatic CW (Morse code) decoding when a slot is configured with mode `CW`. Instead of using Whisper for speech-to-text, the system will automatically decode the Morse code audio signal.

## How It Works

1. **Automatic Detection**: When a slot's mode is set to `CW`, the STT service automatically routes audio to the CW decoder instead of Whisper.

2. **Signal Processing**:
   - Bandpass filtering to isolate the CW tone (typically 400-800 Hz)
   - Envelope detection to identify on/off states
   - Timing analysis to classify dots, dashes, and spaces
   - Auto-detection of tone frequency and WPM (words per minute)

3. **Decoding**: Converts Morse code patterns to text using the International Morse Code alphabet.

## Configuration

### Environment Variables

Add these to your `.env` file or `docker-compose.yml`:

```bash
# Enable/disable CW decoder (default: true)
CW_DECODER_ENABLED=true

# Expected CW tone frequency in Hz (default: 600)
# The decoder will auto-detect if this is off
CW_TONE_FREQ=600

# Expected words per minute (default: 20)
# The decoder will auto-detect if this is off
CW_WPM=20

# Buffer duration for CW decoding in milliseconds (default: 5000 = 5 seconds)
# Longer buffers = better accuracy but more latency
CW_BUFFER_MS=5000
```

### Using the Web UI

1. Configure a slot with mode `CW`
2. Set the frequency to a CW frequency (e.g., 7.023 MHz for 40m CW)
3. Start the slot
4. The system will automatically decode CW signals

### Using the API

```bash
# Start a slot in CW mode
curl -X POST http://localhost:8000/api/slots/1/start \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "kiwi",
    "host": "kiwisdr.example.com",
    "port": 8073,
    "frequency_hz": 7023000,
    "demod_mode": "CW"
  }'
```

## Features

- **Auto-detection**: Automatically detects CW tone frequency and WPM
- **Robust filtering**: Bandpass filtering isolates CW tone from noise
- **Real-time processing**: Processes audio in 5-second buffers (configurable)
- **Standard Morse**: Supports International Morse Code alphabet including letters, numbers, and punctuation
- **Confidence scoring**: Provides confidence scores similar to Whisper transcripts

## Technical Details

### Algorithm

1. **Filtering**: Butterworth bandpass filter (4th order) around the CW tone frequency
2. **Envelope Detection**: Uses Hilbert transform for accurate envelope detection
3. **Threshold Detection**: Adaptive threshold based on signal statistics
4. **Timing Classification**: 
   - Dot: ~1 unit duration
   - Dash: ~3 units duration
   - Character space: ~3 units
   - Word space: ~7 units
5. **Decoding**: Lookup table converts Morse patterns to text

### Performance

- **Latency**: ~5 seconds (configurable via `CW_BUFFER_MS`)
- **Accuracy**: Good for clean signals, degrades with noise
- **CPU Usage**: Low (no GPU required, unlike Whisper)

## Limitations

- Works best with clean, consistent CW signals
- Accuracy decreases with:
  - High noise levels
  - Variable keying speed
  - Multiple overlapping signals
  - Very fast (>40 WPM) or very slow (<10 WPM) speeds
- Requires sufficient signal strength (similar to speech recognition)

## Troubleshooting

### No transcripts appearing

1. Check that mode is set to `CW` (not `USB`, `LSB`, etc.)
2. Verify signal strength - CW decoder needs clear on/off states
3. Check logs: `docker-compose logs -f stt | grep CW`
4. Try adjusting `CW_TONE_FREQ` if tone frequency is non-standard
5. Increase `CW_BUFFER_MS` for slower CW speeds

### Poor accuracy

1. Adjust `CW_TONE_FREQ` to match actual tone frequency
2. Adjust `CW_WPM` to match keying speed (or let auto-detection handle it)
3. Check audio quality - ensure good signal-to-noise ratio
4. Verify filter settings in audio-capture (should be 400-800 Hz for CW)

### Example Log Output

```
[Slot 1] Mode changed: USB -> CW
[Slot 1] Initialized CW decoder
[Slot 1] Transcribing 5000ms (80000 samples), mode=CW
[Slot 1] Detected CW tone frequency: 601.2 Hz
[Slot 1] Auto-detected WPM: 21.3
[Slot 1] CW decoded: 'CQ CQ DE W1AW' (confidence: 0.85)
[Slot 1] Published transcript
```

## Future Improvements

Potential enhancements:
- Machine learning-based decoding for noisy signals
- Support for Farnsworth timing (slow characters, fast spacing)
- Real-time streaming (lower latency)
- Multiple tone detection (split signals)
- Integration with callsign extraction for automatic logging
