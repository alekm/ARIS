# KiwiSDR Setup Guide

## Ready to Switch from HackRF to KiwiSDR

Your ARIS system is now ready to connect to a KiwiSDR! The WebSocket client is fully implemented.

## Quick Switch Steps

### 1. Update `.env` file

```bash
# Change these lines in .env:
MODE=kiwi                          # Switch from hackrf to kiwi
KIWI_HOST=your.kiwi.address.here  # Your KiwiSDR hostname or IP
KIWI_PORT=8073                     # Default port
KIWI_PASSWORD=                     # Leave blank if no password
```

### 2. Restart the stack

```bash
docker compose down && docker compose up -d
```

That's it! The system will now receive **already-demodulated audio** from the KiwiSDR.

## Why KiwiSDR is Better for Testing

✅ **GPS-disciplined clock** - Perfect frequency accuracy
✅ **Server-side demodulation** - Proven, stable
✅ **No frequency drift** - Unlike HackRF's ±20ppm
✅ **Known-good reference** - Industry standard

## KiwiSDR Features Implemented

- ✅ WebSocket connection with authentication
- ✅ All modulation modes (AM, USB, LSB, CW, FM, etc.)
- ✅ Dynamic frequency changes via API/Web UI
- ✅ Dynamic mode changes via API/Web UI
- ✅ Audio streaming with buffering
- ✅ AGC and audio processing
- ✅ Connection keepalive

## Testing After Connection

Once connected, you should see:
```bash
# Check logs
docker compose logs audio-capture | tail -20

# Should see:
# "Connecting to KiwiSDR at ws://..."
# "KiwiSDR WebSocket connected"
# "KiwiSDR configured, starting audio stream"
```

## Real-time Frequency Control

The KiwiSDR mode supports the same API as HackRF:

```bash
# Change frequency (in kHz)
curl -X POST http://localhost:8000/api/control/frequency \
  -H "Content-Type: application/json" \
  -d '{"frequency_khz": 7188}'

# Change mode
curl -X POST http://localhost:8000/api/control/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "LSB"}'
```

Or use the web UI at `http://localhost:8000`

## Troubleshooting

**Connection fails?**
- Check KiwiSDR is accessible from Docker container
- Verify hostname/IP is correct
- Check if KiwiSDR requires password
- Ensure port 8073 is not blocked

**No audio?**
- Check logs: `docker compose logs audio-capture -f`
- Verify KiwiSDR is actually tuned to a frequency with activity
- Try a known active frequency (e.g., WWV on 10000 kHz)

**Authentication errors?**
- Set `KIWI_PASSWORD` in `.env` if your KiwiSDR requires it

## Next Steps

Once you confirm KiwiSDR audio works:
1. Compare with HackRF audio quality
2. Verify STT transcription accuracy
3. Test frequency changes
4. Validate LSB/USB demodulation is correct
