#!/bin/bash
# Quick script to monitor HackRF audio capture on headless system

echo "=== ARIS Audio Capture Monitor ==="
echo ""

# Check if Redis is accessible
if ! docker compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not accessible. Is the service running?"
    exit 1
fi

echo "📊 Redis Stream Statistics:"
echo ""

# Get stream lengths
AUDIO_COUNT=$(docker compose exec -T redis redis-cli XLEN audio:chunks 2>/dev/null | tr -d '\r')
TRANSCRIPTS_COUNT=$(docker compose exec -T redis redis-cli XLEN transcripts 2>/dev/null | tr -d '\r')
CALLSIGNS_COUNT=$(docker compose exec -T redis redis-cli XLEN callsigns 2>/dev/null | tr -d '\r')
QSOS_COUNT=$(docker compose exec -T redis redis-cli XLEN qsos 2>/dev/null | tr -d '\r')

echo "  Audio Chunks:    ${AUDIO_COUNT:-0}"
echo "  Transcripts:     ${TRANSCRIPTS_COUNT:-0}"
echo "  Callsigns:       ${CALLSIGNS_COUNT:-0}"
echo "  QSOs:            ${QSOS_COUNT:-0}"
echo ""

# Check for recent audio
if [ "${AUDIO_COUNT:-0}" -gt 0 ]; then
    echo "✅ Audio chunks detected in Redis!"
    echo ""
    echo "📡 Most recent audio chunk info:"
    
    # Get the most recent chunk ID
    LAST_ID=$(docker compose exec -T redis redis-cli XREVRANGE audio:chunks + - COUNT 1 2>/dev/null | head -1 | awk '{print $1}')
    
    if [ -n "$LAST_ID" ] && [ "$LAST_ID" != "(empty" ]; then
        echo "  Last chunk ID: $LAST_ID"
        echo "  (Audio is flowing ✅)"
    else
        echo "  (Could not retrieve chunk details)"
    fi
else
    echo "⚠️  No audio chunks found in Redis"
    echo "   This could mean:"
    echo "   - Audio capture service is not running"
    echo "   - HackRF is not connected/configured"
    echo "   - No audio is being received"
fi

echo ""
echo "📋 Service Status:"
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAME|audio-capture|stt|redis"

echo ""
echo "📝 Recent Audio Capture Logs (last 10 lines):"
docker compose logs --tail=10 audio-capture 2>&1 | tail -10

echo ""
echo "💡 Tips:"
echo "  - Watch logs in real-time: docker compose logs -f audio-capture"
echo "  - Check API stats: curl http://localhost:8000/api/stats"
echo "  - Verify HackRF: lsusb | grep -i hackrf"
