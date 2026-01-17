#!/bin/bash
# ARIS Quick Setup Script

set -e

echo "🎯 ARIS Setup"
echo "============="
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check for Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker found"
echo "✅ Docker Compose found"
echo ""

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{audio,transcripts,db,redis,summaries}
echo "✅ Data directories created"
echo ""

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "⚠️  Please edit .env to configure your settings"
else
    echo "ℹ️  .env file already exists"
fi
echo ""

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected. STT will run on CPU (slower)"
fi
echo ""

# Check for Ollama
if command -v ollama &> /dev/null; then
    echo "✅ Ollama found"
    echo "   Current models:"
    ollama list 2>/dev/null || echo "   (no models installed)"
else
    echo "⚠️  Ollama not found. Install from https://ollama.ai"
    echo "   Or configure a different LLM backend in .env"
fi
echo ""

echo "🚀 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your configuration"
echo "  2. If using KiwiSDR, edit services/audio-capture/config.yaml"
echo "  3. Run: make up    (or: docker compose up -d)"
echo "  4. View logs: make logs"
echo "  5. Open web UI: http://localhost:8000"
echo ""
echo "For testing without KiwiSDR:"
echo "  make test-mock"
echo ""
