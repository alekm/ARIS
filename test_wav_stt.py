#!/usr/bin/env python3
"""
Test script to send a WAV file directly to the STT service via Redis.
This bypasses the audio-capture pipeline for testing.
"""
import sys
import os
import time
import wave
import numpy as np
import redis

# Add shared models to path
sys.path.insert(0, '/opt/stacks/aris')
from shared.models import AudioChunk, STREAM_AUDIO, RedisMessage

def read_wav_file(wav_path):
    """Read WAV file and return audio data and sample rate"""
    with wave.open(wav_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        
        print(f"WAV file info:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {n_channels}")
        print(f"  Sample width: {sample_width} bytes")
        print(f"  Duration: {n_frames / sample_rate:.2f} seconds")
        print(f"  Total frames: {n_frames}")
        
        # Read audio data
        audio_bytes = wav_file.readframes(n_frames)
        
        # Convert to int16 numpy array
        if sample_width == 2:  # 16-bit
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        elif sample_width == 1:  # 8-bit
            audio_array = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.int16) - 128
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Handle stereo -> mono conversion if needed
        if n_channels == 2:
            audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
        
        return audio_array, sample_rate

def send_wav_to_stt(wav_path, frequency_khz=14058.0, mode="CW", source_id="test-wav"):
    """Send WAV file to STT service via Redis stream"""
    
    # Connect to Redis
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
    
    print(f"Connected to Redis at {redis_host}:{redis_port}")
    
    # Read WAV file
    print(f"\nReading WAV file: {wav_path}")
    audio_array, sample_rate = read_wav_file(wav_path)
    
    # Convert to int16 bytes
    audio_bytes = audio_array.tobytes()
    
    # Split into 1-second chunks (matching the real pipeline)
    chunk_duration_seconds = 1.0
    samples_per_chunk = int(sample_rate * chunk_duration_seconds)
    frequency_hz = int(frequency_khz * 1000)
    
    # Determine filter cutoffs based on mode
    if mode == "CW":
        low_cut = 300
        high_cut = 700
    elif mode == "LSB":
        low_cut = -2700
        high_cut = -300
    elif mode == "USB":
        low_cut = 300
        high_cut = 2700
    elif mode == "AM":
        low_cut = -5000
        high_cut = 5000
    else:
        low_cut = 300
        high_cut = 2700
    
    print(f"\nSending audio chunks to STT service...")
    print(f"  Frequency: {frequency_khz} kHz ({frequency_hz} Hz)")
    print(f"  Mode: {mode}")
    print(f"  Filter: {low_cut}-{high_cut} Hz")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Chunk size: {samples_per_chunk} samples ({chunk_duration_seconds}s)")
    
    num_chunks = len(audio_array) // samples_per_chunk
    if len(audio_array) % samples_per_chunk != 0:
        num_chunks += 1
    
    print(f"  Total chunks: {num_chunks}")
    print()
    
    base_timestamp = time.time()
    
    for i in range(num_chunks):
        start_idx = i * samples_per_chunk
        end_idx = min(start_idx + samples_per_chunk, len(audio_array))
        chunk_samples = audio_array[start_idx:end_idx]
        
        # Convert to bytes
        chunk_bytes = chunk_samples.tobytes()
        
        # Calculate duration
        chunk_duration_ms = int((len(chunk_samples) / sample_rate) * 1000)
        
        # Create AudioChunk
        chunk = AudioChunk(
            timestamp=base_timestamp + (i * chunk_duration_seconds),
            frequency_hz=frequency_hz,
            mode=mode,
            sample_rate=sample_rate,
            duration_ms=chunk_duration_ms,
            data=chunk_bytes,
            source_id=source_id,
            s_meter=0.0,
            signal_strength_db=-150.0,
            squelch_open=True,
            low_cut=low_cut,
            high_cut=high_cut,
            seq=i
        )
        
        # Encode and publish to Redis
        message = RedisMessage.encode(chunk)
        redis_client.xadd(STREAM_AUDIO, message, maxlen=1000)
        
        print(f"  Sent chunk {i+1}/{num_chunks} ({chunk_duration_ms}ms, {len(chunk_bytes)} bytes)")
    
    print(f"\n✓ All {num_chunks} chunks sent to STT service!")
    print(f"  Check STT logs: docker compose logs -f stt")
    print(f"  Check transcripts: docker compose logs stt | grep -i transcript")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Send WAV file to STT service for testing')
    parser.add_argument('wav_file', help='Path to WAV file')
    parser.add_argument('--freq', type=float, default=14058.0, help='Frequency in kHz (default: 14058.0)')
    parser.add_argument('--mode', default='CW', choices=['CW', 'USB', 'LSB', 'AM', 'FM'], help='Demodulation mode (default: CW)')
    parser.add_argument('--source-id', default='test-wav', help='Source ID for this test (default: test-wav)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.wav_file):
        print(f"Error: WAV file not found: {args.wav_file}")
        sys.exit(1)
    
    send_wav_to_stt(args.wav_file, args.freq, args.mode, args.source_id)
