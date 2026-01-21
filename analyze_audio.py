#!/usr/bin/env python3
"""
Compare audio characteristics between WAV file and live stream
"""
import sys
import os
import wave
import numpy as np
import redis
from scipy import signal

sys.path.insert(0, '/opt/stacks/aris')
from shared.models import AudioChunk, STREAM_AUDIO, RedisMessage

def analyze_audio(audio_samples, sample_rate, name):
    """Analyze audio characteristics"""
    print(f"\n=== {name} ===")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio_samples) / sample_rate:.2f} seconds")
    print(f"Samples: {len(audio_samples)}")
    
    # Convert to float32 for analysis
    audio_float = audio_samples.astype(np.float32) / 32768.0
    
    # RMS
    rms = np.sqrt(np.mean(audio_float ** 2))
    print(f"RMS: {rms:.6f}")
    
    # Peak
    peak = np.max(np.abs(audio_float))
    print(f"Peak: {peak:.6f}")
    
    # Dynamic range
    dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
    print(f"Dynamic range: {dynamic_range:.2f} dB")
    
    # Frequency analysis
    fft = np.fft.rfft(audio_float)
    freqs = np.fft.rfftfreq(len(audio_float), 1/sample_rate)
    magnitude = np.abs(fft)
    
    # Find dominant frequency
    dominant_idx = np.argmax(magnitude[1:]) + 1  # Skip DC
    dominant_freq = freqs[dominant_idx]
    dominant_mag = magnitude[dominant_idx]
    print(f"Dominant frequency: {dominant_freq:.1f} Hz (magnitude: {dominant_mag:.2f})")
    
    # Energy in CW band (300-700 Hz)
    cw_band_mask = (freqs >= 300) & (freqs <= 700)
    cw_energy = np.sum(magnitude[cw_band_mask])
    total_energy = np.sum(magnitude[1:])  # Skip DC
    cw_ratio = cw_energy / total_energy if total_energy > 0 else 0
    print(f"Energy in CW band (300-700 Hz): {cw_ratio*100:.2f}%")
    
    # Signal statistics
    signal_on = np.sum(np.abs(audio_float) > (rms * 2))  # Samples above 2x RMS
    signal_ratio = signal_on / len(audio_float)
    print(f"Signal on ratio (>2x RMS): {signal_ratio*100:.2f}%")
    
    return {
        'rms': rms,
        'peak': peak,
        'dominant_freq': dominant_freq,
        'cw_ratio': cw_ratio,
        'signal_ratio': signal_ratio
    }

def read_wav(wav_path):
    """Read WAV file"""
    with wave.open(wav_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(n_frames)
        
        if wav_file.getsampwidth() == 2:  # 16-bit
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {wav_file.getsampwidth()}")
        
        # Convert stereo to mono if needed
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
        
        return audio, sample_rate

def get_live_audio(redis_client, source_id="1", num_chunks=5):
    """Get recent audio chunks from Redis"""
    messages = redis_client.xrevrange(STREAM_AUDIO, count=100)
    
    audio_chunks = []
    for msg_id, data in messages:
        try:
            chunk = RedisMessage.decode(data, AudioChunk)
            if chunk.source_id == source_id and chunk.mode == "CW":
                audio_chunks.append(chunk)
                if len(audio_chunks) >= num_chunks:
                    break
        except:
            continue
    
    if not audio_chunks:
        print("No CW audio chunks found in Redis")
        return None, None
    
    # Concatenate chunks
    all_samples = []
    sample_rate = audio_chunks[0].sample_rate
    
    for chunk in reversed(audio_chunks):  # Reverse to get chronological order
        samples = np.frombuffer(chunk.data, dtype=np.int16)
        all_samples.append(samples)
    
    audio = np.concatenate(all_samples)
    return audio, sample_rate

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare audio characteristics')
    parser.add_argument('wav_file', help='Path to WAV file from web UI')
    parser.add_argument('--source-id', default='1', help='Source ID for live audio (default: 1)')
    
    args = parser.parse_args()
    
    # Analyze WAV file
    wav_audio, wav_sr = read_wav(args.wav_file)
    wav_stats = analyze_audio(wav_audio, wav_sr, "WAV File (Web UI)")
    
    # Analyze live audio
    redis_host = os.getenv('REDIS_HOST', 'redis')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
    live_audio, live_sr = get_live_audio(redis_client, args.source_id)
    
    if live_audio is not None:
        live_stats = analyze_audio(live_audio, live_sr, "Live Stream (Our Capture)")
        
        # Compare
        print("\n=== COMPARISON ===")
        print(f"RMS ratio (WAV/Live): {wav_stats['rms'] / live_stats['rms']:.2f}x")
        print(f"Peak ratio (WAV/Live): {wav_stats['peak'] / live_stats['peak']:.2f}x")
        print(f"CW band energy ratio (WAV/Live): {wav_stats['cw_ratio'] / live_stats['cw_ratio']:.2f}x")
        print(f"Signal ratio (WAV/Live): {wav_stats['signal_ratio'] / live_stats['signal_ratio']:.2f}x")
        print(f"Dominant freq difference: {abs(wav_stats['dominant_freq'] - live_stats['dominant_freq']):.1f} Hz")
    else:
        print("\nCould not retrieve live audio for comparison")
