#!/usr/bin/env python3
"""
Audio Capture Service
Captures audio from KiwiSDR, HackRF, or mock source and streams to Redis
"""
import os
import sys
import time
import logging
import numpy as np
import soundfile as sf
import redis
import yaml
from pathlib import Path
from datetime import datetime
from scipy import signal as scipy_signal

# SoapySDR is optional - only needed for HackRF mode
try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False

sys.path.insert(0, '/app')
from shared.models import AudioChunk, STREAM_AUDIO, RedisMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockAudioSource:
    """Mock audio source for testing without KiwiSDR"""

    def __init__(self, config):
        self.frequency = config.get('frequency_hz', 7200000)
        self.mode = config.get('demod_mode', config.get('mode', 'USB'))
        self.sample_rate = config.get('sample_rate', 16000)
        self.chunk_duration = config.get('chunk_duration_ms', 1000)
        self.chunk_samples = int(self.sample_rate * self.chunk_duration / 1000)
        logger.info(f"Mock source: {self.frequency} Hz, {self.mode}, {self.sample_rate} Hz")

    def generate_test_tone(self, duration_ms):
        """Generate a test tone (simulated voice frequencies)"""
        samples = int(self.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, samples)

        # Mix of tones in voice range (300-3000 Hz)
        signal = (
            0.3 * np.sin(2 * np.pi * 500 * t) +
            0.2 * np.sin(2 * np.pi * 1200 * t) +
            0.1 * np.sin(2 * np.pi * 2500 * t)
        )

        # Add some noise to simulate RF
        noise = np.random.normal(0, 0.05, samples)
        signal = signal + noise

        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8

        # Convert to int16 PCM
        pcm = (signal * 32767).astype(np.int16)
        return pcm.tobytes()

    def read_chunk(self):
        """Return a chunk of mock audio"""
        audio_data = self.generate_test_tone(self.chunk_duration)

        return AudioChunk(
            timestamp=time.time(),
            frequency_hz=self.frequency,
            mode=self.mode,
            sample_rate=self.sample_rate,
            duration_ms=self.chunk_duration,
            data=audio_data
        )


class KiwiSDRAudioSource:
    """Real KiwiSDR audio source"""

    def __init__(self, config):
        self.host = config['kiwi_host']
        self.port = config.get('kiwi_port', 8073)
        self.frequency = config['frequency_hz']
        self.mode = config.get('demod_mode', config.get('mode', 'USB'))
        self.sample_rate = config.get('sample_rate', 12000)
        self.chunk_duration = config.get('chunk_duration_ms', 1000)

        logger.info(f"KiwiSDR source: {self.host}:{self.port}, {self.frequency} Hz, {self.mode}")

        # TODO: Implement actual KiwiSDR connection
        # Will use kiwirecorder.py or custom WebSocket client
        raise NotImplementedError("KiwiSDR connection not yet implemented - use MODE=mock")

    def read_chunk(self):
        """Return a chunk from KiwiSDR"""
        # TODO: Read from KiwiSDR WebSocket stream
        pass


# =============================================================================
# Demodulation Functions
# =============================================================================

def demodulate_usb(iq_samples, rf_rate, audio_rate):
    """
    Demodulate USB (Upper Sideband) from IQ samples using Weaver method.

    Args:
        iq_samples: Complex IQ samples from SDR
        rf_rate: RF sample rate in Hz
        audio_rate: Target audio sample rate in Hz (e.g., 16000)

    Returns:
        Real-valued audio samples at audio_rate
    """
    # USB: shift baseband up by ~1.5kHz, then low-pass filter
    # Weaver method: multiply by complex exponential to shift spectrum
    t = np.arange(len(iq_samples)) / rf_rate
    shift_freq = 1500  # Center of voice passband
    shifted = iq_samples * np.exp(1j * 2 * np.pi * shift_freq * t)

    # Take real part (this gives us USB)
    audio = np.real(shifted)

    # Low-pass filter before decimation (anti-aliasing)
    # Cutoff at ~3kHz for voice
    nyq = rf_rate / 2
    cutoff = 3000 / nyq
    b, a = scipy_signal.butter(5, cutoff, btype='low')
    audio = scipy_signal.filtfilt(b, a, audio)

    # Multi-stage decimation for large ratios
    audio = _decimate_multistage(audio, rf_rate, audio_rate)

    return audio


def demodulate_lsb(iq_samples, rf_rate, audio_rate):
    """
    Demodulate LSB (Lower Sideband) from IQ samples.
    """
    # LSB: shift baseband down, then low-pass filter
    t = np.arange(len(iq_samples)) / rf_rate
    shift_freq = -1500  # Negative shift for LSB
    shifted = iq_samples * np.exp(1j * 2 * np.pi * shift_freq * t)

    # Take real part
    audio = np.real(shifted)

    # Low-pass filter
    nyq = rf_rate / 2
    cutoff = 3000 / nyq
    b, a = scipy_signal.butter(5, cutoff, btype='low')
    audio = scipy_signal.filtfilt(b, a, audio)

    # Decimate
    audio = _decimate_multistage(audio, rf_rate, audio_rate)

    return audio


def demodulate_am(iq_samples, rf_rate, audio_rate):
    """
    Demodulate AM (Amplitude Modulation) from IQ samples.
    Uses envelope detection (magnitude of complex signal).
    """
    # AM envelope detection
    audio = np.abs(iq_samples)

    # Remove DC component
    audio = audio - np.mean(audio)

    # Low-pass filter for voice (~5kHz for AM broadcast)
    nyq = rf_rate / 2
    cutoff = 5000 / nyq
    b, a = scipy_signal.butter(5, cutoff, btype='low')
    audio = scipy_signal.filtfilt(b, a, audio)

    # Decimate
    audio = _decimate_multistage(audio, rf_rate, audio_rate)

    return audio


def demodulate_fm(iq_samples, rf_rate, audio_rate):
    """
    Demodulate FM (Frequency Modulation) from IQ samples.
    Uses frequency discriminator (phase difference).
    """
    # FM demodulation via phase difference
    # d/dt(phase) = frequency
    phase = np.angle(iq_samples)
    # Unwrap phase to avoid discontinuities
    phase = np.unwrap(phase)
    # Differentiate to get frequency
    audio = np.diff(phase)
    # Pad to maintain length
    audio = np.append(audio, audio[-1])

    # Low-pass filter (~15kHz for NFM)
    nyq = rf_rate / 2
    cutoff = min(15000, nyq * 0.9) / nyq
    b, a = scipy_signal.butter(5, cutoff, btype='low')
    audio = scipy_signal.filtfilt(b, a, audio)

    # Decimate
    audio = _decimate_multistage(audio, rf_rate, audio_rate)

    return audio


def _decimate_multistage(audio, from_rate, to_rate):
    """
    Decimate in multiple stages for better filter performance.
    Large decimation ratios (>10) benefit from multi-stage approach.
    """
    ratio = from_rate / to_rate

    if ratio <= 1:
        return audio

    # Use scipy.signal.decimate with multiple stages
    # Maximum decimation per stage is ~13 for good filter response
    max_stage_ratio = 10

    current_rate = from_rate
    while current_rate / to_rate > max_stage_ratio:
        # Decimate by max_stage_ratio
        audio = scipy_signal.decimate(audio, max_stage_ratio, ftype='fir')
        current_rate = current_rate / max_stage_ratio

    # Final decimation to target rate
    final_ratio = int(round(current_rate / to_rate))
    if final_ratio > 1:
        audio = scipy_signal.decimate(audio, final_ratio, ftype='fir')

    return audio


# =============================================================================
# HackRF Audio Source
# =============================================================================

class HackRFAudioSource:
    """HackRF Pro audio source using SoapySDR"""

    def __init__(self, config):
        if not SOAPY_AVAILABLE:
            raise ImportError("SoapySDR not available - install python3-soapysdr and soapysdr-module-hackrf")

        self.frequency = config['frequency_hz']
        self.mode = config.get('demod_mode', config.get('mode', 'USB'))
        self.audio_rate = config.get('sample_rate', 16000)
        self.chunk_duration = config.get('chunk_duration_ms', 1000)

        # HackRF-specific settings
        self.rf_rate = config.get('rf_sample_rate', 2000000)  # 2 MS/s default
        self.lna_gain = config.get('lna_gain', 16)  # 0-40 dB, 8 dB steps
        self.vga_gain = config.get('vga_gain', 20)  # 0-62 dB, 2 dB steps
        self.bandwidth = config.get('bandwidth', 1750000)  # RF bandwidth filter
        self.device_serial = config.get('hackrf_serial', None)

        # Calculate samples needed per chunk
        self.rf_samples_per_chunk = int(self.rf_rate * self.chunk_duration / 1000)

        # Select demodulator based on mode
        self.demodulator = {
            'USB': demodulate_usb,
            'LSB': demodulate_lsb,
            'AM': demodulate_am,
            'FM': demodulate_fm,
        }.get(self.mode.upper(), demodulate_usb)

        # Initialize SoapySDR device
        self._init_device()

        logger.info(f"HackRF source: {self.frequency} Hz, {self.mode}, RF rate {self.rf_rate/1e6:.1f} MS/s")

    def _init_device(self):
        """Initialize HackRF via SoapySDR"""
        # Find HackRF device
        args = {'driver': 'hackrf'}
        if self.device_serial:
            args['serial'] = self.device_serial

        self.sdr = SoapySDR.Device(args)

        # Configure RX channel
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.rf_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.frequency)
        self.sdr.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)

        # Set gains
        self.sdr.setGain(SOAPY_SDR_RX, 0, 'LNA', self.lna_gain)
        self.sdr.setGain(SOAPY_SDR_RX, 0, 'VGA', self.vga_gain)

        # Setup RX stream
        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rx_stream)

        logger.info(f"HackRF initialized: LNA={self.lna_gain}dB, VGA={self.vga_gain}dB")

    def read_chunk(self):
        """Read IQ samples from HackRF and return demodulated audio chunk"""
        # Allocate buffer for IQ samples
        iq_buffer = np.zeros(self.rf_samples_per_chunk, dtype=np.complex64)

        # Read samples (may need multiple reads to fill buffer)
        samples_read = 0
        max_retries = 10
        retries = 0

        while samples_read < self.rf_samples_per_chunk and retries < max_retries:
            remaining = self.rf_samples_per_chunk - samples_read
            sr = self.sdr.readStream(
                self.rx_stream,
                [iq_buffer[samples_read:]],
                remaining,
                timeoutUs=1000000  # 1 second timeout
            )

            if sr.ret > 0:
                samples_read += sr.ret
                retries = 0
            else:
                retries += 1
                logger.warning(f"HackRF read returned {sr.ret}, retry {retries}")
                time.sleep(0.01)

        if samples_read < self.rf_samples_per_chunk:
            logger.warning(f"Short read: got {samples_read}/{self.rf_samples_per_chunk} samples")

        # Demodulate to audio
        audio = self.demodulator(iq_buffer[:samples_read], self.rf_rate, self.audio_rate)

        # Normalize and convert to int16 PCM
        if len(audio) > 0:
            audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.8
        pcm = (audio * 32767).astype(np.int16)

        return AudioChunk(
            timestamp=time.time(),
            frequency_hz=self.frequency,
            mode=self.mode,
            sample_rate=self.audio_rate,
            duration_ms=self.chunk_duration,
            data=pcm.tobytes()
        )

    def close(self):
        """Clean up HackRF resources"""
        if hasattr(self, 'rx_stream'):
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)
        logger.info("HackRF closed")


class AudioCaptureService:
    """Main audio capture service"""

    def __init__(self, config_path='config.yaml'):
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Connect to Redis
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

        # Initialize audio source
        mode = os.getenv('MODE', self.config.get('mode', 'mock'))

        if mode == 'mock':
            self.source = MockAudioSource(self.config)
        elif mode == 'kiwi':
            self.source = KiwiSDRAudioSource(self.config)
        elif mode == 'hackrf':
            self.source = HackRFAudioSource(self.config)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.running = False

    def publish_chunk(self, chunk: AudioChunk):
        """Publish audio chunk to Redis stream"""
        try:
            msg = RedisMessage.encode(chunk)
            self.redis.xadd(STREAM_AUDIO, msg, maxlen=1000)  # Keep last 1000 chunks
            logger.debug(f"Published chunk: {chunk.timestamp}, {len(chunk.data)} bytes")
        except Exception as e:
            logger.error(f"Failed to publish chunk: {e}")

    def run(self):
        """Main capture loop"""
        self.running = True
        logger.info("Starting audio capture service...")

        chunk_count = 0
        start_time = time.time()

        try:
            while self.running:
                chunk = self.source.read_chunk()
                self.publish_chunk(chunk)

                chunk_count += 1
                if chunk_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = chunk_count / elapsed
                    logger.info(f"Captured {chunk_count} chunks ({rate:.1f}/sec)")

                # Sleep to maintain real-time rate
                time.sleep(chunk.duration_ms / 1000 * 0.9)  # Slight underrun

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            self.running = False

    def stop(self):
        self.running = False


if __name__ == '__main__':
    service = AudioCaptureService()
    service.run()
