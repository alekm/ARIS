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
from shared.models import AudioChunk, STREAM_AUDIO, STREAM_CONTROL, RedisMessage

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
    """Real KiwiSDR audio source via WebSocket"""

    def __init__(self, config):
        self.host = config['kiwi_host']
        self.port = config.get('kiwi_port', 8073)
        self.password = config.get('kiwi_password', None)

        # Read from environment variables first
        self.frequency = int(os.getenv('FREQUENCY_HZ', config.get('frequency_hz', 7200000)))
        self.mode = os.getenv('DEMOD_MODE', config.get('demod_mode', config.get('mode', 'USB')))
        self.sample_rate = config.get('sample_rate', 12000)
        self.chunk_duration = config.get('chunk_duration_ms', 1000)

        # Audio buffer
        self.audio_buffer = []
        self.buffer_lock = __import__('threading').Lock()

        logger.info(f"KiwiSDR source: {self.host}:{self.port}, {self.frequency} Hz, {self.mode}")

        # Start WebSocket connection in background thread
        self._init_connection()

    def _init_connection(self):
        """Initialize WebSocket connection to KiwiSDR"""
        import asyncio
        import websockets
        import threading
        import struct
        import base64

        self.ws = None
        self.running = True

        async def connect_and_stream():
            """Connect to KiwiSDR and stream audio"""
            try:
                url = f"ws://{self.host}:{self.port}/kiwi/{int(time.time() * 1000)}/SND"
                logger.info(f"Connecting to KiwiSDR at {url}")

                async with websockets.connect(url, max_size=None) as ws:
                    self.ws = ws
                    logger.info("KiwiSDR WebSocket connected")

                    # Send initial setup commands
                    # Set authentication if password is provided
                    if self.password:
                        await ws.send(f"SET auth t=kiwi p={self.password}")

                    # Set modulation mode
                    mode_num = {'AM': 0, 'AMN': 1, 'USB': 2, 'LSB': 3, 'CW': 4, 'CWN': 5, 'NBFMorIQ': 6, 'DRM': 7}.get(self.mode.upper(), 2)
                    await ws.send(f"SET mod={mode_num} low_cut=300 high_cut=3000 freq={self.frequency / 1000.0:.3f}")

                    # Set audio sample rate
                    await ws.send(f"SET AR OK in={self.sample_rate} out=48000")

                    # Start audio streaming
                    await ws.send("SET agc=1 hang=0 thresh=-100 slope=6 decay=1000 gain=50")
                    await ws.send("SET squelch=0 max=0")
                    await ws.send("SET OVERRIDE port=8073")

                    logger.info("KiwiSDR configured, starting audio stream")

                    # Receive audio data
                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)

                            if isinstance(msg, bytes):
                                # Audio data comes as binary
                                # KiwiSDR sends: 'SND' header + flags + sequence + audio samples
                                if len(msg) > 4 and msg[:3] == b'SND':
                                    # Extract audio samples (int16 PCM)
                                    audio_data = msg[4:]  # Skip header
                                    samples = np.frombuffer(audio_data, dtype=np.int16)

                                    with self.buffer_lock:
                                        self.audio_buffer.extend(samples)

                        except asyncio.TimeoutError:
                            # Send keepalive
                            await ws.send("SET keepalive")
                        except Exception as e:
                            logger.error(f"Error receiving KiwiSDR data: {e}")
                            break

            except Exception as e:
                logger.error(f"KiwiSDR connection error: {e}", exc_info=True)
                self.running = False

        def run_async_loop():
            """Run asyncio loop in thread"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(connect_and_stream())
            finally:
                loop.close()

        # Start WebSocket thread
        self.ws_thread = threading.Thread(target=run_async_loop, daemon=True)
        self.ws_thread.start()

        # Wait for connection
        time.sleep(2)

    def set_frequency(self, frequency_hz):
        """Change frequency dynamically"""
        self.frequency = int(frequency_hz)
        logger.info(f"KiwiSDR frequency changed to {self.frequency} Hz")
        # Note: Would need to send command via WebSocket
        # For now, requires reconnection
        return True

    def set_mode(self, mode):
        """Change demodulation mode dynamically"""
        self.mode = mode.upper()
        logger.info(f"KiwiSDR mode changed to {self.mode}")
        # Note: Would need to send command via WebSocket
        # For now, requires reconnection
        return True

    def read_chunk(self):
        """Return a chunk from KiwiSDR audio buffer"""
        chunk_samples = int(self.sample_rate * self.chunk_duration / 1000)

        # Wait for enough samples
        max_wait = 5  # seconds
        wait_start = time.time()
        while len(self.audio_buffer) < chunk_samples:
            if time.time() - wait_start > max_wait:
                logger.warning(f"Timeout waiting for KiwiSDR audio (have {len(self.audio_buffer)}/{chunk_samples} samples)")
                # Return silent chunk
                pcm = np.zeros(chunk_samples, dtype=np.int16)
                return AudioChunk(
                    timestamp=time.time(),
                    frequency_hz=self.frequency,
                    mode=self.mode,
                    sample_rate=self.sample_rate,
                    duration_ms=self.chunk_duration,
                    data=pcm.tobytes()
                )
            time.sleep(0.1)

        # Extract chunk from buffer
        with self.buffer_lock:
            chunk_data = np.array(self.audio_buffer[:chunk_samples], dtype=np.int16)
            self.audio_buffer = self.audio_buffer[chunk_samples:]

        return AudioChunk(
            timestamp=time.time(),
            frequency_hz=self.frequency,
            mode=self.mode,
            sample_rate=self.sample_rate,
            duration_ms=self.chunk_duration,
            data=chunk_data.tobytes()
        )

    def close(self):
        """Close KiwiSDR connection"""
        self.running = False
        if hasattr(self, 'ws_thread'):
            self.ws_thread.join(timeout=2)
        logger.info("KiwiSDR closed")


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
    # LSB: Conjugate to flip spectrum (negative freqs become positive)
    # Then shift down to center the voice band
    iq_conj = np.conj(iq_samples)

    t = np.arange(len(iq_conj)) / rf_rate
    shift_freq = -1500  # Shift down to center LSB voice
    shifted = iq_conj * np.exp(1j * 2 * np.pi * shift_freq * t)

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

        # Read from environment variables first, then fall back to config file
        self.frequency = int(os.getenv('FREQUENCY_HZ', config.get('frequency_hz', 7200000)))
        self.mode = os.getenv('DEMOD_MODE', config.get('demod_mode', config.get('mode', 'USB')))
        self.audio_rate = config.get('sample_rate', 16000)
        self.chunk_duration = config.get('chunk_duration_ms', 1000)

        # HackRF-specific settings (env vars override config)
        self.rf_rate = int(os.getenv('RF_SAMPLE_RATE', config.get('rf_sample_rate', 2000000)))  # 2 MS/s default
        self.lna_gain = int(os.getenv('LNA_GAIN', config.get('lna_gain', 16)))  # 0-40 dB, 8 dB steps
        self.vga_gain = int(os.getenv('VGA_GAIN', config.get('vga_gain', 20)))  # 0-62 dB, 2 dB steps
        self.bandwidth = int(os.getenv('BANDWIDTH', config.get('bandwidth', 1750000)))  # RF bandwidth filter
        hackrf_serial_env = os.getenv('HACKRF_SERIAL', '')
        self.device_serial = hackrf_serial_env if hackrf_serial_env else config.get('hackrf_serial', None)

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
        # Debug: Check module path
        module_path = os.getenv('SOAPY_SDR_MODULE_PATH', 'not set')
        logger.info(f"SOAPY_SDR_MODULE_PATH: {module_path}")
        
        # Debug: Check if module files exist
        import glob
        module_dirs = module_path.split(':') if module_path != 'not set' else []
        for mod_dir in module_dirs:
            if os.path.exists(mod_dir):
                modules = glob.glob(f"{mod_dir}/*.so")
                logger.info(f"Found {len(modules)} module(s) in {mod_dir}: {[os.path.basename(m) for m in modules]}")
            else:
                logger.warning(f"Module directory does not exist: {mod_dir}")
        
        # Debug: Check if USB device is accessible
        usb_devices = glob.glob('/dev/bus/usb/*/*')
        logger.info(f"Found {len(usb_devices)} USB device(s) in /dev/bus/usb/")
        
        # List all available devices for debugging
        try:
            all_devs = SoapySDR.Device.enumerate()
            logger.info(f"Found {len(all_devs)} total SoapySDR device(s):")
            for i, dev in enumerate(all_devs):
                logger.info(f"  Device {i}: {dev}")
        except Exception as e:
            logger.warning(f"Could not enumerate devices: {e}")
        
        # Try to find HackRF device - first try without driver specification
        # Sometimes SoapySDR needs to auto-detect
        try:
            # Try auto-detection first
            logger.info("Attempting to auto-detect HackRF device...")
            all_devs = SoapySDR.Device.enumerate()
            hackrf_devs = [d for d in all_devs if 'hackrf' in str(d).lower() or 'driver=hackrf' in str(d)]
            if hackrf_devs:
                logger.info(f"Found HackRF device via enumeration: {hackrf_devs[0]}")
                self.sdr = SoapySDR.Device(hackrf_devs[0])
            else:
                # Fall back to explicit driver specification
                logger.info("No HackRF found in enumeration, trying explicit driver...")
                args = {'driver': 'hackrf'}
                if self.device_serial:
                    args['serial'] = self.device_serial
                self.sdr = SoapySDR.Device(args)
        except RuntimeError as e:
            logger.error(f"Failed to create HackRF device: {e}")
            logger.error("This usually means:")
            logger.error("  1. HackRF USB device not accessible (check USB passthrough)")
            logger.error("  2. soapysdr-module-hackrf not found (check module path)")
            logger.error("  3. USB permissions issue")
            logger.error("  4. Module loaded but can't access USB device")
            raise

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

    def set_frequency(self, frequency_hz):
        """Change frequency dynamically"""
        try:
            self.frequency = int(frequency_hz)
            # Deactivate stream before changing frequency (some SDRs require this)
            if hasattr(self, 'rx_stream') and self.rx_stream is not None:
                logger.debug("Deactivating stream for frequency change...")
                self.sdr.deactivateStream(self.rx_stream)
            
            # Change frequency
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.frequency)
            logger.info(f"Frequency changed to {self.frequency} Hz")
            
            # Reactivate stream after frequency change
            if hasattr(self, 'rx_stream') and self.rx_stream is not None:
                logger.debug("Reactivating stream after frequency change...")
                self.sdr.activateStream(self.rx_stream)
                # Small delay to let stream stabilize after reactivation
                time.sleep(0.1)
            
            return True
        except Exception as e:
            logger.error(f"Failed to change frequency to {frequency_hz} Hz: {e}", exc_info=True)
            # Try to reactivate stream even if frequency change failed
            if hasattr(self, 'rx_stream') and self.rx_stream is not None:
                try:
                    self.sdr.activateStream(self.rx_stream)
                except:
                    pass
            return False

    def set_mode(self, mode):
        """Change demodulation mode dynamically"""
        try:
            self.mode = mode.upper()

            # Deactivate stream before changing mode to avoid artifacts
            if hasattr(self, 'rx_stream') and self.rx_stream is not None:
                logger.debug("Deactivating stream for mode change...")
                self.sdr.deactivateStream(self.rx_stream)

            # Change demodulator
            self.demodulator = {
                'USB': demodulate_usb,
                'LSB': demodulate_lsb,
                'AM': demodulate_am,
                'FM': demodulate_fm,
            }.get(self.mode.upper(), demodulate_usb)
            logger.info(f"Mode changed to {self.mode}")

            # Reactivate stream after mode change
            if hasattr(self, 'rx_stream') and self.rx_stream is not None:
                logger.debug("Reactivating stream after mode change...")
                self.sdr.activateStream(self.rx_stream)
                # Small delay to let stream stabilize
                time.sleep(0.1)

            return True
        except Exception as e:
            logger.error(f"Failed to change mode to {mode}: {e}", exc_info=True)
            # Try to reactivate stream even if mode change failed
            if hasattr(self, 'rx_stream') and self.rx_stream is not None:
                try:
                    self.sdr.activateStream(self.rx_stream)
                except Exception as e2:
                    logger.error(f"Failed to reactivate stream after mode change error: {e2}")
            return False

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
            if retries == 0 and samples_read == 0:
                logger.info(f"Attempting to read {remaining} samples from HackRF (timeout: 1s)")
            
            # Use threading to enforce timeout since readStream may not respect timeoutUs
            import threading
            read_result = [None]
            read_exception = [None]
            
            def do_read():
                try:
                    read_result[0] = self.sdr.readStream(
                        self.rx_stream,
                        [iq_buffer[samples_read:]],
                        remaining,
                        timeoutUs=1000000  # 1 second in microseconds
                    )
                except Exception as e:
                    read_exception[0] = e
            
            read_thread = threading.Thread(target=do_read, daemon=True)
            read_thread.start()
            read_thread.join(timeout=2.0)  # 2 second hard timeout
            
            if read_thread.is_alive():
                logger.error(f"readStream timed out after 2 seconds (retry {retries + 1}/{max_retries})")
                retries += 1
                time.sleep(0.1)
                continue
            
            if read_exception[0]:
                logger.error(f"Exception during readStream: {read_exception[0]}", exc_info=True)
                retries += 1
                time.sleep(0.01)
                continue
            
            sr = read_result[0]
            if sr is None:
                logger.warning(f"readStream returned None (retry {retries + 1}/{max_retries})")
                retries += 1
                time.sleep(0.01)
                continue
            
            logger.debug(f"readStream returned: ret={sr.ret}, flags={sr.flags}")

            if sr.ret > 0:
                samples_read += sr.ret
                retries = 0
            else:
                retries += 1
                if retries <= 3 or retries % 10 == 0:  # Log first 3 and every 10th retry
                    logger.warning(f"HackRF read returned {sr.ret}, retry {retries}/{max_retries}")
                time.sleep(0.01)

        if samples_read < self.rf_samples_per_chunk:
            logger.warning(f"Short read: got {samples_read}/{self.rf_samples_per_chunk} samples")
            if samples_read == 0:
                logger.error("No samples read from HackRF - device may not be streaming. Check USB connection.")
                # Return a silent chunk to keep the service running
                pcm = np.zeros(self.audio_rate * self.chunk_duration // 1000, dtype=np.int16)
                return AudioChunk(
                    timestamp=time.time(),
                    frequency_hz=self.frequency,
                    mode=self.mode,
                    sample_rate=self.audio_rate,
                    duration_ms=self.chunk_duration,
                    data=pcm.tobytes()
                )

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
        self.control_group = "audio-capture-service"
        self.control_consumer = "main"

    def publish_chunk(self, chunk: AudioChunk):
        """Publish audio chunk to Redis stream"""
        try:
            msg = RedisMessage.encode(chunk)
            self.redis.xadd(STREAM_AUDIO, msg, maxlen=1000)  # Keep last 1000 chunks
            logger.debug(f"Published chunk: {chunk.timestamp}, {len(chunk.data)} bytes")
        except Exception as e:
            logger.error(f"Failed to publish chunk: {e}")

    def check_control_commands(self):
        """Check for control commands from Redis stream"""
        try:
            logger.info("check_control_commands: starting...")
            import threading
            
            # First, process any pending messages for this consumer
            try:
                logger.info(f"Checking for pending messages in group {self.control_group}, consumer {self.control_consumer}")
                pending_info = self.redis.xpending_range(
                    STREAM_CONTROL,
                    self.control_group,
                    min='-',
                    max='+',
                    count=10,
                    consumername=self.control_consumer
                )
                logger.info(f"xpending_range returned {len(pending_info) if pending_info else 0} pending messages")
                if pending_info:
                    msg_ids = [msg['message_id'] for msg in pending_info]
                    logger.info(f"Found {len(msg_ids)} pending messages, claiming them...")
                    if msg_ids:
                        claimed = self.redis.xclaim(
                            STREAM_CONTROL,
                            self.control_group,
                            self.control_consumer,
                            min_idle_time=0,
                            message_ids=msg_ids
                        )
                        logger.info(f"xclaim returned {len(claimed) if claimed else 0} claimed messages")
                        if claimed:
                            logger.info(f"Processing {len(claimed)} claimed messages...")
                            # xclaim returns list of (msg_id, {data}) tuples
                            for msg_id, msg_data in claimed:
                                self._process_control_message(msg_id, msg_data)
                else:
                    logger.debug("No pending messages found")
            except Exception as e_pending:
                logger.error(f"Error checking pending messages: {e_pending}", exc_info=True)
            
            # Now read new messages (non-blocking with timeout)
            messages_result = [None]
            read_exception = [None]
            
            def do_read():
                try:
                    messages_result[0] = self.redis.xreadgroup(
                        self.control_group,
                        self.control_consumer,
                        {STREAM_CONTROL: '>'},  # Read new messages
                        count=10,
                        block=0  # Non-blocking
                    )
                except Exception as e:
                    read_exception[0] = e
            
            read_thread = threading.Thread(target=do_read, daemon=True)
            read_thread.start()
            read_thread.join(timeout=0.3)  # 300ms timeout
            
            if read_thread.is_alive():
                logger.debug("xreadgroup timed out after 300ms, will try again next iteration")
                return
            
            if read_exception[0]:
                # Handle exceptions
                if isinstance(read_exception[0], redis.exceptions.ResponseError) and 'NOGROUP' in str(read_exception[0]):
                    logger.warning(f"Consumer group missing, recreating: {read_exception[0]}")
                    try:
                        self.redis.xgroup_create(STREAM_CONTROL, self.control_group, id='0', mkstream=True)
                        logger.info("Consumer group recreated")
                        return
                    except Exception as e2:
                        logger.error(f"Failed to recreate consumer group: {e2}")
                        return
                else:
                    logger.debug(f"xreadgroup exception: {read_exception[0]}")
                    return
            
            messages = messages_result[0] if messages_result[0] is not None else []
            logger.debug(f"xreadgroup got {len(messages)} message(s)")
            
            if not messages:
                return
            
            # Process new messages from xreadgroup
            for stream, msgs in messages:
                for msg_id, msg_data in msgs:
                    self._process_control_message(msg_id, msg_data)
                    
        except Exception as e:
            logger.warning(f"Unexpected error in check_control_commands: {e}")
    
    def _process_control_message(self, msg_id, msg_data):
        """Process a single control message and acknowledge it"""
        try:
            # Decode command
            command = {}
            for k, v in msg_data.items():
                key = k.decode('utf-8') if isinstance(k, bytes) else k
                val = v.decode('utf-8') if isinstance(v, bytes) else v
                command[key] = val
            
            # Handle frequency change
            if command.get('command') == 'set_frequency':
                frequency = int(command.get('frequency_hz', 0))
                if frequency > 0:
                    if hasattr(self.source, 'set_frequency'):
                        success = self.source.set_frequency(frequency)
                        logger.info(f"Control command: set_frequency to {frequency} Hz - {'success' if success else 'failed'}")
                    else:
                        logger.warning(f"Source does not support set_frequency")
            
            # Handle mode change
            elif command.get('command') == 'set_mode':
                mode = command.get('mode', '')
                if mode:
                    if hasattr(self.source, 'set_mode'):
                        success = self.source.set_mode(mode)
                        logger.info(f"Control command: set_mode to {mode} - {'success' if success else 'failed'}")
                    else:
                        logger.warning(f"Source does not support set_mode")
            
            # Acknowledge message
            self.redis.xack(STREAM_CONTROL, self.control_group, msg_id)
            logger.debug(f"Acknowledged message {msg_id}")
            
        except Exception as e:
            logger.error(f"Error processing control message {msg_id}: {e}", exc_info=True)
                    
        except Exception as e:
            # Ignore errors (stream might not exist yet, or no messages)
            pass

    def run(self):
        """Main capture loop"""
        self.running = True
        logger.info("Starting audio capture service...")
        
        # Create consumer group for control stream
        try:
            logger.info("Creating Redis consumer group...")
            self.redis.xgroup_create(STREAM_CONTROL, self.control_group, id='0', mkstream=True)
            logger.info("Redis consumer group created")
        except redis.exceptions.ResponseError as e:
            # Group already exists, that's fine
            logger.info(f"Redis consumer group already exists: {e}")
            pass

        chunk_count = 0
        start_time = time.time()
        last_control_check = 0

        try:
            logger.info("Entering main capture loop...")
            import sys
            sys.stdout.flush()
            loop_iteration = 0
            while self.running:
                loop_iteration += 1
                if loop_iteration == 1:
                    logger.info(f"First loop iteration, self.running={self.running}")
                logger.debug(f"Loop iteration {loop_iteration} starting...")
                # Check for control commands every second
                if time.time() - last_control_check > 1.0:
                    logger.info("Checking for control commands...")
                    self.check_control_commands()
                    logger.info("check_control_commands() returned")
                    last_control_check = time.time()
                
                try:
                    logger.info("About to call read_chunk()...")
                    import sys
                    sys.stdout.flush()
                    chunk = self.source.read_chunk()
                    logger.info("read_chunk() completed, publishing...")
                    self.publish_chunk(chunk)

                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = chunk_count / elapsed
                        logger.info(f"Captured {chunk_count} chunks ({rate:.1f}/sec)")

                    # Sleep to maintain real-time rate
                    time.sleep(chunk.duration_ms / 1000 * 0.9)  # Slight underrun
                except Exception as e:
                    logger.error(f"Error in capture loop: {e}", exc_info=True)
                    time.sleep(1)  # Wait before retrying

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            self.running = False
        except Exception as e:
            logger.error(f"Fatal error in capture service: {e}", exc_info=True)
            raise

    def stop(self):
        self.running = False


if __name__ == '__main__':
    service = AudioCaptureService()
    service.run()
