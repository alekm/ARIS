#!/usr/bin/env python3
"""
Audio Capture Service
Captures audio from KiwiSDR or mock source and streams to Redis
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
            data=audio_data,
            s_meter=0.0,
            signal_strength_db=-150.0,
            squelch_open=True,
            rssi=None,
            low_cut=300,
            high_cut=2700
        )


class KiwiSDRAudioSource:
    """
    KiwiSDR audio source using WebSocket protocol.
    Connects to KiwiSDR and receives demodulated audio.
    Handles reconnection on disconnect (for daily reboots/updates).
    """

    def __init__(self, config):
        import asyncio
        import websockets
        import struct
        import threading
        import queue
        import re
        import socket

        # Store imports for later use
        self._asyncio = asyncio
        self._websockets = websockets
        self._struct = struct
        self._threading = threading
        self._queue = queue
        self._re = re
        self._socket = socket

        # Read from environment variables first, then fall back to config file
        host_raw = os.getenv('KIWI_HOST', config.get('kiwi_host', ''))
        port_raw = os.getenv('KIWI_PORT', config.get('kiwi_port', 8073))
        password_raw = os.getenv('KIWI_PASSWORD', config.get('kiwi_password', ''))
        frequency_raw = os.getenv('FREQUENCY_HZ', config.get('frequency_hz', 7200000))
        mode_raw = os.getenv('DEMOD_MODE', config.get('demod_mode', config.get('mode', 'USB')))
        
        # Validate and sanitize inputs
        self.host = self._validate_host(host_raw)
        self.port = self._validate_port(port_raw)
        self.password = password_raw  # Don't sanitize - send as-is like kiwirecorder
        self.frequency = self._validate_frequency(frequency_raw)
        self.mode = self._validate_mode(mode_raw)
        
        self.sample_rate = config.get('sample_rate', 12000)  # KiwiSDR native is 12kHz
        self.chunk_duration = config.get('chunk_duration_ms', 1000)
        self.target_sample_rate = 16000  # Target for Whisper
        
        # Feature flags
        self.use_wss = os.getenv('KIWI_USE_WSS', 'false').lower() == 'true'
        self.debug_kiwi = os.getenv('DEBUG_KIWI', 'false').lower() == 'true'
        self.squelch_threshold = float(os.getenv('SQUELCH_THRESHOLD', config.get('squelch_threshold', 0.0)))
        self.noise_blanker = os.getenv('NOISE_BLANKER', config.get('noise_blanker', 'false')).lower() == 'true'
        
        # Connection metrics
        self.reconnect_count = 0
        self.max_reconnect_attempts = int(os.getenv('KIWI_MAX_RECONNECT', 0))  # 0 = unlimited
        self.reconnect_base_delay = 1.0   # Start with 1 second (exponential: 1, 2, 4, 8, 16, 32, 60)
        self.reconnect_max_delay = 60.0   # Cap at 60 seconds
        self.reconnect_jitter = 0.2       # ±20% random jitter to prevent thundering herd
        
        # AGC Settings
        self.agc_enabled = os.getenv('KIWI_AGC', str(config.get('kiwi_agc', True))).lower() == 'true'
        self.agc_threshold = int(os.getenv('KIWI_AGC_THRESH', config.get('kiwi_agc_thresh', -100)))
        self.agc_slope = int(os.getenv('KIWI_AGC_SLOPE', config.get('kiwi_agc_slope', 6)))
        self.agc_decay = int(os.getenv('KIWI_AGC_DECAY', config.get('kiwi_agc_decay', 1000)))
        self.agc_hang = os.getenv('KIWI_AGC_HANG', str(config.get('kiwi_agc_hang', False))).lower() == 'true'
        self.manual_gain = int(os.getenv('KIWI_MAN_GAIN', config.get('kiwi_man_gain', 48))) # Used if AGC is off
        self.audio_endian = os.getenv('KIWI_AUDIO_ENDIAN', config.get('kiwi_audio_endian', 'big')).lower() # big or little (KiwiSDR is usually big endian)
        self.user = os.getenv('KIWI_USER', config.get('kiwi_user', 'ARIS'))
        
        # Filter Settings
        # Default bandwidths based on mode
        default_low = 300
        default_high = 2700
        
        if self.mode == 'LSB':
            default_low = -2700
            default_high = -300
        elif self.mode == 'AM':
            default_low = -5000
            default_high = 5000
        elif self.mode == 'CW':
            default_low = 400
            default_high = 800
            
        self.low_cut = int(os.getenv('KIWI_LOW_CUT', config.get('kiwi_low_cut', default_low)))
        self.high_cut = int(os.getenv('KIWI_HIGH_CUT', config.get('kiwi_high_cut', default_high)))

        # Audio buffer
        self.audio_queue = queue.Queue(maxsize=100)
        self.audio_buffer = b''
        self.samples_per_chunk = int(self.target_sample_rate * self.chunk_duration / 1000)

        # Connection state (protected by _state_lock)
        self.connected = False
        self.ws = None
        self.ws_thread = None
        self.loop = None  # Store event loop reference
        self.running = True
        self.reconnect_delay = self.reconnect_base_delay
        self.last_audio_time = time.time()
        self.last_error = None
        self.packet_loss_count = 0
        self.last_sequence = None
        
        # Thread synchronization lock for shared state
        self._state_lock = self._threading.Lock()

        logger.info(f"KiwiSDR source: {self.host}:{self.port}, {self.frequency} Hz, {self.mode}")

        if self.debug_kiwi:
            logger.setLevel(logging.DEBUG)
            logger.debug("KiwiSDR debug logging enabled")

        if not self.host:
            raise ValueError("KIWI_HOST not configured")
        
        # Start WebSocket connection in background thread
        self._start_ws_thread()
    
    def _validate_host(self, host):
        """Validate host is IP address or hostname (prevent SSRF)"""
        if not host:
            raise ValueError("Host cannot be empty")
        
        # Check if it's an IP address
        try:
            self._socket.inet_aton(host)
            return host
        except self._socket.error:
            pass
        
        # Check if it's a valid hostname (alphanumeric, dots, hyphens)
        if self._re.match(r'^[a-zA-Z0-9.-]+$', host):
            return host
        
        raise ValueError(f"Invalid host format: {host}")
    
    def _validate_port(self, port):
        """Validate port is in valid range"""
        try:
            port_int = int(port)
            if 1 <= port_int <= 65535:
                return port_int
            raise ValueError(f"Port must be between 1 and 65535, got {port_int}")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid port: {port}")
    
    def _sanitize_password(self, password):
        """Sanitize password (remove dangerous characters)"""
        if not password:
            return ""
        # Remove any characters that could be used for command injection
        # Allow alphanumeric and common safe characters
        sanitized = self._re.sub(r'[^a-zA-Z0-9_\-=]', '', password)
        return sanitized
    
    def _validate_frequency(self, frequency):
        """Validate frequency is within KiwiSDR limits (0-30 MHz)"""
        try:
            freq_int = int(frequency)
            # KiwiSDR typically supports 0-30 MHz, but allow up to 65 MHz for some models
            if 0 <= freq_int <= 65000000:
                return freq_int
            raise ValueError(f"Frequency must be between 0 and 65 MHz, got {freq_int} Hz")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid frequency: {frequency}")
    
    def _validate_mode(self, mode):
        """Validate and sanitize mode (whitelist)"""
        valid_modes = ['USB', 'LSB', 'AM', 'FM', 'CW']
        mode_upper = str(mode).upper()
        if mode_upper in valid_modes:
            return mode_upper
        raise ValueError(f"Mode must be one of {valid_modes}, got {mode}")
    
    def _sanitize_set_command(self, value):
        """Sanitize value for SET command (prevent command injection)"""
        # Remove any characters that could break SET command syntax
        sanitized = str(value).replace(' ', '').replace(';', '').replace('\n', '').replace('\r', '')
        return sanitized

    def _start_ws_thread(self):
        """Start the WebSocket connection thread"""
        logger.info("Starting KiwiSDR WebSocket thread...")
        self.ws_thread = self._threading.Thread(target=self._ws_loop, daemon=True)
        self.ws_thread.start()
        logger.info("KiwiSDR WebSocket thread started")
    
    def _get_state(self):
        """Thread-safe getter for connection state"""
        with self._state_lock:
            return {
                'ws': self.ws,
                'connected': self.connected,
                'loop': self.loop
            }
    
    def _set_state(self, ws=None, connected=None, loop=None):
        """Thread-safe setter for connection state"""
        with self._state_lock:
            if ws is not None:
                self.ws = ws
            if connected is not None:
                self.connected = connected
            if loop is not None:
                self.loop = loop

    def _ws_loop(self):
        """Main WebSocket loop with reconnection handling and exponential backoff"""
        import random
        
        logger.info("WebSocket loop thread started")
        # Create persistent event loop for this thread
        loop = self._asyncio.new_event_loop()
        self._asyncio.set_event_loop(loop)
        
        try:
            while self.running:
                # Check max reconnect attempts
                if self.max_reconnect_attempts > 0 and self.reconnect_count >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Stopping.")
                    self.running = False
                    break
                
                try:
                    logger.info(f"Connecting to KiwiSDR at {self.host}:{self.port}... (attempt {self.reconnect_count + 1})")
                    # Store loop reference before connection
                    self._set_state(loop=loop)
                    # Run connection in persistent loop
                    logger.info("Calling _connect_and_receive()...")
                    loop.run_until_complete(self._connect_and_receive())
                    logger.info("_connect_and_receive() returned")
                    # Reset reconnect delay on successful connection
                    self.reconnect_delay = self.reconnect_base_delay
                    self.reconnect_count = 0
                except Exception as e:
                    logger.error(f"KiwiSDR connection error: {e}", exc_info=True)
                    # Clear loop reference on error
                    self._set_state(loop=None, connected=False, ws=None)
                    self.reconnect_count += 1

                if self.running:
                    # Exponential backoff with jitter
                    delay = min(self.reconnect_delay, self.reconnect_max_delay)
                    jitter = random.uniform(-self.reconnect_jitter * delay, self.reconnect_jitter * delay)
                    actual_delay = max(0.1, delay + jitter)
                    
                    logger.info(f"Reconnecting in {actual_delay:.1f} seconds... (delay: {delay:.1f}s, jitter: {jitter:.2f}s)")
                    time.sleep(actual_delay)
                    
                    # Increase delay for next attempt (exponential backoff)
                    self.reconnect_delay *= 2
        finally:
            # Clean up event loop
            try:
                pending = self._asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(self._asyncio.gather(*pending, return_exceptions=True))
                loop.close()
            except Exception as e:
                logger.error(f"Error closing event loop: {e}")
            self._set_state(loop=None)

    async def _connect_and_receive(self):
        """Connect to KiwiSDR and receive audio data"""
        import random

        # Get current event loop (already set in _ws_loop)
        loop = self._asyncio.get_event_loop()
        # Store loop reference atomically with connection attempt
        self._set_state(loop=loop)

        # KiwiSDR WebSocket URL format
        client_id = random.randint(0, 999999)
        protocol = "wss" if self.use_wss else "ws"
        url = f"{protocol}://{self.host}:{self.port}/kiwi/{client_id}/SND"
        
        # Mask password in logs
        url_log = url.replace(self.password, "***") if self.password else url

        try:
            logger.info(f"Attempting WebSocket connection to {url_log}...")
            # Connect with keepalive to prevent timeout during dead air (KiwiSDR drops after 60s)
            async with self._websockets.connect(
                url,
                ping_interval=20,   # Send keepalive ping every 20 seconds
                ping_timeout=20,    # Expect pong within 20 seconds
                max_size=None
            ) as ws:
                # Store WebSocket reference atomically
                self._set_state(ws=ws, connected=True)
                self.last_error = None
                logger.info(f"Connected to KiwiSDR: {url_log}")

                # Store WS reference for sending AR OK response
                self._current_ws = ws

                # Send AUTH immediately (as per kiwiclient protocol)
                # Password must be sent before sample_rate is received
                auth_cmd = f"SET auth t=kiwi p={self.password}" if self.password else "SET auth t=kiwi p="
                if self.debug_kiwi:
                    logger.debug(f"Sent: {auth_cmd}")
                await ws.send(auth_cmd)

                logger.info("Sent AUTH, waiting for sample_rate...")

                # Receive audio data
                message_count = 0
                last_keepalive_time = time.time()
                logger.info("Starting message receive loop...")
                async for message in ws:
                    if not self.running:
                        logger.info("Service stopping, breaking from message loop")
                        break

                    message_count += 1
                    if message_count == 1:
                        if isinstance(message, bytes):
                            logger.info(f"Received first message from KiwiSDR (type: bytes, len: {len(message)}, first 20 bytes: {message[:20].hex()})")
                        else:
                            logger.info(f"Received first message from KiwiSDR (type: {type(message).__name__}, content: {str(message)[:100]})")
                    elif message_count % 100 == 0:
                        logger.debug(f"Received {message_count} messages from KiwiSDR")

                    # Send keepalive every 5 seconds
                    current_time = time.time()
                    if current_time - last_keepalive_time >= 5.0:
                        await ws.send("SET keepalive")
                        last_keepalive_time = current_time
                        if self.debug_kiwi and message_count % 100 == 0:
                            logger.debug("Sent periodic keepalive")

                    if isinstance(message, bytes):
                        # All KiwiSDR messages come as bytes (even status messages)
                        await self._process_audio_message(message)
                    else:
                        # Text message (shouldn't happen, but handle it)
                        logger.info(f"Received text message from KiwiSDR: {message}")
                        await self._process_text_message(message)
                
                logger.warning(f"Message loop exited after {message_count} messages (connection may have closed)")

        except self._websockets.exceptions.ConnectionClosed as e:
            self.last_error = f"Connection closed: {e}"
            logger.warning(f"KiwiSDR connection closed: {e}")
        except self._websockets.exceptions.InvalidStatusCode as e:
            self.last_error = f"Invalid status code: {e}"
            logger.error(f"KiwiSDR connection failed (invalid status): {e}")
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"KiwiSDR WebSocket error: {e}")
        finally:
            # Clear connection state atomically
            self._set_state(connected=False, ws=None)
            # Note: Keep loop reference until thread exits

    async def _send_setup_commands(self, ws):
        """Send KiwiSDR setup commands"""
        # Send client identification (required by some KiwiSDRs)
        # Note: AUTH is sent in _connect_and_receive before this

        # Packet Loss / Squelch defaults
        await ws.send("SET squelch=0 max=0")
        if self.debug_kiwi:
            logger.debug("Sent: SET squelch=0 max=0")

        # Start Generator (must be before setting freq/mode in some versions?)
        # kiwiclient sends this early
        await ws.send("SET gen=0 mix=-1")
        if self.debug_kiwi:
            logger.debug("Sent: SET gen=0 mix=-1")
            
        if self.debug_kiwi:
            logger.debug(f"Sent: SET ident_user={self.user}")
        await ws.send(f"SET ident_user={self.user}")

        # NOTE: Do NOT send "SET AR OK" here - it must be sent as a response to MSG audio_rate

        # Set zoom level (0-14, where higher = more zoomed in)
        # Zoom 3 gives a good balance for voice signals (~10 kHz passband)
        await ws.send("SET zoom=3 start=0")

        # Set frequency (in kHz for KiwiSDR)
        freq_khz = self.frequency / 1000.0
        FREQ_CMD = f"SET mod={self.mode.lower()} low_cut={self.low_cut} high_cut={self.high_cut} freq={freq_khz:.3f}"
        if self.debug_kiwi:
            logger.debug(f"Sent: {FREQ_CMD}")
        await ws.send(FREQ_CMD)

        # Configure AGC
        # manGain is used only when AGC is off (agc=0)
        agc_val = 1 if self.agc_enabled else 0
        hang_val = 1 if self.agc_hang else 0

        AGC_CMD = f"SET agc={agc_val} hang={hang_val} thresh={self.agc_threshold} slope={self.agc_slope} decay={self.agc_decay} manGain={self.manual_gain}"
        if self.debug_kiwi:
            logger.debug(f"Sent: {AGC_CMD}")
        await ws.send(AGC_CMD)

        # Enable noise blanker if configured
        if self.noise_blanker:
            await ws.send("SET nb=1")
            if self.debug_kiwi:
                logger.debug("Sent: SET nb=1")

        # Disable audio compression (we don't have ADPCM decoder)
        await ws.send("SET compression=0")
        if self.debug_kiwi:
            logger.debug("Sent: SET compression=0")

        # Start audio stream - already sent above
        # await ws.send("SET gen=0 mix=-1")

        # Send initial keepalive
        await ws.send("SET keepalive")
        if self.debug_kiwi:
            logger.debug("Sent: SET keepalive")

        logger.info(f"KiwiSDR setup complete: {freq_khz:.3f} kHz, {self.mode}, waiting for server audio_rate message...")

    async def _process_audio_message(self, data):
        """Process binary audio message from KiwiSDR"""
        # KiwiSDR sends two types of binary messages:
        # 1. "MSG " - status/configuration messages (should be decoded as UTF-8 text)
        # 2. "SND" - audio data packets
        
        # Check for MSG header (status messages)
        if len(data) >= 4 and data[0:4] == b'MSG ':
            # This is a status message, decode as text and process
            try:
                text_msg = data.decode('utf-8', errors='ignore')
                await self._process_text_message(text_msg)
            except Exception as e:
                if self.debug_kiwi:
                    logger.debug(f"Error decoding MSG message: {e}")
            return

        if len(data) < 10:
            if self.debug_kiwi:
                logger.debug(f"Received short message from KiwiSDR (len={len(data)})")
            return

        # Check for SND header (audio data)
        if data[0:3] == b'SND':
            # Extract header fields
            # Header: SND(3) + flags(1) + seq(4) + rssi(2) = 10 bytes
            flags = data[3]
            sequence = self._struct.unpack('<I', data[4:8])[0]  # Little-endian uint32
            rssi_raw = self._struct.unpack('<H', data[8:10])[0]  # Little-endian uint16
            
            # Track packet loss
            if self.last_sequence is not None:
                expected_seq = (self.last_sequence + 1) % (2**32)
                if sequence != expected_seq:
                    lost = (sequence - expected_seq) % (2**32)
                    if lost > 1:
                        self.packet_loss_count += lost - 1
                        if self.debug_kiwi:
                            logger.debug(f"Packet loss detected: expected {expected_seq}, got {sequence}, lost {lost-1} packets")
            self.last_sequence = sequence
            
            # Extract RSSI (KiwiSDR RSSI is in dBm, typically -150 to 0)
            # Convert to S-meter reading (S1 = -121 dBm, S9 = -73 dBm, S9+20 = -53 dBm)
            rssi_db = -(rssi_raw & 0x7FFF)  # Mask sign bit, negate
            self.last_rssi = rssi_db
            
            # Check flags for compression (SND_FLAG_COMPRESSED = 0x10)
            is_compressed = (flags & 0x10) != 0
            is_stereo = (flags & 0x08) != 0

            # Extract audio data
            audio_data = data[10:]

            if len(audio_data) > 0:
                self.last_audio_time = time.time()

                # Log first SND packet received
                if not hasattr(self, '_first_snd_logged'):
                    logger.info(f"Received first SND audio packet (seq={sequence}, len={len(audio_data)}, rssi={rssi_db:.1f} dBm, flags=0x{flags:02x}, compressed={is_compressed}, stereo={is_stereo})")
                    self._first_snd_logged = True

                # CRITICAL: If audio is compressed, we can't use it (we don't have ADPCM decoder)
                if is_compressed:
                    if self.debug_kiwi and not hasattr(self, '_compression_warning_logged'):
                        logger.warning("KiwiSDR sending compressed audio despite compression=0! Audio will be garbled.")
                        self._compression_warning_logged = True
                    return  # Skip compressed packets

                # Add to buffer
                try:
                    self.audio_queue.put_nowait(audio_data)
                except self._queue.Full:
                    # Drop oldest if queue is full
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(audio_data)
                    except:
                        pass
        else:
            # Not an SND or MSG message - log it for debugging
            if self.debug_kiwi:
                logger.debug(f"Received unknown message from KiwiSDR (len={len(data)}, first 20 bytes: {data[:20].hex()})")

    async def _process_text_message(self, message):
        """Process text message from KiwiSDR"""
        if self.debug_kiwi:
            logger.debug(f"KiwiSDR message: {message[:100] if len(message) > 100 else message}")

        # Parse MSG messages (format: "MSG key=value key2=value2 ...")
        if message.startswith("MSG "):
            parts = message[4:].split()
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    
                    if key == 'sample_rate':
                        logger.info("Received sample_rate. Triggering setup sequence...")
                        if hasattr(self, '_current_ws') and self._current_ws:
                            await self._send_setup_commands(self._current_ws)
                            
                    elif key == 'audio_rate':
                        # Server sent audio rate - respond with AR OK
                        audio_rate = int(value.split('.')[0])  # May be "11998.909149" or "12000"
                        if hasattr(self, '_current_ws') and self._current_ws:
                            try:
                                await self._current_ws.send(f"SET AR OK in={audio_rate} out=44100")
                                if self.debug_kiwi:
                                    logger.debug(f"Sent: SET AR OK in={audio_rate} out=44100 (response to audio_rate)")
                            except Exception as e:
                                logger.error(f"Failed to send AR OK response: {e}")

        # Handle specific messages
        msg_lower = message.lower()
        if "too_busy" in msg_lower:
            self.last_error = "KiwiSDR is too busy"
            logger.warning("KiwiSDR is too busy, will retry...")
            # Trigger reconnection with backoff
            self._set_state(connected=False, ws=None)
        elif "badp=1" in msg_lower:
            # badp=1 means password was rejected
            self.last_error = "KiwiSDR password rejected"
            logger.error("KiwiSDR password rejected")
            # Don't retry on auth failure
            self.running = False
        elif "badp=0" in msg_lower:
            # badp=0 means no password required - this is fine, not an error
            if self.debug_kiwi:
                logger.debug("KiwiSDR: no password required (badp=0)")
        elif "error" in msg_lower or "fail" in msg_lower:
            self.last_error = message
            logger.warning(f"KiwiSDR error message: {message}")

    def set_frequency(self, frequency_hz):
        """Change frequency dynamically with retry logic"""
        self.frequency = int(frequency_hz)
        
        # Retry logic with exponential backoff
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            # Get current state safely
            state = self._get_state()
            ws = state['ws']
            connected = state['connected']
            loop = state['loop']
            
            if not connected or not ws or not loop:
                if attempt < max_retries - 1:
                    logger.debug(f"KiwiSDR not ready, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.warning(f"KiwiSDR not connected after {max_retries} attempts (ws={ws is not None}, connected={connected}, loop={loop is not None})")
                    return False
            
            # Validate loop is still running
            if not loop.is_running():
                logger.warning("Event loop is not running, cannot change frequency")
                self._set_state(connected=False, ws=None, loop=None)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return False
            
            try:
                freq_khz = self.frequency / 1000.0
                
                # Verify WebSocket is still open (atomic check)
                with self._state_lock:
                    # Get fresh reference while holding lock
                    current_ws = self.ws
                    current_loop = self.loop
                
                if not current_ws or not current_loop:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return False
                
                # Send via the running event loop (no sanitization - send as-is like kiwirecorder)
                cmd = f"SET mod={self.mode.lower()} low_cut={self.low_cut} high_cut={self.high_cut} freq={freq_khz:.3f}"
                self._asyncio.run_coroutine_threadsafe(
                    current_ws.send(cmd),
                    current_loop
                )
                if self.debug_kiwi:
                    logger.debug(f"Sent: {cmd}")
                logger.info(f"KiwiSDR frequency changed to {freq_khz:.3f} kHz")
                return True
            except RuntimeError as e:
                # Loop is closed or invalid
                error_msg = str(e)
                if "is closed" in error_msg or "no current event loop" in error_msg.lower():
                    logger.error(f"Event loop error when changing frequency: {e}")
                    self._set_state(connected=False, ws=None, loop=None)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                else:
                    logger.error(f"Runtime error when changing frequency: {e}")
                    return False
            except Exception as e:
                logger.error(f"Failed to change KiwiSDR frequency: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return False
        
        return False

    def set_mode(self, mode):
        """Change demodulation mode dynamically with retry logic"""
        self.mode = mode.upper()
        
        # Update filter settings based on new mode
        if self.mode == 'LSB':
            self.low_cut = -2700
            self.high_cut = -300
        elif self.mode == 'AM':
            self.low_cut = -5000
            self.high_cut = 5000
        elif self.mode == 'CW':
            self.low_cut = 400
            self.high_cut = 800
        else: # USB and others
            self.low_cut = 300
            self.high_cut = 2700
            
        logger.info(f"Mode changed to {self.mode}, updating filters to {self.low_cut}/{self.high_cut}")
        
        # Retry logic with exponential backoff
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            # Get current state safely
            state = self._get_state()
            ws = state['ws']
            connected = state['connected']
            loop = state['loop']
            
            if not connected or not ws or not loop:
                if attempt < max_retries - 1:
                    logger.debug(f"KiwiSDR not ready, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.warning(f"KiwiSDR not connected after {max_retries} attempts (ws={ws is not None}, connected={connected}, loop={loop is not None})")
                    return False
            
            # Validate loop is still running
            if not loop.is_running():
                logger.warning("Event loop is not running, cannot change mode")
                self._set_state(connected=False, ws=None, loop=None)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return False
            
            try:
                freq_khz = self.frequency / 1000.0
                
                # Verify WebSocket is still open (atomic check)
                with self._state_lock:
                    # if ws.closed:
                    #    logger.warning("WebSocket is closed, cannot change mode")
                    #    self._set_state(connected=False, ws=None)
                    #    if attempt < max_retries - 1:
                    #        time.sleep(retry_delay)
                    #        retry_delay *= 2
                    #        continue
                    #    return False
                    # Get fresh reference while holding lock
                    current_ws = self.ws
                    current_loop = self.loop
                
                if not current_ws or not current_loop:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return False
                
                # Send mode change (no sanitization - send as-is like kiwirecorder)
                cmd = f"SET mod={self.mode.lower()} low_cut={self.low_cut} high_cut={self.high_cut} freq={freq_khz:.3f}"
                self._asyncio.run_coroutine_threadsafe(
                    current_ws.send(cmd),
                    current_loop
                )
                if self.debug_kiwi:
                    logger.debug(f"Sent: {cmd}")
                logger.info(f"KiwiSDR mode changed to {self.mode}")
                return True
            except RuntimeError as e:
                # Loop is closed or invalid
                error_msg = str(e)
                if "is closed" in error_msg or "no current event loop" in error_msg.lower():
                    logger.error(f"Event loop error when changing mode: {e}")
                    self._set_state(connected=False, ws=None, loop=None)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                else:
                    logger.error(f"Runtime error when changing mode: {e}")
                    return False
            except Exception as e:
                logger.error(f"Failed to change KiwiSDR mode: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return False
        
        return False

    def read_chunk(self):
        """Return a chunk of audio from KiwiSDR"""
        # Collect audio data from queue
        collected = b''
        bytes_needed = self.samples_per_chunk * 2  # 16-bit = 2 bytes per sample

        timeout = self.chunk_duration / 1000 * 2  # Wait up to 2x chunk duration
        start = time.time()

        while len(collected) < bytes_needed and (time.time() - start) < timeout:
            try:
                data = self.audio_queue.get(timeout=0.1)
                collected += data
            except self._queue.Empty:
                # Check connection state safely
                state = self._get_state()
                if not state['connected']:
                    # Not connected - return silence
                    if self.debug_kiwi:
                        logger.debug(f"KiwiSDR not connected, collected {len(collected)}/{bytes_needed} bytes")
                    break
                # Still connected but queue empty - continue waiting
                continue

        # Convert to numpy array
        if len(collected) >= 2:
            # Log if we got enough audio data
            if not hasattr(self, '_first_chunk_logged'):
                logger.info(f"read_chunk: collected {len(collected)}/{bytes_needed} bytes from audio queue")
                self._first_chunk_logged = True
            # KiwiSDR sends 12kHz audio, we need to resample to 16kHz
            # KiwiSDR sends 12kHz audio. Data is usually Big Endian (>i2).
            # We use >i2 (Big Endian) or <i2 (Little Endian) based on config.
            dtype_str = '>i2' if self.audio_endian == 'big' else '<i2'
            audio_12k = np.frombuffer(collected[:bytes_needed], dtype=np.dtype(dtype_str))

            # Resample from 12kHz to 16kHz using high-quality polyphase filtering
            if len(audio_12k) > 0:
                # Log raw audio statistics before resampling (for debugging)
                if self.debug_kiwi:
                    if not hasattr(self, '_debug_chunk_count'):
                        self._debug_chunk_count = 0
                    self._debug_chunk_count += 1
                    if self._debug_chunk_count % 10 == 0:
                        raw_min = np.min(audio_12k)
                        raw_max = np.max(audio_12k)
                        raw_rms = np.sqrt(np.mean(audio_12k.astype(np.float32) ** 2))
                        raw_std = np.std(audio_12k.astype(np.float32))
                        logger.debug(f"Raw 12kHz audio: min={raw_min}, max={raw_max}, RMS={raw_rms:.1f}, std={raw_std:.1f}")

                # Use scipy polyphase resampler: 12kHz * (4/3) = 16kHz
                # Kaiser window with beta=5.0 provides excellent quality
                audio_16k = scipy_signal.resample_poly(audio_12k, up=4, down=3, window=('kaiser', 5.0))

                # No volume reduction with manual gain (already at proper level)
                VOLUME_REDUCTION = 1.0
                audio_16k = (audio_16k * VOLUME_REDUCTION).astype(np.int16)
            else:
                audio_16k = np.zeros(self.samples_per_chunk, dtype=np.int16)
        else:
            # No audio - return silence
            audio_16k = np.zeros(self.samples_per_chunk, dtype=np.int16)
            state = self._get_state()
            if not state['connected']:
                logger.warning("KiwiSDR not connected, returning silence")

        # Calculate signal strength for squelch
        signal_strength_db = None
        s_meter = 0.0
        squelch_open = True
        
        if len(audio_16k) > 0:
            # Calculate RMS power
            rms = np.sqrt(np.mean(audio_16k.astype(np.float32) ** 2))
            # Convert to dB (relative to full scale)
            signal_strength_db = 20 * np.log10(rms / 32767.0 + 1e-10)
            
            # Use RSSI from KiwiSDR if available, otherwise use calculated
            if hasattr(self, 'last_rssi') and self.last_rssi is not None:
                signal_strength_db = self.last_rssi
            
            # Calculate S-meter from RSSI (S1 = -121 dBm, S9 = -73 dBm, S9+20 = -53 dBm)
            if signal_strength_db >= -53:
                s_meter = 9.9  # S9+20 or higher
            elif signal_strength_db >= -73:
                s_meter = 9.0 + (signal_strength_db + 73) / 2.0  # S9 to S9+20
            elif signal_strength_db >= -121:
                s_meter = 1.0 + (signal_strength_db + 121) / 6.0  # S1 to S9
            else:
                s_meter = max(0.0, 1.0 + (signal_strength_db + 130) / 9.0)  # Below S1
            
            s_meter = min(9.9, max(0.0, s_meter))
            
            # Check squelch threshold
            if self.squelch_threshold > 0:
                squelch_open = s_meter >= self.squelch_threshold
                if not squelch_open:
                    # Return silence if squelch is closed
                    audio_16k = np.zeros(self.samples_per_chunk, dtype=np.int16)
        else:
            signal_strength_db = -150.0
            s_meter = 0.0
            squelch_open = False

        return AudioChunk(
            timestamp=time.time(),
            frequency_hz=self.frequency,
            mode=self.mode,
            sample_rate=self.target_sample_rate,
            duration_ms=self.chunk_duration,
            data=audio_16k.tobytes(),
            s_meter=s_meter,
            signal_strength_db=signal_strength_db if signal_strength_db is not None else -150.0,
            squelch_open=squelch_open,
            rssi=getattr(self, 'last_rssi', None),
            low_cut=getattr(self, 'low_cut', 300),
            high_cut=getattr(self, 'high_cut', 2700)
        )

    def close(self):
        """Clean up KiwiSDR connection"""
        self.running = False
        
        # Get current state safely
        state = self._get_state()
        ws = state['ws']
        loop = state['loop']
        
        if ws and loop:
            try:
                # Check if loop is still running
                if loop.is_running():
                    # Schedule close in the event loop
                    self._asyncio.run_coroutine_threadsafe(
                        ws.close(),
                        loop
                    )
                else:
                    # Loop is closed, just log
                    logger.debug("Event loop already closed, skipping WebSocket close")
            except RuntimeError as e:
                # Loop is closed or invalid
                logger.debug(f"Could not close WebSocket (loop invalid): {e}")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
        

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
        else:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: mock, kiwi")

        self.running = False
        self.paused = False  # Control whether to capture or pause
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
            # logger.info("check_control_commands: starting...") # Reduced log spam
            
            # First, process any pending messages for this consumer
            try:
                # Check pending less frequently or just rely on xautoclaim/xreadgroup? 
                # For simplicity, we keep the existing logic but maybe don't log every time unless there are items
                pending_info = self.redis.xpending_range(
                    STREAM_CONTROL,
                    self.control_group,
                    min='-',
                    max='+',
                    count=10,
                    consumername=self.control_consumer
                )
                
                if pending_info:
                    logger.info(f"xpending_range found {len(pending_info)} pending messages")
                    msg_ids = [msg['message_id'] for msg in pending_info]
                    if msg_ids:
                        claimed = self.redis.xclaim(
                            STREAM_CONTROL,
                            self.control_group,
                            self.control_consumer,
                            min_idle_time=0,
                            message_ids=msg_ids
                        )
                        if claimed:
                            logger.info(f"Processing {len(claimed)} claimed messages...")
                            for msg_id, msg_data in claimed:
                                self._process_control_message(msg_id, msg_data)
            except Exception as e_pending:
                logger.error(f"Error checking pending messages: {e_pending}")
            
            # Read new messages
            try:
                # block=100ms
                messages = self.redis.xreadgroup(
                    self.control_group,
                    self.control_consumer,
                    {STREAM_CONTROL: '>'},
                    count=10,
                    block=100 
                )
            except redis.exceptions.ResponseError as e:
                if 'NOGROUP' in str(e):
                    logger.warning(f"Consumer group missing, recreating: {e}")
                    try:
                        self.redis.xgroup_create(STREAM_CONTROL, self.control_group, id='0', mkstream=True)
                        logger.info("Consumer group recreated")
                        return
                    except Exception as e2:
                        logger.error(f"Failed to recreate consumer group: {e2}")
                        return
                else:
                    raise
            
            if not messages:
                return

            # Process new messages
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
            
            logger.debug(f"Processing control command: {command}")
            
            # Handle frequency change
            if command.get('command') == 'set_frequency':
                frequency = int(command.get('frequency_hz', 0))
                if frequency > 0:
                    if hasattr(self.source, 'set_frequency'):
                        success = self.source.set_frequency(frequency)
                        if success:
                            logger.info(f"Control command: set_frequency to {frequency} Hz - success")
                        else:
                            logger.warning(f"Control command: set_frequency to {frequency} Hz - failed")
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
            elif command.get('command') == 'set_filter':
                low_cut = command.get('low_cut')
                high_cut = command.get('high_cut')
                if hasattr(self.source, 'set_filter'):
                    success = self.source.set_filter(low_cut, high_cut)
                    logger.info(f"Control command: set_filter low_cut={low_cut} high_cut={high_cut} - {'success' if success else 'failed'}")
                else:
                    logger.warning(f"Source does not support set_filter")
            elif command.get('command') == 'set_agc':
                agc_mode = command.get('agc_mode')
                manual_gain = command.get('manual_gain')
                threshold = command.get('threshold')
                slope = command.get('slope')
                decay = command.get('decay')
                if hasattr(self.source, 'set_agc'):
                    success = self.source.set_agc(agc_mode, manual_gain, threshold, slope, decay)
                    logger.info(f"Control command: set_agc - {'success' if success else 'failed'}")
                else:
                    logger.warning(f"Source does not support set_agc")
            elif command.get('command') == 'set_noise_blanker':
                enabled = command.get('enabled', 'false').lower() == 'true'
                if hasattr(self.source, 'set_noise_blanker'):
                    success = self.source.set_noise_blanker(enabled)
                    logger.info(f"Control command: set_noise_blanker enabled={enabled} - {'success' if success else 'failed'}")
                else:
                    logger.warning(f"Source does not support set_noise_blanker")

            # Handle start/stop capture
            elif command.get('command') == 'stop_capture':
                self.paused = True
                logger.info("Control command: Audio capture PAUSED")

            elif command.get('command') == 'start_capture':
                self.paused = False
                logger.info("Control command: Audio capture RESUMED")

            # Acknowledge message
            self.redis.xack(STREAM_CONTROL, self.control_group, msg_id)
            logger.debug(f"Acknowledged message {msg_id}")
                    
        except Exception as e:
            logger.error(f"Error processing control message {msg_id}: {e}", exc_info=True)

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

                # If paused, sleep and skip chunk reading
                if self.paused:
                    time.sleep(1.0)
                    continue

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
