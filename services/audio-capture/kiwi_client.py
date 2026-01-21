import asyncio
import websockets
import time
import random
import logging
import struct
import numpy as np
import array

import os

logger = logging.getLogger(__name__)

# Reduce noise from websockets library
logging.getLogger("websockets.client").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)
logging.getLogger("websockets.protocol").setLevel(logging.WARNING)

# IMAADPCM Tables
stepSizeTable = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34,
    37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494,
    544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552,
    1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026,
    4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442,
    11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623,
    27086, 29794, 32767)

indexAdjustTable = [
    -1, -1, -1, -1,  # +0 - +3, decrease the step size
     2, 4, 6, 8,     # +4 - +7, increase the step size
    -1, -1, -1, -1,  # -0 - -3, decrease the step size
     2, 4, 6, 8      # -4 - -7, increase the step size
]

def clamp(x, xmin, xmax):
    if x < xmin:
        return xmin
    if x > xmax:
        return xmax
    return x

class ImaAdpcmDecoder(object):
    def __init__(self):
        self.index = 0
        self.prev = 0

    def preset(self, index, prev):
        self.index = index
        self.prev = prev

    def _decode_sample(self, code):
        step = stepSizeTable[self.index]
        self.index = clamp(self.index + indexAdjustTable[code], 0, len(stepSizeTable) - 1)
        difference = step >> 3
        if ( code & 1 ):
            difference += step >> 2
        if ( code & 2 ):
            difference += step >> 1
        if ( code & 4 ):
            difference += step
        if ( code & 8 ):
            difference = -difference
        sample = clamp(self.prev + difference, -32768, 32767)
        self.prev = sample
        return sample

    def decode(self, data):
        fcn = ord if isinstance(data, str) else lambda x : x
        samples = array.array('h')
        for b in map(fcn, data):
            sample0 = self._decode_sample(b & 0x0F)
            sample1 = self._decode_sample(b >> 4)
            samples.append(sample0)
            samples.append(sample1)
        return samples

class KiwiSDRClient:
    def __init__(self, host, port, password="", frequency_hz=14200000, mode="USB", audio_endian="big"):
        self.host = host
        self.port = port
        self.password = password
        self.frequency_hz = frequency_hz
        self.mode = mode.upper()
        self.running = False
        self.ws = None
        self.sample_rate = 12000
        self.audio_endian = audio_endian.lower()  # "big" or "little"
        
        # Audio Callback: async func(audio_bytes)
        self.on_audio_data = None 
        self.last_audio_time = time.time()
        
        # Decoder - reset state on each connection
        self.decoder = ImaAdpcmDecoder()
        # Reset decoder state to defaults (important for clean decoding)
        self.decoder.preset(0, 0)

    async def connect(self):
        self.running = True
        timestamp = int(time.time())
        # Standard KiwiSDR URL format: /{timestamp}/SND
        url = f"ws://{self.host}:{self.port}/{timestamp}/SND"
        
        logger.info(f"Connecting to KiwiSDR at {self.host}:{self.port}")
        logger.debug(f"WebSocket URL: {url}")
        
        try:
            # Set connection timeout to 30 seconds (default is often 10s which may be too short)
            # open_timeout controls the WebSocket handshake timeout
            # This helps with slow networks or when KiwiSDR is under load
            async with websockets.connect(
                url, 
                ping_interval=None,
                open_timeout=30.0,  # 30 second timeout for handshake
                close_timeout=10.0  # 10 second timeout for close
            ) as ws:
                self.ws = ws
                logger.info("WebSocket Connected")
                
                # Reset decoder state for new connection
                self.decoder.preset(0, 0)
                
                # Handshake Sequence
                logger.info("Sending Auth...")
                await self._send_auth()
                
                async for message in ws:
                    if not self.running: break
                    
                    if isinstance(message, str):
                        await self._handle_text(message)
                    else:
                        if message.startswith(b'MSG'):
                            try:
                                msg_text = message.decode('utf-8', errors='ignore')
                                logger.info(f"Kiwi BINARY MSG: {msg_text[4:]}")
                                await self._handle_msg(msg_text[4:])
                            except:
                                pass
                            continue

                        if message.startswith(b'ntsw'):
                            continue

                        await self._handle_binary(message)
                        
        except Exception as e:
            logger.error(f"KiwiSDR Connection Error: {e}")
            raise e
        finally:
            self.running = False
            pass

    async def stop(self):
        self.running = False
        if self.ws:
            await self.ws.close()

    async def set_frequency(self, frequency_hz: int):
        """Update frequency dynamically without reconnecting"""
        if not self.ws or not self.running:
            logger.warning("Cannot set frequency: not connected")
            return
        
        self.frequency_hz = frequency_hz
        freq_khz = frequency_hz / 1000.0
        
        # Send frequency update command
        cmd = f"SET freq={freq_khz:.4f}"
        logger.info(f"[KiwiSDR] Frequency update: {frequency_hz} Hz ({freq_khz:.4f} kHz)")
        logger.info(f"[KiwiSDR] >> {cmd}")
        await self.ws.send(cmd)
        await asyncio.sleep(0.01)  # Small delay for command processing

    async def set_mode(self, mode: str):
        """Update demodulation mode and filters dynamically without reconnecting"""
        if not self.ws or not self.running:
            logger.warning("Cannot set mode: not connected")
            return
        
        self.mode = mode.upper()
        
        # Calculate filter cutoffs based on mode
        low, high = 300, 2700 
        if self.mode == "LSB": low, high = -2700, -300
        elif self.mode == "AM": low, high = -5000, 5000
        elif self.mode == "CW": low, high = 300, 700
        
        freq_khz = self.frequency_hz / 1000.0
        
        # Send mode and filter update commands
        cmds = [
            f"SET mod={self.mode.lower()} low_cut={low} high_cut={high}",
        ]
        
        logger.info(f"[KiwiSDR] Mode update: {self.mode}, filter={low}-{high}Hz")
        for cmd in cmds:
            logger.info(f"[KiwiSDR] >> {cmd}")
            await self.ws.send(cmd)
            await asyncio.sleep(0.01)  # Small delay for command processing

    async def _send_auth(self):
        auth_msg = f"SET auth t=kiwi p={self.password}"
        logger.debug(f">> {auth_msg}")
        await self.ws.send(auth_msg)
        
        # No client_type or ver per official minimalist approach

    async def _handle_msg(self, msg_text):
        if "audio_rate=" in msg_text:
            try:
                parts = msg_text.split()
                rate = 12000 # Default
                for p in parts:
                    if p.startswith("audio_rate="):
                        rate = int(p.split("=")[1])
                logger.info(f"Received audio_rate={rate}, sending AR OK and Config...")
                await self.ws.send(f"SET AR OK in={rate} out={rate}")
                await self._send_config()
            except Exception as e:
                logger.error(f"Failed to parse audio_rate: {e}")

        elif "sample_rate=" in msg_text:
            logger.info("Received sample_rate, sending configuration...")
            await self._send_config()

    async def _send_config(self):
        low, high = 300, 2700 
        if self.mode == "LSB": low, high = -2700, -300
        elif self.mode == "AM": low, high = -5000, 5000
        elif self.mode == "CW": low, high = 300, 700  # Asymmetric, positive side only (matches kiwiclient default)
        freq_khz = self.frequency_hz / 1000.0
        
        # Log exact frequency being sent for debugging
        # Note: KiwiSDR applies a BFO offset (typically 500 Hz) in CW mode, which shifts the signal
        # to an audible tone. The frequency sent here is the center frequency; the actual audio tone
        # will appear at the BFO offset frequency (e.g., 500 Hz) in the demodulated audio.
        logger.info(f"[KiwiSDR] Config: mode={self.mode}, freq_hz={self.frequency_hz}, freq_khz={freq_khz:.6f}, filter={low}-{high}Hz")
        
        # AGC settings: Match kiwiclient defaults exactly
        # kiwiclient default: set_agc(on=True) = agc=1 hang=0 thresh=-100 slope=6 decay=1000 gain=50
        # Use same gain for all modes - let AGC handle level differences
        agc_cmd = "SET agc=1 hang=0 thresh=-100 slope=6 decay=1000 manGain=50"
        
        # Command order matters: enable sound first, then compression, then configure mode/filters
        cmds = [
            "SET snd=1",  # Enable sound streaming FIRST (required before other audio settings)
            "SET compression=1", # ADPCM (matches web client - required for audio stream)
            f"SET mod={self.mode.lower()} low_cut={low} high_cut={high} freq={freq_khz:.4f}",
            "SET ident_user=ARIS_BOT",
            "SET keepalive",
            agc_cmd,
            "SET squelch=0 max=0",
            "SET genattn=0",
            "SET gen=0 mix=-1"
        ]
        
        for cmd in cmds:
            logger.info(f"[KiwiSDR] >> {cmd}")  # Changed to INFO so we can see what's being sent
            await self.ws.send(cmd)
            # Small delay to ensure commands are processed in order
            await asyncio.sleep(0.01)

        if not hasattr(self, 'keepalive_task') or not self.keepalive_task or self.keepalive_task.done():
            logger.info("Starting Keepalive Loop")
            self.keepalive_task = asyncio.create_task(self._keepalive_loop())
        else:
            logger.info("Keepalive Loop already running")

    async def _keepalive_loop(self):
        while self.running and self.ws:
            try:
                await asyncio.sleep(5)
                logger.debug(">> SET keepalive")
                await self.ws.send("SET keepalive")
            except Exception as e:
                break

    async def _handle_text(self, msg):
        logger.debug(f"<< TEXT: {msg}") 
        if msg.startswith("MSG"):
            logger.info(f"Kiwi MSG: {msg[4:]}")
            await self._handle_msg(msg[4:])

    async def _handle_binary(self, data):
        # KiwiSDR audio packet format: 
        # Bytes 0-2: 'SND'
        # Byte 3: Flags (0x10 = Compressed/ADPCM, 0x00 = PCM)
        # Bytes 4-7: Sequence Number (Little Endian)
        # Bytes 8+: Audio Data

        if data.startswith(b'MSG') or data.startswith(b'ntsw'):
            return

        self.last_audio_time = time.time()
        
        if self.on_audio_data:
            try:
                flags = data[3]
                seq = struct.unpack('<I', data[4:8])[0]
                raw_data = data[8:]
                
                # Check for Compression Flag (0x10)
                is_compressed = bool(flags & 0x10)
                
                if is_compressed:
                    # ADPCM Decode
                    samples = self.decoder.decode(raw_data)
                    audio_bytes = samples.tobytes()
                    logger.debug(f"Audio: ADPCM decoded, {len(audio_bytes)} bytes")
                else:
                    # Uncompressed PCM - handle endianness conversion
                    # KiwiSDR typically sends PCM in big-endian format
                    if self.audio_endian == "big":
                        # Convert big-endian to little-endian (native for intel)
                        # Unpack as big-endian 16-bit signed integers, repack as little-endian
                        num_samples = len(raw_data) // 2
                        samples = struct.unpack(f'>{num_samples}h', raw_data)
                        audio_bytes = struct.pack(f'<{num_samples}h', *samples)
                        logger.debug(f"Audio: PCM big-endian converted, {len(audio_bytes)} bytes")
                    else:
                        # Already little-endian, use as-is
                        audio_bytes = raw_data
                        logger.debug(f"Audio: PCM little-endian (as-is), {len(audio_bytes)} bytes")

                await self.on_audio_data(audio_bytes, seq)
            except Exception as e:
                logger.error(f"Binary decode error: {e}", exc_info=True)
                pass
