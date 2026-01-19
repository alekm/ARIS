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
    def __init__(self, host, port, password="", frequency_hz=14200000, mode="USB"):
        self.host = host
        self.port = port
        self.password = password
        self.frequency_hz = frequency_hz
        self.mode = mode.upper()
        self.running = False
        self.ws = None
        self.sample_rate = 12000
        
        # Audio Callback: async func(audio_bytes)
        self.on_audio_data = None 
        self.last_audio_time = time.time()
        
        # Decoder
        self.decoder = ImaAdpcmDecoder()

    async def connect(self):
        self.running = True
        timestamp = int(time.time())
        # Standard KiwiSDR URL format: /{timestamp}/SND
        url = f"ws://{self.host}:{self.port}/{timestamp}/SND"
        
        logger.info(f"Connecting to KiwiSDR: {url}")
        
        try:
            async with websockets.connect(url, ping_interval=None) as ws:
                self.ws = ws
                logger.info("WebSocket Connected")
                
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
        elif self.mode == "CW": low, high = 400, 800
        freq_khz = self.frequency_hz / 1000.0
        
        cmds = [
            f"SET mod={self.mode.lower()} low_cut={low} high_cut={high} freq={freq_khz:.2f}",
            "SET compression=1", # ADPCM
            "SET ident_user=ARIS_BOT",
            "SET keepalive",
            "SET agc=1 hang=0 thresh=-100 slope=6 decay=1000 manGain=48",
            "SET squelch=0 max=0",
            "SET genattn=0",
            "SET gen=0 mix=-1",
            "SET snd=1" # Keep enabled
        ]
        
        for cmd in cmds:
            logger.debug(f">> {cmd}")
            await self.ws.send(cmd)

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
                if flags & 0x10:
                    # ADPCM Decode
                    samples = self.decoder.decode(raw_data)
                    audio_bytes = samples.tobytes()
                else:
                    # PCM (Already Little Endian? or Big?)
                    # If PCM, it is usually Big Endian (>h) per previous findings,
                    # but ADPCM output is native (Little Endian on intel).
                    audio_bytes = raw_data

                await self.on_audio_data(audio_bytes, seq)
            except Exception as e:
                logger.error(f"Binary decode error: {e}")
                pass
