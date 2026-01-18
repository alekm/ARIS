import asyncio
import websockets
import time
import struct
import logging
import sys

# Configure logging to stdout
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test")

URI = "ws://kiwisdr.moxley.us:8073/kiwi/12345/SND"

async def test_connect():
    try:
        logger.info(f"Connecting to {URI}")
        async with websockets.connect(URI, ping_interval=20, ping_timeout=20) as ws:
            logger.info("Connected!")
            time.sleep(1)

            logger.info("Sending ident_user...")
            await ws.send("SET ident_user=TEST_SCRIPT")
            time.sleep(1)

            logger.info("Sending auth...")
            await ws.send("SET auth t=kiwi p=")
            time.sleep(1)
            
            logger.info("Sending zoom...")
            await ws.send("SET zoom=3 start=0")
            time.sleep(1)

            logger.info("Sending mod/freq...")
            # Using new filter settings
            await ws.send("SET mod=lsb low_cut=300 high_cut=2700 freq=7188.000")
            time.sleep(1)

            logger.info("Sending compression...")
            await ws.send("SET compression=0")
            time.sleep(1)

            logger.info("Sending keepalive...")
            await ws.send("SET keepalive")
            time.sleep(1)
            
            # NOT sending gen=0 mix=-1 yet
            
            logger.info("Listening...")
            async for msg in ws:
                if isinstance(msg, bytes) and msg.startswith(b'MSG '):
                    text = msg.decode('utf-8', errors='ignore')
                    logger.info(f"MSG: {text}")
                    if "audio_rate=" in text:
                         # Respond to AR
                         val = text.split("audio_rate=")[1].split()[0]
                         rate = int(float(val))
                         logger.info(f"Sending AR OK for rate {rate}")
                         await ws.send(f"SET AR OK in={rate} out=44100")
                         
                         logger.info("Now starting stream...")
                         await ws.send("SET gen=0 mix=-1")

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_connect())
