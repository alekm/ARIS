import asyncio
import websockets
import time
import random
import sys

HOST = "kiwisdr.moxley.us"
PORT = 8073
PASSWORD = ""

async def test_connection():
    timestamp = int(time.time())
    client_id = random.randint(0, 999999)
    
    # URL Format 1: Standard (what we have now)
    url = f"ws://{HOST}:{PORT}/{timestamp}/{client_id}"
    
    # URL Format 2: Alternative (some servers use this)
    # url = f"ws://{HOST}:{PORT}/kiwi/{client_id}/SND"
    
    print(f"Connecting to {url}...")
    
    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            print("Connected!")
            
            # 1. Send Auth
            auth_msg = f"SET auth t=kiwi p={PASSWORD}"
            print(f"Sending: {auth_msg}")
            await ws.send(auth_msg)
            
            # 2. Setup Audio (Required to start stream?)
            # Common sequence from kiwiclient
            setup_msgs = [
                "SET mod=lsb low_cut=-2700 high_cut=-300 freq=7200.00",
                "SET compression=0",
                "SET ident_user=ARIS_DEBUG",
                "SET ar_ok=1", # Audio Rate OK?
                "SET keepalive"
            ]
            
            for msg in setup_msgs:
                print(f"Sending: {msg}")
                await ws.send(msg)

            print("Waiting for messages...")
            async for message in ws:
                if isinstance(message, str):
                    print(f"RX TEXT: {message}")
                else:
                    print(f"RX BINARY: {len(message)} bytes")
                    # If we get binary, it's working!
                    break
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
