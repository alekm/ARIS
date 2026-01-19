
import asyncio
import websockets
import json
import sys

async def test():
    uri = "ws://localhost:8000/ws"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            # Wait for first message
            msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            print(f"Received: {msg}")
            
            # Expecting SLOT_UPDATE shortly
            if "SLOT_UPDATE" in msg:
                print("SUCCESS: SLOT_UPDATE received.")
            else:
                # Wait for next
                msg2 = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"Received 2: {msg2}")
                if "SLOT_UPDATE" in msg2:
                   print("SUCCESS: SLOT_UPDATE received.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test())
