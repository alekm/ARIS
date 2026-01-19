
import asyncio
import websockets
import sys

async def test_connect():
    uri = "ws://localhost:8000/ws"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            msg = await websocket.recv()
            print(f"Received: {msg}")
            
            # Send ping
            await websocket.send("ping")
            pong = await websocket.recv()
            print(f"Received: {pong}")
            
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_connect())
