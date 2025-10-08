import asyncio
from client import Client  # your modified file with _start_webrtc_services in __init__

if __name__ == "__main__":
    # This connects to your Whisper Live server at 9090 (WS) and also starts WebRTC signaling (8765)
    c = Client(host="localhost", port=9090, lang="ar", model="small", use_wss=False)

    # Keep the process alive so the signaling WebSocket stays up
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print("Shutting down…")
