# signaling_bridge.py
import asyncio
import json
from aiohttp import web
from config_rtc import SIGNALING_HOST, SIGNALING_PORT

class SignalingServer:
    """
    Minimal signaling over WebSocket:
      - Browser connects to /ws
      - Sends {"type":"offer","sdp":...} then ICE {"type":"ice","candidate":...}
      - We reply with {"type":"answer","sdp":...} and relay ICE from server.
    """
    def __init__(self, on_offer, on_ice_from_client, on_ws_ready=None, logger=print):
        self.on_offer = on_offer
        self.on_ice_from_client = on_ice_from_client
        self.on_ws_ready = on_ws_ready
        self.logger = logger
        self._ws = None

    async def _ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws = ws
        if self.on_ws_ready:
            await self.on_ws_ready(ws)

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "offer":
                    ans_sdp = await self.on_offer(data["sdp"])
                    await ws.send_str(json.dumps({"type": "answer", "sdp": ans_sdp}))
                elif data.get("type") == "ice":
                    await self.on_ice_from_client(data["candidate"])
            else:
                continue
        return ws

    async def send_ice_to_client(self, candidate: dict):
        if self._ws:
            await self._ws.send_str(json.dumps({"type": "ice", "candidate": candidate}))

    def run(self):
        app = web.Application()
        app.router.add_get("/ws", self._ws_handler)
        web.run_app(app, host=SIGNALING_HOST, port=SIGNALING_PORT)
