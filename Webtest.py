# ws_gateway.py
import os
import json
import asyncio
import logging
import numpy as np
import websockets
from websockets.server import serve
from openai import AsyncAzureOpenAI

# Your existing LLM for text only (unchanged)
from Ai_engine import LLM_Client
from prompts import SYSTEM_PROMPT as _SYS_PROMPT, TTS_INSTRUCTIONS

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("ws_gateway")

WHISPER_WS_URL = os.getenv("WHISPER_WS_URL", "ws://127.0.0.1:9090")

# Azure TTS (streaming)
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://ttsmodel3.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_TTS_MODEL = os.getenv("AZURE_TTS_DEPLOYMENT", "gpt-4o-mini-tts")
AZURE_TTS_VOICE = os.getenv("AZURE_TTS_VOICE", "onyx")

HOST = os.getenv("WS_HOST", "0.0.0.0")
PORT = int(os.getenv("WS_PORT", "8765"))

# ---- Simple float32/16k resampler for safety (gateway expects 16k float32; browser already sends that) ----
def resample_linear_float32(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n = x.shape[0]
    if n == 0:
        return x
    t_out = np.linspace(0, 1, int(np.round(n * sr_out / sr_in)), endpoint=False)
    t_in = np.linspace(0, 1, n, endpoint=False)
    y = np.interp(t_out, t_in, x.astype(np.float32))
    return y.astype(np.float32)

class WhisperBridge:
    """Maintains a WS to your existing Whisper server; forwards audio, listens for partial/EOS."""
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.uid = "ws-bridge"
        self.buffer = []
        self.on_partial = None
        self.on_eos = None

        # metering
        self._tx_bytes = 0
        self._t0 = None

    async def start(self):
        self.ws = await websockets.connect(self.url, max_size=10*1024*1024)
        opts = {
            "uid": self.uid,
            "language": None,
            "task": "transcribe",
            "model": "small",
            "use_vad": True,
            "send_last_n_segments": 10,
            "no_speech_thresh": 0.45,
            "clip_audio": False,
            "same_output_threshold": 10,
            "enable_translation": False,
        }
        await self.ws.send(json.dumps(opts))
        asyncio.create_task(self._recv_loop())

    async def _recv_loop(self):
        async for raw in self.ws:
            try:
                msg = json.loads(raw)
            except:
                continue
            if msg.get("uid") != self.uid:
                continue
            if msg.get("message") == "SERVER_READY":
                LOG.info("[whisper] ready")
            elif "segments" in msg:
                segs = msg["segments"]
                self.buffer = [s.get("text","").strip() for s in segs[-4:]]
                if self.on_partial and self.buffer and self.buffer[-1]:
                    await self.on_partial(self.buffer[-1])
            elif msg.get("message") == "EOS":
                text = " ".join(self.buffer).strip()
                LOG.info(f"[whisper] EOS: {text}")
                if self.on_eos:
                    await self.on_eos(text)
                self.buffer = []

    async def send_pcm_f32_16k(self, chunk: bytes):
        if not chunk or self.ws is None:
            return
        await self.ws.send(chunk)
        # meter
        now = asyncio.get_event_loop().time()
        if self._t0 is None:
            self._t0 = now
        self._tx_bytes += len(chunk)
        if now - self._t0 >= 1.0:
            kbps = (self._tx_bytes * 8) / 1000.0 / (now - self._t0)
            LOG.info(f"[ingest→whisper] {self._tx_bytes/1024:.1f} KiB in {now - self._t0:.1f}s (~{kbps:.1f} kbps)")
            self._tx_bytes = 0
            self._t0 = now

    async def close(self):
        if self.ws:
            try:
                await self.ws.send(b"END_OF_AUDIO")
            except:
                pass
            try:
                await self.ws.close()
            except:
                pass

class TTSPusher:
    """Streams Azure TTS PCM to a browser over a WebSocket as 24k int16 frames."""
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version="2025-03-01-preview",
        )

    async def speak_to_ws(self, ws_client, text: str):
        # Tell browser a new TTS stream is starting
        await ws_client.send(json.dumps({"type": "tts_start"}))
        try:
            async with self.client.audio.speech.with_streaming_response.create(
                model=AZURE_TTS_MODEL,
                voice=AZURE_TTS_VOICE,
                input=text,
                instructions=TTS_INSTRUCTIONS,
                response_format="pcm",   # 24 kHz int16 mono
            ) as resp:
                async for chunk in resp.iter_bytes():
                    # Send binary chunks to browser (tag: tts_pcm24)
                    await ws_client.send(chunk)
        finally:
            # Tell browser we finished
            await ws_client.send(json.dumps({"type": "tts_end"}))

async def ws_handler(websocket):
    """
    Protocol (browser <-> gateway):
      - Browser sends JSON {"type":"hello"} once, then sends binary frames: 16k mono float32 PCM (20 ms suggested)
      - Browser may send {"type":"eos"} to force end-of-turn (optional; VAD in Whisper will also trigger EOS)
      - Gateway forwards audio to Whisper; on EOS calls LLM and streams TTS back as:
            JSON {"type":"tts_start"} 
            ... binary pcm24 chunks ...
            JSON {"type":"tts_end"}
    """
    LOG.info("[ws] client connected")
    whisper = WhisperBridge(WHISPER_WS_URL)
    await whisper.start()

    tts = TTSPusher()
    llm = LLM_Client(system_prompt=_SYS_PROMPT)

    async def on_partial(text: str):
        # Optional: forward partial to browser
        await websocket.send(json.dumps({"type":"partial","text":text}))
    whisper.on_partial = on_partial

    async def on_eos(text: str):
        # Generate assistant text and stream TTS back
        if not text:
            return
        reply = llm.generate_reply(text)
        await websocket.send(json.dumps({"type":"assistant_text","text":reply}))
        await tts.speak_to_ws(websocket, reply)
    whisper.on_eos = on_eos

    try:
        async for message in websocket:
            # JSON control?
            if isinstance(message, str):
                try:
                    ctrl = json.loads(message)
                except:
                    continue
                t = ctrl.get("type")
                if t == "hello":
                    await websocket.send(json.dumps({"type":"ack"}))
                elif t == "eos":
                    # Force EOS downstream (optional)
                    await whisper.ws.send(json.dumps({"message":"FORCE_EOS","uid":whisper.uid}))
                continue

            # Otherwise binary audio from browser
            if isinstance(message, (bytes, bytearray)):
                # expects 16k mono float32 little-endian
                await whisper.send_pcm_f32_16k(message)
    except websockets.exceptions.ConnectionClosed:
        LOG.info("[ws] client disconnected")
    finally:
        await whisper.close()

async def main():
    async with serve(ws_handler, host=HOST, port=PORT, max_size=10*1024*1024):
        LOG.info(f"[ws] listening on ws://{HOST}:{PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
