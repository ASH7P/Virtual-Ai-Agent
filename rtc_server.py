# rtc_gateway.py
import os
import json
import ssl
import asyncio
import logging
import numpy as np
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
)
from av.audio.frame import AudioFrame
import websockets

from openai import AsyncAzureOpenAI

# === your text agent (unchanged) ===
# We ONLY import LLM_Client to get the text reply. We do not touch its TTS-to-speaker code.
from Ai_engine import LLM_Client
from prompts import SYSTEM_PROMPT as _SYS_PROMPT, TTS_INSTRUCTIONS

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("rtc_gateway")

# --------- ENV / CONFIG ----------
WHISPER_WS_URL = os.getenv("WHISPER_WS_URL", "ws://127.0.0.1:9090")
SYSTEM_PROMPT = _SYS_PROMPT  # avoid name shadowing

# Azure creds read here only for TTS (your LLM_Client will read its own envs for chat)
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://ttsmodel3.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY",
    "2nnLldWRJFxVegf69V94gzqF2QzlkgYaaISUiCl2bLt6YHRDFRqZJQQJ99BIACHYHv6XJ3w3AAABACOG0eBS",
)
AZURE_TTS_MODEL = os.getenv("AZURE_TTS_DEPLOYMENT", "gpt-4o-mini-tts")
AZURE_TTS_VOICE = os.getenv("AZURE_TTS_VOICE", "onyx")

# ICE / TURN
TURN_URL = os.getenv("TURN_URL", "")          # e.g. "turn:turn.yourdomain.com:3478"
TURN_USER = os.getenv("TURN_USER", "")        # e.g. "webrtc"
TURN_PASS = os.getenv("TURN_PASS", "")        # e.g. strong password
STUN_URL = os.getenv("STUN_URL", "stun:stun.l.google.com:19302")

# App network bind
RTC_HOST = os.getenv("RTC_HOST", "0.0.0.0")
RTC_PORT = int(os.getenv("RTC_PORT", "8080"))

# HTTPS (optional; usually put TLS on a reverse proxy instead)
ENABLE_SSL = os.getenv("ENABLE_SSL", "false").lower() == "true"
SSL_CERT = os.getenv("SSL_CERT", "")
SSL_KEY = os.getenv("SSL_KEY", "")

# --------- helpers ----------
def resample_linear_int16(x_int16: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x_int16
    n = x_int16.shape[0]
    if n == 0:
        return x_int16
    t_out = np.linspace(0, 1, int(np.round(n * sr_out / sr_in)), endpoint=False)
    t_in = np.linspace(0, 1, n, endpoint=False)
    y = np.interp(t_out, t_in, x_int16.astype(np.float32))
    y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    return y

def float32_from_int16(x_int16: np.ndarray) -> np.ndarray:
    return (x_int16.astype(np.float32) / 32768.0)

# --------- outbound audio track (to browser) ----------
class OutboundAudioTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self, sample_rate=48000):
        super().__init__()
        self.queue = asyncio.Queue()
        self.sample_rate = sample_rate

    async def recv(self) -> AudioFrame:
        pcm = await self.queue.get()  # bytes int16 mono @ sample_rate
        samples = np.frombuffer(pcm, dtype=np.int16)
        frame = AudioFrame(format="s16", layout="mono", samples=len(samples))
        frame.sample_rate = self.sample_rate
        frame.planes[0].update(samples.tobytes())
        return frame

# --------- Whisper websocket bridge ----------
class WhisperBridge:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.ws = None
        self.uid = "rtc-bridge"
        self.current_text = []
        self.on_eos = None  # async def on_eos(text: str): ...

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url)
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
            except Exception:
                continue
            if msg.get("uid") != self.uid:
                continue

            if "status" in msg:
                LOG.info(f"[whisper] {msg['status']}: {msg.get('message','')}")
                continue

            if msg.get("message") == "SERVER_READY":
                LOG.info("[whisper] ready")
                continue

            if "segments" in msg:
                segs = msg["segments"]
                last = [s.get("text", "").strip() for s in segs[-4:]]
                self.current_text = last
                if last and last[-1]:
                    LOG.info(f"[whisper] partial: {last[-1]}")
                continue

            if msg.get("message") == "EOS":
                text = " ".join(self.current_text).strip()
                LOG.info(f"[whisper] EOS: {text}")
                if self.on_eos:
                    try:
                        await self.on_eos(text)
                    except Exception as e:
                        LOG.exception(f"on_eos failed: {e}")
                self.current_text = []

    async def send_audio(self, pcm_f32_mono_16k: bytes):
        if self.ws is None or not pcm_f32_mono_16k:
            return
        try:
            # IMPORTANT: bytes => Binary WS frame
            await self.ws.send(pcm_f32_mono_16k)

            # --- metering: log once per ~1s ---
            t = asyncio.get_event_loop().time()
            if not hasattr(self, "_meter_bytes"):
                self._meter_bytes = 0
                self._meter_last = t
            self._meter_bytes += len(pcm_f32_mono_16k)
            if (t - self._meter_last) >= 1.0:
                kbps = (self._meter_bytes * 8) / 1000.0 / (t - self._meter_last)
                LOG.info(f"[ingest→whisper] {self._meter_bytes/1024:.1f} KiB sent in {t - self._meter_last:.1f}s (~{kbps:.1f} kbps)")
                self._meter_bytes = 0
                self._meter_last = t
        except Exception as e:
            LOG.error(f"[whisper] send error: {e}")

# --------- Azure TTS → WebRTC track ----------
class TTSToTrack:
    def __init__(self, track: OutboundAudioTrack, playback_rate=48000):
        self.track = track
        self.playback_rate = playback_rate
        if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
            LOG.warning("Azure TTS envs missing; set ENDPOINT_URL and AZURE_OPENAI_API_KEY.")
        self.tts = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version="2025-03-01-preview",
        )

    async def speak(self, text: str, voice: str = AZURE_TTS_VOICE, model: str = AZURE_TTS_MODEL):
        # Azure returns 24kHz int16 PCM. Resample to 48kHz for WebRTC.
        async with self.tts.audio.speech.with_streaming_response.create(
            model=model,
            instructions=TTS_INSTRUCTIONS,
            voice=voice,
            input=text,
            response_format="pcm",
        ) as resp:
            async for chunk in resp.iter_bytes():
                in_24k = np.frombuffer(chunk, dtype=np.int16)
                out_48k = resample_linear_int16(in_24k, 24000, self.playback_rate)
                await self.track.queue.put(out_48k.tobytes())

# --------- HTTP / signaling ----------
routes = web.RouteTableDef()
pcs = set()

def _rtc_config():
    ice_servers = []
    if STUN_URL:
        ice_servers.append(RTCIceServer(urls=[STUN_URL]))
    if TURN_URL and TURN_USER and TURN_PASS:
        ice_servers.append(RTCIceServer(urls=[TURN_URL], username=TURN_USER, credential=TURN_PASS))
    return RTCConfiguration(iceServers=ice_servers)

@routes.post("/offer")
async def offer(request: web.Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # 1) Create the PeerConnection for THIS offer
    pc = RTCPeerConnection(configuration=_rtc_config())
    pcs.add(pc)
    LOG.info("PeerConnection created")

    # 2) Outbound audio (TTS -> browser)
    tts_track = OutboundAudioTrack(sample_rate=48000)
    pc.addTrack(tts_track)

    # 3) Whisper bridge (WS) + LLM + TTS
    whisper = WhisperBridge(WHISPER_WS_URL)
    await whisper.connect()

    llm = LLM_Client(system_prompt=SYSTEM_PROMPT)
    tts = TTSToTrack(tts_track, playback_rate=48000)

    async def on_eos(text: str):
        if not text:
            return
        reply = llm.generate_reply(text)
        LOG.info(f"[LLM] reply: {reply}")
        await tts.speak(reply)

    whisper.on_eos = on_eos

    # 4) Register the on_track handler ON THIS pc (MUST be inside offer)
    @pc.on("track")
    def on_track(track):
        if track.kind != "audio":
            return
        LOG.info("Browser audio track received")

        async def forward_audio_to_whisper():
            LOG.info("[ingest] forwarder started")
            FRAME_SAMPLES = 320      # 20 ms @16k
            FRAME_BYTES   = 320 * 4  # float32
            buf = bytearray()
            frames_sent = 0
            first_log = True
            while True:
                try:
                    frame = await track.recv()
                except Exception as e:
                    LOG.info(f"[ingest] audio recv ended: {e}")
                    break

                arr = frame.to_ndarray()
                if arr.ndim == 2:
                    if hasattr(frame, "layout") and getattr(frame.layout, "channels", None) and arr.shape[0] == frame.layout.channels:
                        mono = arr[0, :]
                    else:
                        mono = arr[:, 0]
                else:
                    mono = arr

                if mono.dtype == np.int16:
                    pcm_s16 = mono
                elif mono.dtype == np.int32:
                    pcm_s16 = (mono >> 16).astype(np.int16)
                elif mono.dtype == np.float32:
                    pcm_s16 = np.clip(mono * 32768.0, -32768, 32767).astype(np.int16)
                elif mono.dtype == np.float64:
                    pcm_s16 = np.clip(mono * 32768.0, -32768, 32767).astype(np.int16)
                else:
                    pcm_s16 = mono.astype(np.int16, copy=False)

                sr_in = getattr(frame, "sample_rate", None) or 48000
                if first_log:
                    LOG.info(f"[ingest] first frame: sr_in={sr_in}, shape={arr.shape}, dtype={arr.dtype}, mono_samples={pcm_s16.shape[0]}")
                    first_log = False

                if pcm_s16.size == 0:
                    continue
                pcm16k = resample_linear_int16(pcm_s16.reshape(-1), sr_in, 16000)
                if pcm16k.size == 0:
                    continue

                buf.extend((pcm16k.astype(np.float32) / 32768.0).tobytes())

                while len(buf) >= FRAME_BYTES:
                    payload = bytes(buf[:FRAME_BYTES]); del buf[:FRAME_BYTES]
                    if payload:
                        await whisper.send_audio(payload)
                        frames_sent += 1
                        if frames_sent % 25 == 0:
                            LOG.info(f"[ingest] frames_sent={frames_sent} (frame={FRAME_BYTES}B)")

            if 0 < len(buf) < FRAME_BYTES:
                buf.extend(b"\x00" * (FRAME_BYTES - len(buf)))
            if buf:
                await whisper.send_audio(bytes(buf))
                LOG.info(f"[ingest] flushed tail {len(buf)} bytes")

        asyncio.create_task(forward_audio_to_whisper())

    LOG.info("on_track handler registered")  # <-- you should see this

    # 5) Connection state cleanup (also inside offer so it closes the right pc)
    @pc.on("connectionstatechange")
    async def _on_state():
        LOG.info(f"pc.state={pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await whisper.close()
            await pc.close()
            pcs.discard(pc)

    # 6) Finish SDP handshake
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


# ---------- Simple client (served at "/") ----------
INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AI Receptionist (WebRTC)</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:2rem}
 button{padding:.6rem 1rem;font-size:1rem}
 #status{margin-top:1rem;white-space:pre-line}
</style>
</head>
<body>
<h1>AI Receptionist (WebRTC)</h1>
<p>Click Start, allow mic, speak. TTS reply streams back via WebRTC.</p>
<button id="start">Start</button>
<div id="status"></div>
<audio id="remote" autoplay playsinline></audio>
<script>
const log = m => document.getElementById('status').textContent += m + "\\n";
document.getElementById('start').onclick = async () => {
  const pc = new RTCPeerConnection({
    iceServers: [
      { urls: "%STUN_URL%" }%TURN_BLOCK%
    ]
  });
  pc.onconnectionstatechange = () => log("pc: " + pc.connectionState);
  pc.oniceconnectionstatechange = () => log("ice: " + pc.iceConnectionState);
  pc.ontrack = ev => {
    if (ev.track.kind === 'audio') {
      document.getElementById('remote').srcObject = ev.streams[0];
      log("Remote audio attached.");
    }
  };
  const stream = await navigator.mediaDevices.getUserMedia({audio:true, video:false});
  stream.getAudioTracks().forEach(t => pc.addTrack(t, stream));
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const resp = await fetch('/offer', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({sdp: offer.sdp, type: offer.type})
  });
  const answer = await resp.json();
  await pc.setRemoteDescription(answer);
  log("Connected. Speak!");
};
</script>
</body>
</html>
""".replace(
    "%STUN_URL%", STUN_URL or "stun:stun.l.google.com:19302"
).replace(
    "%TURN_BLOCK%",
    (", { urls: '%s', username: '%s', credential: '%s' }" % (TURN_URL, TURN_USER, TURN_PASS))
    if (TURN_URL and TURN_USER and TURN_PASS) else ""
)

@routes.get("/")
async def index(_):
    return web.Response(text=INDEX_HTML, content_type="text/html")

def _ssl_ctx():
    if not ENABLE_SSL:
        return None
    if not (SSL_CERT and SSL_KEY):
        raise RuntimeError("ENABLE_SSL=true but SSL_CERT/SSL_KEY not set")
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(SSL_CERT, SSL_KEY)
    return ctx

def main():
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app, host=RTC_HOST, port=RTC_PORT, ssl_context=_ssl_ctx())

if __name__ == "__main__":
    main()
