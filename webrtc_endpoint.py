# webrtc_endpoint.py
import asyncio
import json
import numpy as np
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from aiortc.rtcrtpsender import RTCRtpSender
from av.audio.resampler import AudioResampler

from config_rtc import ICE_SERVERS, OPUS_BITRATE, OPUS_PTIME_MS, WEBRTC_AUDIO_RATE, MONO

def _make_rtc_config():
    servers = []
    # ICE_SERVERS is your list of dicts like {"urls": "stun:stun.l.google.com:19302", ...}
    for s in ICE_SERVERS:
        if isinstance(s, dict):
            servers.append(RTCIceServer(
                urls=s.get("urls"),
                username=s.get("username"),
                credential=s.get("credential"),
            ))
        elif isinstance(s, str):
            servers.append(RTCIceServer(urls=s))
    return RTCConfiguration(iceServers=servers)

class _TTSSourceTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self, rate=WEBRTC_AUDIO_RATE):
        super().__init__()
        self._queue = asyncio.Queue()
        self._rate = rate

    async def push_f32_mono_48k(self, f32: np.ndarray):
        await self._queue.put(f32.astype(np.float32))

    async def recv(self):
        samples = await self._queue.get()
        s16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(samples))
        frame.sample_rate = self._rate
        frame.planes[0].update(s16)
        return frame

class WebRTCEndpoint:
    """
    One PeerConnection that:
      - receives browser mic (Opus -> PCM 48k) and calls on_uplink_frame(f32_48k)
      - sends TTS via an internal audio track (we encode to Opus automatically)
    """
    def __init__(self, on_uplink_frame=None, logger=print):
        self.pc = RTCPeerConnection(configuration=_make_rtc_config())
        self.on_uplink_frame = on_uplink_frame
        self._logger = logger
        self._resampler_48 = AudioResampler(format="s16", layout="mono", rate=WEBRTC_AUDIO_RATE)
        self._tts_track = _TTSSourceTrack(rate=WEBRTC_AUDIO_RATE)
        self.pc.addTrack(self._tts_track)

        @self.pc.on("track")
        def on_track(track):
            if track.kind != "audio":
                return
            self._logger("[RTC] audio track received")
            asyncio.create_task(self._consume_uplink(track))

    async def _consume_uplink(self, track: MediaStreamTrack):
        """Convert incoming Opus to f32 mono 48k and callback."""
        try:
            while True:
                frame = await track.recv()  # av.AudioFrame (various rates)
                frame.sample_rate = getattr(frame, "sample_rate", WEBRTC_AUDIO_RATE)
                for f in self._resampler_48.resample(frame):
                    s16 = np.frombuffer(f.planes[0].to_bytes(), dtype="<i2")
                    f32 = (s16.astype(np.float32) / 32768.0).clip(-1, 1)
                    if self.on_uplink_frame:
                        await self.on_uplink_frame(f32)  # f32 mono @ 48k
        except Exception as e:
            self._logger(f"[RTC] uplink ended: {e}")

    async def create_answer(self, offer_sdp: str):
        await self.pc.setRemoteDescription(RTCSessionDescription(offer_sdp, "offer"))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        # Opus tuning hint (most stacks honor via RTCP/bitrate control anyway)
        RTCRtpSender.getCapabilities("audio")
        return self.pc.localDescription.sdp

    async def add_ice(self, candidate: dict):
        try:
            await self.pc.addIceCandidate(candidate)
        except Exception:
            pass

    async def send_tts_pcm_f32_48k(self, f32: np.ndarray):
        """Push float32 mono @48k to downlink track."""
        await self._tts_track.push_f32_mono_48k(f32)

    async def close(self):
        await self.pc.close()
