# webrtc_endpoint.py
import asyncio
import json
import numpy as np
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from aiortc.rtcrtpsender import RTCRtpSender
from av.audio.resampler import AudioResampler
from fractions import Fraction

from config_rtc import ICE_SERVERS, WEBRTC_AUDIO_RATE


def _make_rtc_config():
    servers = []
    # ICE_SERVERS is your list of dicts like:
    # {"urls": "stun:stun.l.google.com:19302"} or strings with the URL directly
    for s in ICE_SERVERS:
        if isinstance(s, dict):
            servers.append(
                RTCIceServer(
                    urls=s.get("urls"),
                    username=s.get("username"),
                    credential=s.get("credential"),
                )
            )
        elif isinstance(s, str):
            servers.append(RTCIceServer(urls=s))
    return RTCConfiguration(iceServers=servers)


class _TTSSourceTrack(MediaStreamTrack):
    """
    Custom aiortc audio track that takes float32 mono @48k input and emits
    20 ms (960-sample) s16 frames to the PeerConnection with stable timing.

    Key points:
    - Uses an asyncio.Queue for precise backpressure instead of polling.
    - Pre-chunks on push so recv() simply wraps already-sized packets.
    - Drops oldest packets when queue is full to prevent latency buildup.
    """
    kind = "audio"

    def __init__(self, rate=WEBRTC_AUDIO_RATE, queue_packets=60):
        """
        :param rate: sample rate (Hz), default 48000
        :param queue_packets: max queued 20ms packets (60 -> ~1.2s of audio)
        """
        super().__init__()
        self._rate = rate
        self._chunk = int(self._rate * 0.020)  # 20 ms -> 960 at 48k
        self._ts = 0  # running timestamp in samples
        self._q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_packets)
        self._stash = np.empty(0, dtype=np.float32)  # leftover samples < 960
        self._sent = 0
        self._closed = False

    async def push_f32_mono_48k(self, f32: np.ndarray):
        """
        Accepts arbitrary-length float32 mono @48k samples in [-1, 1],
        chunks into 960-sample packets, converts to s16 LE bytes,
        and enqueues packets for send.
        """
        if self._closed or f32 is None or f32.size == 0:
            return

        f32 = f32.astype(np.float32).reshape(-1)
        # Accumulate with any leftover
        buf = np.concatenate((self._stash, f32))
        n = buf.size

        # Number of full packets available
        full = (n // self._chunk) * self._chunk
        if full > 0:
            # Vectorized convert each contiguous block of size _chunk
            packets = buf[:full].reshape(-1, self._chunk)

            # Convert [-1,1] float32 -> s16 bytes for each packet
            # Clip first to avoid wrap-around
            s16_all = (np.clip(packets, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

            # Split the big byte blob into per-packet slices (each packet is _chunk * 2 bytes)
            pkt_size_bytes = self._chunk * 2
            for i in range(0, len(s16_all), pkt_size_bytes):
                packet = s16_all[i : i + pkt_size_bytes]

                # Drop oldest if queue is full (keep latency low)
                if self._q.full():
                    try:
                        _ = self._q.get_nowait()
                        self._q.task_done()
                    except asyncio.QueueEmpty:
                        pass

                try:
                    self._q.put_nowait(packet)
                except asyncio.QueueFull:
                    # If still full, drop this packet silently to avoid blocking
                    pass

        # Save leftover samples (< chunk) for next call
        self._stash = buf[full:]

    async def recv(self):
        """
        Await a pre-chunked packet and wrap it into an av.AudioFrame with proper
        timing. This method runs on aiortc's internal scheduler.
        """
        if self._closed:
            raise asyncio.CancelledError("TTS track closed")

        # Wait for the next 20ms packet
        s16 = await self._q.get()
        self._q.task_done()

        frame = av.AudioFrame(format="s16", layout="mono", samples=self._chunk)
        frame.sample_rate = self._rate
        frame.planes[0].update(s16)
        frame.time_base = Fraction(1, self._rate)
        frame.pts = self._ts
        self._ts += self._chunk

        self._sent += 1
        if self._sent % 50 == 0:
            print(f"[RTC] downlink frames sent: {self._sent}")

        return frame

    def stop(self):
        self._closed = True
        # Drain queue to unblock any awaiters
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except asyncio.QueueEmpty:
            pass
        super().stop()


class WebRTCEndpoint:
    """
    One PeerConnection that:
      - receives browser mic (Opus -> PCM 48k) and calls on_uplink_frame(f32_48k)
      - sends TTS via an internal audio track (encoded to Opus by aiortc)
    """
    def __init__(self, on_uplink_frame=None, logger=print, opus_max_bitrate_bps=None):
        """
        :param on_uplink_frame: async callable taking (f32_mono_48k: np.ndarray)
        :param logger: logging function
        :param opus_max_bitrate_bps: optional int (e.g., 32000) to hint Opus bitrate
        """
        self.pc = RTCPeerConnection(configuration=_make_rtc_config())
        self.on_uplink_frame = on_uplink_frame
        self._logger = logger
        self._resampler_48 = AudioResampler(format="s16", layout="mono", rate=WEBRTC_AUDIO_RATE)
        self._tts_track = _TTSSourceTrack(rate=WEBRTC_AUDIO_RATE)
        self._opus_max_bitrate_bps = opus_max_bitrate_bps

        # Attach downlink track
        sender = self.pc.addTrack(self._tts_track)

        # Optional: tune Opus bitrate
        if self._opus_max_bitrate_bps is not None:
            try:
                params = sender.getParameters()
                for enc in params.encodings:
                    enc.maxBitrate = int(self._opus_max_bitrate_bps)
                asyncio.create_task(sender.setParameters(params))
            except Exception as e:
                self._logger(f"[RTC] setParameters failed: {e}")

        @self.pc.on("track")
        def on_track(track):
            if track.kind != "audio":
                return
            self._logger("[RTC] audio track received")
            asyncio.create_task(self._consume_uplink(track))

        @self.pc.on("connectionstatechange")
        async def on_conn_state():
            self._logger(f"[RTC] connection state: {self.pc.connectionState}")

        @self.pc.on("iceconnectionstatechange")
        async def on_ice_state():
            self._logger(f"[RTC] ICE state: {self.pc.iceConnectionState}")

        @self.pc.on("signalingstatechange")
        async def on_sig_state():
            self._logger(f"[RTC] signaling state: {self.pc.signalingState}")

    async def _consume_uplink(self, track: MediaStreamTrack):
        """
        Convert incoming Opus to f32 mono 48k and call the callback.
        """
        try:
            while True:
                frame = await track.recv()  # av.AudioFrame from the remote mic
                # Let the resampler read original metadata; no manual sample_rate override
                for f in self._resampler_48.resample(frame):
                    arr = f.to_ndarray()  # shape: (samples,) or (channels, samples)
                    if arr.ndim == 2:
                        arr = arr[0]  # mono: take channel 0
                    if arr.dtype != np.int16:
                        arr = arr.astype(np.int16, copy=False)
                    f32 = (arr.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
                    if self.on_uplink_frame is not None:
                        await self.on_uplink_frame(f32)
        except Exception as e:
            self._logger(f"[RTC] uplink ended: {e}")

    async def _wait_ice_complete(self):
        """
        Wait until local ICE gathering is 'complete' so the SDP carries candidates.
        Useful when you don't do trickle ICE.
        """
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)

    async def create_answer(self, offer_sdp: str) -> str:
        """
        Set remote offer, create local answer, and wait for ICE gathering to complete.
        Returns SDP string with embedded ICE candidates (no trickle required).
        """
        await self.pc.setRemoteDescription(RTCSessionDescription(offer_sdp, "offer"))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        await self._wait_ice_complete()  # ensure candidates are in SDP
        # Touch capabilities (optional: keeps parity with your original code)
        RTCRtpSender.getCapabilities("audio")
        return self.pc.localDescription.sdp

    async def add_ice(self, candidate: dict):
        """
        Accept remote trickled ICE candidates (if your client trickles).
        Safe to call even if you rely on non-trickle flow on the server.
        """
        try:
            await self.pc.addIceCandidate(candidate)
        except Exception as e:
            self._logger(f"[RTC] addIceCandidate ignored/failed: {e}")

    async def send_tts_pcm_f32_48k(self, f32: np.ndarray):
        """
        Push float32 mono @48k to downlink track; the track handles chunking/queueing.
        """
        await self._tts_track.push_f32_mono_48k(f32)

    async def close(self):
        try:
            self._tts_track.stop()
        except Exception:
            pass
        await self.pc.close()
