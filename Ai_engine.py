import os
import json
import asyncio
import numpy as np

from openai import AzureOpenAI, AsyncAzureOpenAI
from aiortc import MediaStreamTrack
from av.audio.frame import AudioFrame
import sounddevice as sd  # NEW: play PCM to local speakers

#New:
from config_rtc import WEBRTC_AUDIO_RATE
from audio_resample import resample_f32_mono
import numpy as np


# ---------------------------
# WebRTC track placeholder (unchanged, but unused for speaker demo)
# ---------------------------
class AzureTTSTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=16000):
        super().__init__()
        self.queue = asyncio.Queue()
        self.sample_rate = sample_rate

    async def recv(self) -> AudioFrame:
        pcm_bytes = await self.queue.get()
        samples = np.frombuffer(pcm_bytes, np.int16)
        frame = AudioFrame(format="s16", layout="mono", samples=len(samples))
        frame.sample_rate = self.sample_rate
        frame.planes[0].update(samples.tobytes())
        return frame

endpoint = os.getenv("ENDPOINT_URL", "https://ttsmodel3.openai.azure.com/")
# deployment = os.getenv("DEPLOYMENT_NAME", "gpt-5-nano")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "2nnLldWRJFxVegf69V94gzqF2QzlkgYaaISUiCl2bLt6YHRDFRqZJQQJ99BIACHYHv6XJ3w3AAABACOG0eBS")

# ---------------------------
# Chat + TTS client
# ---------------------------
class LLM_Client:
    def __init__(self, system_prompt: str):
        self.chat_client = AzureOpenAI(
            api_key=subscription_key,
            azure_endpoint=endpoint,
            api_version="2025-01-01-preview",
        )
        self.tts_client = AsyncAzureOpenAI(
            api_key=subscription_key,
            azure_endpoint=endpoint,
            api_version="2025-03-01-preview",
        )

        self.chat_deployment = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1-nano")
        self.tts_deployment = os.getenv("AZURE_TTS_DEPLOYMENT", "gpt-4o-mini-tts")

        # Persist conversation history
        self.history = [{"role": "system", "content": system_prompt}]

        self.on_pcm_f32 = None  # set by client.py to push into WebRTC

    async def handle_tts_chunk(self, pcm_bytes_24k_le: bytes):
        # Example if Azure returns 24 kHz s16 mono PCM:
        arr = np.frombuffer(pcm_bytes_24k_le, dtype="<i2").astype(np.float32) / 32768.0
        f32_48k = resample_f32_mono(arr, 24000, WEBRTC_AUDIO_RATE)
        if self.on_pcm_f32:
            await self.on_pcm_f32(f32_48k)

    def generate_reply(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})

        completion = self.chat_client.chat.completions.create(
            model=self.chat_deployment,
            messages=self.history,
            max_completion_tokens=16384,
            stop=None,
            stream=False,
        )

        try:
            message = json.loads(completion.choices[0].message.content.strip())["message"]
        except Exception:
            message = completion.choices[0].message.content

        self.history.append({"role": "assistant", "content": message})
        return message

    async def stream_tts(self, text: str, track: AzureTTSTrack, instructions: str):
        """
        Stream Azure TTS as PCM directly to local speakers.
        Signature unchanged so server.py doesn't need edits.
        The 'track' argument is intentionally ignored for this local demo.
        """
        _ = track  # keep signature; not used for local speaker playback

        # Azure's TTS PCM default samplerate is typically 24000 Hz for gpt-4o-mini-tts
        playback_rate = 24000
        stream = sd.OutputStream(
            samplerate=playback_rate,
            channels=1,
            dtype="int16",
            blocksize=0,
        )
        stream.start()
        try:
            async with self.tts_client.audio.speech.with_streaming_response.create(
                model=self.tts_deployment,
                voice="onyx",
                input=text,
                instructions=instructions,
                response_format="pcm",   # 16-bit PCM; no sample_rate arg here
            ) as response:
                async for chunk in response.iter_bytes():
                    samples = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)
                    stream.write(samples)
        finally:
            stream.stop()
            stream.close()