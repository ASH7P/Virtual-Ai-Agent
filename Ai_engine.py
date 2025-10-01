import os
import json
import asyncio
import numpy as np

from openai import AzureOpenAI, AsyncAzureOpenAI
from aiortc import MediaStreamTrack
from av.audio.frame import AudioFrame


# ---------------------------
# WebRTC track for audio
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


# ---------------------------
# Chat + TTS client
# ---------------------------
class LLM_Client:
    def __init__(self, system_prompt: str):
        self.chat_client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-01-01-preview",
        )
        self.tts_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )

        self.chat_deployment = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1-nano")
        self.tts_deployment = os.getenv("AZURE_TTS_DEPLOYMENT", "gpt-4o-mini-tts")

        # Persist conversation history
        self.history = [{"role": "system", "content": system_prompt}]

    def generate_reply(self, user_input: str) -> str:
        # 3. Generate completion (YOUR original logic preserved)
        self.history.append({"role": "user", "content": user_input})

        completion = self.chat_client.chat.completions.create(
            model=self.chat_deployment,
            messages=self.history,
            max_completion_tokens=16384,
            stop=None,
            stream=False,
        )

        # Your JSON parsing logic intact
        try:
            message = json.loads(completion.choices[0].message.content)["message"]
        except Exception:
            message = completion.choices[0].message.content

        self.history.append({"role": "assistant", "content": message})
        return message

    async def stream_tts(self, text: str, track: AzureTTSTrack, instructions: str):
        # 5. Async function for TTS (switched from .stream_to_file → queue for WebRTC)
        async with self.tts_client.audio.speech.with_streaming_response.create(
            model=self.tts_deployment,
            voice="onyx",
            input=text,
            instructions=instructions,
            response_format="pcm",   # PCM is best for RTP/WebRTC
            sample_rate=16000,
        ) as response:
            async for chunk in response.aiter_bytes():
                await track.queue.put(chunk)