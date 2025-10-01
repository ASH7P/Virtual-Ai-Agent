from whisper_live.client import TranscriptionClient
client = TranscriptionClient("localhost", 9090,
    model="Systran/faster-whisper-large-v3",  # or: "distil-whisper/distil-large-v3"
    lang="ar", translate=False, use_vad=True
)


client()