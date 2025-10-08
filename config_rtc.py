# config_rtc.py
import os

WEBRTC_ENABLED = os.getenv("WEBRTC_ENABLED", "1") == "1"

# STUN/TURN (TURN strongly recommended in prod)
ICE_SERVERS = [
    {"urls": "stun:stun.l.google.com:19302"},
    # Example TURN:
    # {"urls": "turn:your.turn.server:3478", "username": os.getenv("TURN_USER"), "credential": os.getenv("TURN_PASS")}
]

# Opus tuning (WebRTC will encode)
OPUS_BITRATE = 32000      # ~32 kbps
OPUS_PTIME_MS = 20        # 20 ms frames
MONO = True               # speech

# Signaling WS (browser connects here)
SIGNALING_HOST = os.getenv("SIGNALING_HOST", "0.0.0.0")
SIGNALING_PORT = int(os.getenv("SIGNALING_PORT", "8765"))

# Whisper server capture sample rate (your code uses 16k)
ASR_SAMPLE_RATE = 16000

# Downlink playback rate for browser (WebRTC likes 48k)
WEBRTC_AUDIO_RATE = 48000
