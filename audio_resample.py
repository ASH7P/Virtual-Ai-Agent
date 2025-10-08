# audio_resample.py
import numpy as np
import av
from av.audio.resampler import AudioResampler

def int16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

def float32_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype("<i2")

def resample_pcm_s16_mono(pcm_s16: bytes, in_hz: int, out_hz: int) -> bytes:
    if in_hz == out_hz:
        return pcm_s16
    arr = np.frombuffer(pcm_s16, dtype="<i2")
    frame_in = av.AudioFrame(format="s16", layout="mono", samples=len(arr))
    frame_in.planes[0].update(pcm_s16)
    frame_in.sample_rate = in_hz
    resampler = AudioResampler(format="s16", layout="mono", rate=out_hz)
    out_bytes = bytearray()
    for f in resampler.resample(frame_in):
        out_bytes += f.planes[0].to_bytes()
    return bytes(out_bytes)

def resample_f32_mono(pcm_f32: np.ndarray, in_hz: int, out_hz: int) -> np.ndarray:
    if in_hz == out_hz:
        return pcm_f32
    # via int16 for simplicity/quality (PyAV SRC)
    s16 = float32_to_int16(pcm_f32).tobytes()
    s16_res = resample_pcm_s16_mono(s16, in_hz, out_hz)
    return int16_to_float32(np.frombuffer(s16_res, dtype="<i2"))
