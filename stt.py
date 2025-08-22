# stt.py
import os
import io

# Local offline STT
from faster_whisper import WhisperModel

# Optional cloud STT (if you want)
_USE_OPENAI = False  # set True to use OpenAI API if OPENAI_API_KEY is set

_openai_client = None
try:
    import openai
    if os.getenv("OPENAI_API_KEY"):
        _openai_client = openai.OpenAI()
except Exception:
    pass

_model = None

def init_stt(model_size="base"):
    global _model
    if _USE_OPENAI and _openai_client:
        return
    if _model is None:
        # CPU-friendly: use compute_type="int8"
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    """
    Returns transcribed text ("" if nothing meaningful).
    """
    if _USE_OPENAI and _openai_client:
        # Cloud path
        audio_io = io.BytesIO(wav_bytes)
        audio_io.name = "audio.wav"
        resp = _openai_client.audio.transcriptions.create(
            model="whisper-1", file=audio_io
        )
        return (resp.text or "").strip()

    # Local path
    if _model is None:
        init_stt()
    segments, info = _model.transcribe(io.BytesIO(wav_bytes), vad_filter=False)
    text = " ".join(seg.text for seg in segments).strip()
    return text
