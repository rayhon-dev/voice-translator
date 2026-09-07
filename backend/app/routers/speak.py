import io
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import soundfile as sf

from app.models.tts import text_to_speech

router = APIRouter()


class SpeakRequest(BaseModel):
    text: str
    lang: str


@router.post("/speak")
async def speak(request: SpeakRequest):
    req_t0 = time.perf_counter()

    # text_to_speech() is synchronous and GPU-bound; it blocks the event loop
    # for its duration (fine for a single user).
    tts_t0 = time.perf_counter()
    audio_array, sample_rate = text_to_speech(request.text, request.lang)
    tts_s = time.perf_counter() - tts_t0

    enc_t0 = time.perf_counter()
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    wav_bytes = buffer.tell()
    buffer.seek(0)
    enc_s = time.perf_counter() - enc_t0

    total_s = time.perf_counter() - req_t0
    print(
        f"[/speak] lang={request.lang} text_len={len(request.text)} "
        f"text_to_speech={tts_s:.2f}s wav_encode={enc_s:.2f}s "
        f"wav_bytes={wav_bytes} audio_seconds={len(audio_array) / sample_rate:.1f} "
        f"total={total_s:.2f}s",
        flush=True,
    )

    return StreamingResponse(buffer, media_type="audio/wav")
