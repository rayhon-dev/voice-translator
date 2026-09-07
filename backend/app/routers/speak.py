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
    t0 = time.perf_counter()

    audio_array, sample_rate = text_to_speech(request.text, request.lang)

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)

    print(
        f"[/speak] lang={request.lang} {len(request.text)} chars -> WAV "
        f"in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    return StreamingResponse(buffer, media_type="audio/wav")
