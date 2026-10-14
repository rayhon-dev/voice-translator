import io
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import soundfile as sf

from app.models.tts import text_to_speech
import logging

logger = logging.getLogger(__name__)


router = APIRouter()


class SpeakRequest(BaseModel):
    text: str
    lang: str


@router.post("/speak")
def speak(request: SpeakRequest):
    # DIALOG_PLAN.md 4-bosqich (D3): translate.py'dagi bilan bir xil sabab —
    # text_to_speech() sinxron GPU chaqiruvi, `async def` ichida to'g'ridan-
    # to'g'ri chaqirilsa event loop'ni bloklardi (shu payt Dialog rejimidagi
    # WebSocket ham to'xtab qolardi). Oddiy `def` — Starlette avtomatik
    # threadpool'ga chiqaradi.
    t0 = time.perf_counter()

    audio_array, sample_rate = text_to_speech(request.text, request.lang)

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)

    logger.info(f"[/speak] lang={request.lang} {len(request.text)} chars -> WAV "
        f"in {time.perf_counter() - t0:.2f}s")
    return StreamingResponse(buffer, media_type="audio/wav")
