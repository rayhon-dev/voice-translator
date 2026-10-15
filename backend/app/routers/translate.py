import time

from fastapi import APIRouter
from pydantic import BaseModel
from app.models.translator import translate_text
import logging

logger = logging.getLogger(__name__)


router = APIRouter()

class TranslateRequest(BaseModel):
    text: str
    target_lang: str
    source_lang: str = "en"


@router.post("/translate")
def translate(request: TranslateRequest):
    # DIALOG_PLAN.md 4-bosqich (D3): `async def` ICHIDA sinxron GPU chaqiruvi
    # (translate_text) to'g'ridan-to'g'ri qilinsa, FastAPI'ning YAGONA event
    # loop'i butun chaqiruv davomida BLOKLANIB qoladi — shu payt /ws/dialog
    # WebSocket'i ham audio qabul qila olmaydi/natija yubora olmaydi. Oddiy
    # `def` (async emas) qilib qo'yilsa, Starlette buni avtomatik alohida
    # threadpool'da ishga tushiradi, event loop bo'sh qoladi.
    t0 = time.perf_counter()
    translation, mt_provider = translate_text(request.text, request.target_lang, request.source_lang)
    logger.info(f"[/translate] provider={mt_provider} {request.source_lang!r} -> {request.target_lang!r} "
        f"{len(request.text)} chars in {time.perf_counter() - t0:.2f}s")
    return {"translation": translation, "mt_provider": mt_provider}
