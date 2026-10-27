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
    t0 = time.perf_counter()
    translation, mt_provider = translate_text(request.text, request.target_lang, request.source_lang)
    logger.info(f"[/translate] provider={mt_provider} {request.source_lang!r} -> {request.target_lang!r} "
        f"{len(request.text)} chars in {time.perf_counter() - t0:.2f}s")
    return {"translation": translation, "mt_provider": mt_provider}
