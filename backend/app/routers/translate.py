import time

from fastapi import APIRouter
from pydantic import BaseModel
from app.models.translator import translate_text

router = APIRouter()

class TranslateRequest(BaseModel):
    text: str
    target_lang: str


@router.post("/translate")
async def translate(request: TranslateRequest):
    # translate_text() is synchronous and GPU-bound; called directly in this
    # async route it blocks the event loop for its whole duration.
    t0 = time.perf_counter()
    translation = translate_text(request.text, request.target_lang)
    print(
        f"[/translate] total={time.perf_counter() - t0:.2f}s "
        f"target={request.target_lang!r} in_chars={len(request.text)}",
        flush=True,
    )
    return {"translation": translation}
