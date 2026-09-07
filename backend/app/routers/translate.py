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
    t0 = time.perf_counter()
    translation = translate_text(request.text, request.target_lang)
    print(
        f"[/translate] target={request.target_lang!r} "
        f"{len(request.text)} chars in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    return {"translation": translation}
