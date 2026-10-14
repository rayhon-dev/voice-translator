import time
from contextlib import asynccontextmanager
import logging

# DIALOG_PLAN.md 4-bosqich: `print` o'rniga logging. Root logger'ni ANIQ
# shu yerda, boshqa app.* modullar import qilinishidan OLDIN sozlash kerak —
# aks holda ularning logger.info() chaqiruvlari uvicorn hech qanday
# handler/level o'rnatmagan root logger orqali JIMGINA yo'qolib ketardi
# (standart root level — WARNING).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import speak, translate, live_transcribe, dialog
from app.models import stt, translator, speaker_diart, tts

logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 64)
    logger.info("[warmup] Preloading models before accepting requests...")
    overall_t0 = time.perf_counter()

    warmups = (
        ("Whisper (STT)", stt.warmup),
        ("NLLB (translation)", translator.warmup),
        ("diart (speaker ID)", speaker_diart.warmup),
        ("MMS-TTS (uz/ko/ru/en)", tts.warmup),
    )
    for name, fn in warmups:
        t0 = time.perf_counter()
        try:
            fn()
            logger.info(f"[warmup]   {name}: ready in {time.perf_counter() - t0:.2f}s")
        except Exception as e:
            logger.error(f"[warmup]   {name}: FAILED after {time.perf_counter() - t0:.2f}s "
                f"-> {e!r}")

    logger.info(f"[warmup] All models ready in {time.perf_counter() - overall_t0:.2f}s — "
        f"server is now accepting requests")
    logger.info("=" * 64)

    yield


app = FastAPI(title="Voice Translator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(translate.router)
app.include_router(speak.router)
app.include_router(live_transcribe.router)
app.include_router(dialog.router)

@app.get("/")
async def root():
    return {"status": "Voice Translator backend is running"}
