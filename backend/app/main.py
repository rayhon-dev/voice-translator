import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import speak, identify_speaker, transcribe, translate, live_transcribe
from app.models import stt, translator, speaker_id, tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 64, flush=True)
    print("[warmup] Preloading models before accepting requests...", flush=True)
    overall_t0 = time.perf_counter()

    warmups = (
        ("Whisper (STT)", stt.warmup),
        ("NLLB (translation)", translator.warmup),
        ("Resemblyzer (speaker ID)", speaker_id.warmup),
        ("MMS-TTS (uz/ko/ru)", tts.warmup),
    )
    for name, fn in warmups:
        t0 = time.perf_counter()
        try:
            fn()
            print(
                f"[warmup]   {name}: ready in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )
        except Exception as e:
            print(
                f"[warmup]   {name}: FAILED after {time.perf_counter() - t0:.2f}s "
                f"-> {e!r}",
                flush=True,
            )

    print(
        f"[warmup] All models ready in {time.perf_counter() - overall_t0:.2f}s — "
        f"server is now accepting requests",
        flush=True,
    )
    print("=" * 64, flush=True)

    yield


app = FastAPI(title="Voice Translator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe.router)
app.include_router(translate.router)
app.include_router(speak.router)
app.include_router(identify_speaker.router)
app.include_router(live_transcribe.router)

@app.get("/")
async def root():
    return {"status": "Voice Translator backend is running"}
