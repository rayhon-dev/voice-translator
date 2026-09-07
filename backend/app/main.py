import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import speak, identify_speaker, transcribe, translate, live_transcribe
from app.models import stt, translator, speaker_id, tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Uvicorn will not start accepting connections until this startup phase
    # (everything before `yield`) has finished, so every model below is fully
    # in GPU/CPU memory before the first real request arrives.
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
            # A warm-up failure must not stop the server from booting — it will
            # just fall back to lazy-loading on first use (with a visible error).
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
    # No shutdown work needed.


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
