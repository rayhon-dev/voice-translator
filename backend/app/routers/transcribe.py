import os
import shutil
import time
import uuid

from fastapi import APIRouter, File, UploadFile

from app.models.stt import transcribe_audio

router = APIRouter()


# NOTE: as of the live-transcription refactor the frontend does NOT call this
# endpoint anymore for the "final transcription" — that now comes from the
# /ws/transcribe WebSocket. This route is kept for the non-live /transcribe
# path and for manual testing. Instrumentation is here anyway so we can A/B it.
@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    req_t0 = time.perf_counter()
    temp_filename = f"/tmp/{uuid.uuid4()}.wav"

    # File I/O: writes the uploaded bytes straight to disk. No audio decoding or
    # format conversion happens here in Python — faster-whisper/ffmpeg decodes
    # the file by content, so the ".wav" name is cosmetic. For a <1 min clip
    # this write is a few MB and should be single-digit milliseconds.
    write_t0 = time.perf_counter()
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    write_s = time.perf_counter() - write_t0
    size_mb = os.path.getsize(temp_filename) / (1024 * 1024)

    try:
        stt_t0 = time.perf_counter()
        # transcribe_audio() is synchronous and GPU/CPU-bound. It is called
        # directly in this async route, so it blocks the event loop for its
        # whole duration (fine for a single user, matters under concurrency).
        text = transcribe_audio(temp_filename)
        stt_s = time.perf_counter() - stt_t0
    finally:
        os.remove(temp_filename)

    total_s = time.perf_counter() - req_t0
    print(
        f"[/transcribe] upload={size_mb:.2f}MB write={write_s * 1000:.0f}ms "
        f"stt={stt_s:.2f}s total={total_s:.2f}s",
        flush=True,
    )

    return {"text": text}
