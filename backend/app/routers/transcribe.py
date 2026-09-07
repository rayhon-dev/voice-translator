import os
import shutil
import time
import uuid

from fastapi import APIRouter, File, UploadFile

from app.models.stt import transcribe_audio

router = APIRouter()


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    t0 = time.perf_counter()
    temp_filename = f"/tmp/{uuid.uuid4()}.wav"

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        text = transcribe_audio(temp_filename)
    finally:
        os.remove(temp_filename)

    print(f"[/transcribe] {len(text)} chars in {time.perf_counter() - t0:.2f}s", flush=True)
    return {"text": text}
