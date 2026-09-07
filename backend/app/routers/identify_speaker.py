import os
import time
import uuid

from fastapi import APIRouter, File, UploadFile

from app.models.speaker_id import identify_speaker, reset_speakers

router = APIRouter()


@router.post("/identify-speaker")
async def identify(audio: UploadFile = File(...)):
    t0 = time.perf_counter()
    temp_filename = f"/tmp/{uuid.uuid4()}.webm"

    with open(temp_filename, "wb") as buffer:
        buffer.write(await audio.read())

    try:
        speaker = identify_speaker(temp_filename)
    finally:
        os.remove(temp_filename)

    print(
        f"[/identify-speaker] speaker={speaker!r} ({time.perf_counter() - t0:.2f}s)",
        flush=True,
    )
    return {"speaker": speaker}


@router.post("/reset-dialog")
async def reset_dialog():
    reset_speakers()
    return {"status": "ok"}
