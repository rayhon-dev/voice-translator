import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from app.models.speaker_id import identify_speaker, reset_speakers

router = APIRouter()

# Persistent folder (NOT /tmp) so the exact audio sent to /identify-speaker
# survives for manual inspection. <repo>/backend/debug_audio/
DEBUG_AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "debug_audio"
DEBUG_AUDIO_DIR.mkdir(exist_ok=True)


@router.post("/identify-speaker")
async def identify(audio: UploadFile = File(...)):
    req_t0 = time.perf_counter()

    raw = await audio.read()

    # The frontend uploads this as "recording.wav" but the bytes are actually a
    # WebM/Opus container from MediaRecorder — save with the true extension so
    # ffmpeg/librosa pick the right demuxer, and keep it around for inspection.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_path = DEBUG_AUDIO_DIR / f"identify_{ts}.webm"
    debug_path.write_bytes(raw)
    print(
        f"[/identify-speaker] received {len(raw)} bytes "
        f"(upload filename={audio.filename!r}, content_type={audio.content_type!r}) "
        f"-> saved {debug_path}",
        flush=True,
    )

    speaker = identify_speaker(str(debug_path))

    print(
        f"[/identify-speaker] total={time.perf_counter() - req_t0:.2f}s "
        f"speaker={speaker!r} file={debug_path.name}",
        flush=True,
    )
    return {"speaker": speaker}


@router.post("/reset-dialog")
async def reset_dialog():
    reset_speakers()
    return {"status": "ok"}
