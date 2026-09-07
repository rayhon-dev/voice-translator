import asyncio
import os
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.stt import transcribe_audio

router = APIRouter()

# How often the processor loop wakes up to look for new audio to transcribe.
PROCESS_INTERVAL = 0.3


@router.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    temp_filename = f"/tmp/{uuid.uuid4()}_live.webm"

    # Shared state between the receiver and processor loops.
    #   latest: the most recent audio chunk received, or None
    #   dirty:  True when `latest` has not been transcribed yet
    state = {"latest": None, "dirty": False}
    disconnected = asyncio.Event()

    async def receiver():
        try:
            while True:
                data = await websocket.receive_bytes()
                state["latest"] = data
                state["dirty"] = True
        except WebSocketDisconnect:
            pass
        finally:
            disconnected.set()

    async def transcribe_latest(final: bool = False):
        state["dirty"] = False
        data = state["latest"]

        with open(temp_filename, "wb") as f:
            f.write(data)

        tag = "final" if final else "live"
        try:
            text = await loop.run_in_executor(None, transcribe_audio, temp_filename)
        except Exception as e:
            print(f"[/ws/transcribe] {tag} pass failed: {e!r}", flush=True)
            return

        try:
            await websocket.send_json({"text": text})
        except Exception:
            pass

        if final:
            print(f"[/ws/transcribe] final transcription: {len(text)} chars", flush=True)

    async def processor():
        while not disconnected.is_set():
            await asyncio.sleep(PROCESS_INTERVAL)
            if state["dirty"] and state["latest"] is not None:
                await transcribe_latest()

        if state["dirty"] and state["latest"] is not None:
            await transcribe_latest(final=True)

    receiver_task = asyncio.create_task(receiver())
    try:
        await processor()
    finally:
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
