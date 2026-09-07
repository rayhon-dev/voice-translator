import asyncio
import os
import time
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
        """Continuously drain the socket, keeping ONLY the newest chunk.

        Older unprocessed chunks are overwritten rather than queued, so a
        slow transcription can never build up a backlog.
        """
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
        """Transcribe the current latest chunk without blocking the loop."""
        # Clear the flag first so audio arriving mid-transcription re-arms it.
        state["dirty"] = False
        data = state["latest"]

        with open(temp_filename, "wb") as f:
            f.write(data)
        size_mb = len(data) / (1024 * 1024)

        tag = "FINAL pass" if final else "live pass"
        t0 = time.perf_counter()
        try:
            # transcribe_audio() is synchronous and CPU/GPU-bound; run it in a
            # worker thread so the event loop stays free to receive audio.
            text = await loop.run_in_executor(None, transcribe_audio, temp_filename)
        except Exception as e:
            print(
                f"[/ws/transcribe] {tag} FAILED after "
                f"{time.perf_counter() - t0:.2f}s ({size_mb:.2f}MB): {e!r}",
                flush=True,
            )
            return
        elapsed = time.perf_counter() - t0

        sent = False
        try:
            await websocket.send_json({"text": text})
            sent = True
        except Exception:
            pass

        print(
            f"[/ws/transcribe] {tag}: transcribe={elapsed:.2f}s blob={size_mb:.2f}MB "
            f"chars={len(text)} sent_to_client={sent}",
            flush=True,
        )

    async def processor():
        """Transcribe only the most recent audio state, one job at a time.

        Because each transcription is awaited before the next check, a new
        job never starts while the previous one is still running. Any audio
        that arrived in the meantime is picked up on the following pass, so
        stale intermediate chunks are simply skipped.
        """
        while not disconnected.is_set():
            await asyncio.sleep(PROCESS_INTERVAL)
            if state["dirty"] and state["latest"] is not None:
                await transcribe_latest()

        # Final pass: make sure the last blob sent right before the client
        # closed the socket still gets transcribed. NOTE: the client closes the
        # socket ~1.2s after sending this blob, so if this pass takes longer,
        # send_to_client will be False and the client keeps the last live text.
        if state["dirty"] and state["latest"] is not None:
            print("[/ws/transcribe] client disconnected — running FINAL pass", flush=True)
            await transcribe_latest(final=True)
        else:
            print(
                "[/ws/transcribe] client disconnected — no un-transcribed final blob",
                flush=True,
            )

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
