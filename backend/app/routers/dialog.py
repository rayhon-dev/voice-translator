import asyncio
import json
import os
import uuid

from fastapi import APIRouter, WebSocket

from app.models.stt import (
    transcribe_audio,
    _decode_waveform_bytes,
    _write_wav,
    SAMPLE_RATE,
)
from app.models.speaker_id import enroll_speaker, identify_from_enrolled, reset_speakers
from app.models.translator import translate_text
from app.config import MIN_TURN_SECONDS

router = APIRouter()


@router.websocket("/ws/dialog")
async def websocket_dialog(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    reset_speakers()


    state = {
        "phase": "enroll_a",
        "speaker_langs": {},
        "turn_buffer": bytearray(),
    }
    turn_tasks = set()

    async def send_status(message: str, **extra):
        try:
            await websocket.send_json({"status": message, **extra})
        except Exception:
            pass


    async def finish_enrollment(label: str):
        temp_filename = f"/tmp/{uuid.uuid4()}_enroll_{label}.webm"
        with open(temp_filename, "wb") as f:
            f.write(state["turn_buffer"])

        try:
            success = await loop.run_in_executor(None, enroll_speaker, temp_filename, label)
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        state["turn_buffer"] = bytearray()

        if not success:
            await send_status("enroll_failed", speaker=label)
            return False

        await send_status("enrolled", speaker=label)
        return True


    async def process_turn(data: bytes):
        wav = await loop.run_in_executor(None, _decode_waveform_bytes, data)
        if wav is None or wav.size == 0:
            print("[/ws/dialog] turn skipped (empty audio)", flush=True)
            await send_status("turn_skipped", reason="empty_audio")
            return

        duration_s = wav.size / SAMPLE_RATE
        if duration_s < MIN_TURN_SECONDS:
            print(f"[/ws/dialog] turn skipped (too short: {duration_s:.2f}s)", flush=True)
            await send_status("turn_skipped", reason="too_short")
            return

        temp_filename = f"/tmp/{uuid.uuid4()}_turn.wav"
        await loop.run_in_executor(None, _write_wav, temp_filename, wav)

        try:
            speaker = await loop.run_in_executor(None, identify_from_enrolled, temp_filename)

            if speaker == "unknown":
                print("[/ws/dialog] turn skipped (speaker=unknown)", flush=True)
                await send_status("turn_skipped", reason="unknown_speaker")
                return

            speaker_lang = state["speaker_langs"].get(speaker, "en")
            text = await loop.run_in_executor(
                None, transcribe_audio, temp_filename, speaker_lang
            )
        except Exception as e:
            print(f"[/ws/dialog] turn processing failed: {e!r}", flush=True)
            return
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        text = (text or "").strip()
        if not text:
            print("[/ws/dialog] turn skipped (empty transcription)", flush=True)
            await send_status("turn_skipped", reason="empty_transcription")
            return

        other_speaker = "B" if speaker == "A" else "A"
        target_lang = state["speaker_langs"].get(other_speaker, "en")

        try:
            translation = await loop.run_in_executor(
                None, translate_text, text, target_lang, speaker_lang
            )
        except Exception as e:
            print(f"[/ws/dialog] translation failed: {e!r}", flush=True)
            return

        await send_status(
            "turn",
            speaker=speaker,
            original=text,
            translation=translation,
            source_lang=speaker_lang,
            target_lang=target_lang,
        )


    async def receiver():
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                if state["phase"] in ("enroll_a", "enroll_b"):
                    state["turn_buffer"].extend(data)
                elif state["phase"] == "listening":
                    task = asyncio.create_task(process_turn(data))
                    turn_tasks.add(task)
                    task.add_done_callback(turn_tasks.discard)
                continue

            text = message.get("text")
            if text is not None:
                await handle_client_message(text)


    async def handle_client_message(raw_message):
        try:
            msg = json.loads(raw_message)
        except Exception:
            return

        action = msg.get("action")

        if action == "set_language" and state["phase"] in ("enroll_a", "enroll_b"):
            label = "A" if state["phase"] == "enroll_a" else "B"
            state["speaker_langs"][label] = msg.get("lang", "en")
            await send_status("awaiting_enroll_audio", speaker=label)

        elif action == "finish_enroll":
            if state["phase"] == "enroll_a":
                ok = await finish_enrollment("A")
                if ok:
                    state["phase"] = "enroll_b"
                    await send_status("enroll_prompt", speaker="B")
            elif state["phase"] == "enroll_b":
                ok = await finish_enrollment("B")
                if ok:
                    state["phase"] = "listening"
                    await send_status("listening_started")

        elif action == "stop_dialog":
            await websocket.close()


    await send_status("enroll_prompt", speaker="A")

    try:
        await receiver()
    except Exception:
        pass
    finally:
        for task in list(turn_tasks):
            task.cancel()
        if turn_tasks:
            await asyncio.gather(*turn_tasks, return_exceptions=True)

        reset_speakers()

        if websocket.client_state.name != "DISCONNECTED":
            try:
                await websocket.close()
            except Exception:
                pass
