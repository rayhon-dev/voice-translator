import asyncio
import json
import os
import time
import uuid

import numpy as np
from fastapi import APIRouter, WebSocket

from app.models.stt import (
    transcribe_audio,
    _decode_pcm_bytes,
    _write_wav,
    compute_speech_rms,
    SAMPLE_RATE,
)
from app.models.speaker_diart import (
    create_session,
    identify_speaker,
    apply_speaker_correction,
    remove_turn_prototype,
    merge_speakers_manual,
    has_known_speakers,
    embed_for_match,
    quick_speaker_match,
    save_turn_label,
    _save_debug_audio_async,
)
from app.models.translator import translate_text
from app.models.segmenter import segment_turn
from app.config import MIN_TURN_SECONDS, NLLB_LANG_CODES
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

SESSION_LANGS = ["en", "ru"]

LANGUAGE_MATCH_MIN_PROB = 0.15

MERGE_GAP_S = 0.3
MERGE_CONTINUATION_MAX_DURATION_S = 3.0

RETRO_MERGE_WINDOW_S = 2.5

# process_turn ichidagi _merge_same_known_speaker.
SEG_SAME_SPEAKER_MIN_SIM = float(os.environ.get("SEG_SAME_SPEAKER_MIN_SIM", "0.35"))

MAX_MERGED_CHARS = int(os.environ.get("MAX_MERGED_CHARS", "300"))
MAX_MERGED_AUDIO_S = float(os.environ.get("MAX_MERGED_AUDIO_S", "25"))

FAST_PATH_MAX_DURATION_S = 2.0
FAST_PATH_MAX_RMS = 0.01

MAX_TURN_HISTORY = 50

NEAR_REF_MIN_TURNS = int(os.environ.get("NEAR_REF_MIN_TURNS", "2"))
NEAR_REF_MAX_SAMPLES = 20
FAR_FIELD_TURN_RATIO = float(os.environ.get("FAR_FIELD_TURN_RATIO", "0.35"))


@router.websocket("/ws/dialog")
async def websocket_dialog(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    state = {
        "phase": "listening",
        "speaker_session": create_session(),
        "turn_seq": 0,
        "pending": None,
        "turn_history": {},
        "last_turn_by_speaker": {},
        "last_chunk": None,
    }
    turn_queue: asyncio.Queue = asyncio.Queue()

    async def send_status(message: str, **extra):
        try:
            await websocket.send_json({"status": message, **extra})
        except Exception:
            pass

    def _trim_turn_history():
        history = state["turn_history"]
        while len(history) > MAX_TURN_HISTORY:
            del history[min(history)]

    async def _resend_turn(seq: int, entry: dict) -> None:
        await send_status(
            "turn",
            seq=seq,
            speaker=entry["speaker"],
            original=entry["text"],
            translation=entry["translation"],
            source_lang=entry["source_lang"],
            target_lang=entry["target_lang"],
            uncertain=entry.get("uncertain", False),
            mt_provider=entry.get("mt_provider"),
            translation_error=entry.get("mt_provider") == "error",
            client_seqs=entry.get("client_seqs", []),
        )

    async def _apply_merges_and_notify(merges: list[tuple[int, int, float]]) -> None:
        for frm, into, sim in merges:
            for entry in state["turn_history"].values():
                if entry["speaker"] == frm:
                    entry["speaker"] = into
            frm_last = state["last_turn_by_speaker"].pop(frm, None)
            if frm_last is not None:
                into_last = state["last_turn_by_speaker"].get(into)
                if into_last is None or frm_last["sent_at"] > into_last["sent_at"]:
                    state["last_turn_by_speaker"][into] = frm_last
            await send_status("speakers_merged", **{"from": frm, "into": into})

    async def _finish_translation(out_seq, out_text, source_lang, target_lang):
        t_mt0 = time.perf_counter()
        try:
            translation, mt_provider = await loop.run_in_executor(
                None, translate_text, out_text, target_lang, source_lang
            )
        except Exception as e:
            logger.error(f"[/ws/dialog] translation failed: {e!r}")
            translation = ""
            mt_provider = "error"
        mt_elapsed = time.perf_counter() - t_mt0

        entry = state["turn_history"].get(out_seq)
        if entry is None or entry.get("text") != out_text:
            logger.info(
                f"[/ws/dialog] translation for seq={out_seq} discarded (eskirgan — "
                f"matn shu orada o'zgargan/o'chirilgan) mt={mt_elapsed:.2f}s"
            )
            return
        entry["translation"] = translation
        entry["mt_provider"] = mt_provider
        logger.info(f"[TURN][MT] seq={out_seq} mt={mt_elapsed:.2f}s provider={mt_provider!r}")
        await _resend_turn(out_seq, entry)

    async def _process_subsegment(
        wav, seq, client_seqs: list, is_multi: bool, purity: float,
        sub_index: int = 0, sub_total: int = 1, continuation_lang: str | None = None,
    ):
        client_seqs = client_seqs or []

        t_turn_start = time.perf_counter()
        stt_elapsed = 0.0

        duration_s = wav.size / SAMPLE_RATE
        if duration_s < MIN_TURN_SECONDS:
            logger.info(f"[/ws/dialog] turn skipped (too short: {duration_s:.2f}s)")
            await send_status("turn_skipped", reason="too_short", client_seqs=client_seqs)
            return

        if duration_s < FAST_PATH_MAX_DURATION_S and has_known_speakers(state["speaker_session"]):
            rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
            if rms < FAST_PATH_MAX_RMS:
                logger.info(f"[/ws/dialog] turn skipped (fast_path_noise: duration={duration_s:.2f}s "
                    f"rms={rms:.4f} < {FAST_PATH_MAX_RMS}, kamida bitta spiker allaqachon ma'lum) — "
                    f"STT/embedding chaqirilmadi")
                await send_status("turn_skipped", reason="unknown_speaker", client_seqs=client_seqs)
                return

        temp_filename = f"/tmp/{uuid.uuid4()}_turn.wav"
        await loop.run_in_executor(None, _write_wav, temp_filename, wav)

        t_stt0 = time.perf_counter()
        try:
            try:
                text, detected_lang, lang_prob, unsupported_language, stt_diag = await loop.run_in_executor(
                    None, transcribe_audio, temp_filename, None, SESSION_LANGS, continuation_lang
                )
            except Exception as e:
                logger.error(f"[/ws/dialog] turn processing failed: {e!r}")
                await send_status("turn_skipped", reason="processing_error", client_seqs=client_seqs)
                return
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        stt_elapsed = time.perf_counter() - t_stt0

        if unsupported_language:
            reason = "foreign_subsegment" if is_multi else "unsupported_language"
            # Diagnostika: til bo'yicha rad etilgan audio ham tinglash uchun saqlanadi.
            rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
            _save_debug_audio_async(wav, SAMPLE_RATE, duration_s, rms, f"rejected_lang_seq{seq}")
            logger.info(f"[/ws/dialog] turn skipped (reason={reason}: session_langs={SESSION_LANGS})")
            await send_status("turn_skipped", reason=reason, client_seqs=client_seqs)
            return

        text = (text or "").strip()
        if not text:
            logger.info("[/ws/dialog] turn skipped (empty transcription)")
            await send_status("turn_skipped", reason="empty_transcription", client_seqs=client_seqs)
            return
        if state.get("last_chunk") is not None:
            state["last_chunk"]["lang"] = detected_lang

        debug_label: dict = {}
        spk_result = await loop.run_in_executor(
            None,
            identify_speaker,
            state["speaker_session"],
            wav,
            SAMPLE_RATE,
            seq,
            detected_lang,
            debug_label,
            purity,
        )
        speaker = spk_result["speaker"]
        uncertain = spk_result["uncertain"]

        await _apply_merges_and_notify(spk_result.get("merges", []))

        for r_seq, r_speaker in spk_result.get("reassigned", []):
            r_entry = state["turn_history"].get(r_seq)
            if r_entry is None:
                continue
            r_entry["speaker"] = r_speaker
            r_entry["uncertain"] = False
            state["last_turn_by_speaker"][r_speaker] = {
                "seq": r_seq,
                "lang": r_entry["source_lang"],
                "sent_at": time.monotonic(),
            }
            logger.info(f"[/ws/dialog] unknown buffer reassigned seq={r_seq} -> speaker={r_speaker}")
            await _resend_turn(r_seq, r_entry)

        save_turn_label(debug_label.get("wav_filename"), speaker=speaker, lang=detected_lang, text=text)

        source_lang = detected_lang
        target_lang = "ru" if source_lang == "en" else "en"

        merge_seq = None
        merge_prev_entry = None
        last_turn = state["last_turn_by_speaker"].get(speaker) if speaker is not None else None
        if (
            last_turn is not None
            and last_turn["lang"] == source_lang
            and (time.monotonic() - last_turn["sent_at"]) <= RETRO_MERGE_WINDOW_S
            and not uncertain
        ):
            candidate_entry = state["turn_history"].get(last_turn["seq"])
            if (
                candidate_entry is not None
                and not candidate_entry.get("uncertain", False)
                and len(candidate_entry["text"]) < MAX_MERGED_CHARS
                and (candidate_entry["wav"].size / SAMPLE_RATE) < MAX_MERGED_AUDIO_S
            ):
                merge_seq = last_turn["seq"]
                merge_prev_entry = candidate_entry

        if merge_prev_entry is not None:
            out_seq = merge_seq
            out_text = f"{merge_prev_entry['text']} {text}".strip()
            out_wav = np.concatenate([merge_prev_entry["wav"], wav])
            merged_seqs = list(merge_prev_entry.get("merged_seqs", [out_seq]))
            merged_seqs.append(seq)
            out_client_seqs = list(merge_prev_entry.get("client_seqs", [])) + client_seqs
            logger.info(
                f"[/ws/dialog] retro-merging seq={seq} into seq={out_seq} "
                f"(speaker={speaker} lang={source_lang})"
            )
        else:
            out_seq = seq
            out_text = text
            out_wav = wav
            merged_seqs = [out_seq]
            out_client_seqs = client_seqs

        mt_provider = None
        translation_pending = False
        if source_lang == target_lang:
            translation = out_text
        elif (
            source_lang not in NLLB_LANG_CODES
            or target_lang not in NLLB_LANG_CODES
            or lang_prob < LANGUAGE_MATCH_MIN_PROB
        ):
            logger.info(f"[/ws/dialog] translation skipped (source_lang={source_lang!r} "
                f"target_lang={target_lang!r} lang_prob={lang_prob:.2f}) — showing original text")
            translation = out_text
        else:
            translation = None
            translation_pending = True

        state["turn_history"][out_seq] = {
            "wav": out_wav,
            "speaker": speaker,
            "text": out_text,
            "translation": translation,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "uncertain": uncertain,
            # correct_speaker/delete_turn uchun — qarang yuqoridagi izoh.
            "merged_seqs": merged_seqs,
            "mt_provider": mt_provider,
            "client_seqs": out_client_seqs,
        }
        _trim_turn_history()

        if speaker is not None:
            state["last_turn_by_speaker"][speaker] = {
                "seq": out_seq,
                "lang": source_lang,
                "sent_at": time.monotonic(),
            }

        logger.info(
            f"[TURN] seq={out_seq} sub={sub_index + 1}/{sub_total} audio={duration_s:.2f}s "
            f"stt={stt_elapsed:.2f}s mt={'pending' if translation_pending else '0.00'}s "
            f"total={time.perf_counter() - t_turn_start:.2f}s "
            f"merged={merge_prev_entry is not None} speaker={speaker} uncertain={uncertain} "
            f"decision={spk_result['decision']} purity={purity:.3f} "
            f"seg_lang_check={stt_diag.get('seg_lang_check', False)}"
            f"({stt_diag.get('seg_lang_check_ms', 0.0):.1f}ms)"
        )

        if translation_pending:
            asyncio.create_task(_finish_translation(out_seq, out_text, source_lang, target_lang))
        else:
            await _resend_turn(out_seq, state["turn_history"][out_seq])

    def _merge_same_known_speaker(subsegments: list[dict]) -> list[dict]:
        matches = []
        for sub in subsegments:
            emb, _, _ = embed_for_match(sub["wav"], SAMPLE_RATE)
            best_id, best_sim, _ = quick_speaker_match(state["speaker_session"], emb)
            matches.append(best_id if best_sim >= SEG_SAME_SPEAKER_MIN_SIM else None)
        merged = [dict(subsegments[0])]
        merged_matches = [matches[0]]
        for sub, match in zip(subsegments[1:], matches[1:]):
            if match is not None and match == merged_matches[-1]:
                last = merged[-1]
                last["wav"] = np.concatenate([last["wav"], sub["wav"]])
                last["end"] = sub["end"]
                last["purity"] = min(last["purity"], sub["purity"])
            else:
                merged.append(dict(sub))
                merged_matches.append(match)
        if len(merged) < len(subsegments):
            logger.info(
                f"[/ws/dialog] {len(subsegments)} qism bir xil tanilgan spikerga mos — "
                f"{len(merged)} qismga qayta birlashtirildi (matches={matches})"
            )
        return merged

    async def process_turn(
        wav, seq: int, client_seqs: list | None = None,
        reason: str | None = None, arrived_at: float | None = None,
    ):
        client_seqs = client_seqs or []
        arrived_at = time.monotonic() if arrived_at is None else arrived_at

        continuation_lang = None
        last_chunk = state.get("last_chunk")
        if last_chunk is not None and last_chunk["cut"] and last_chunk["lang"] is not None:
            gap_s = arrived_at - wav.size / SAMPLE_RATE - last_chunk["arrived_at"]
            if gap_s <= RETRO_MERGE_WINDOW_S:
                continuation_lang = last_chunk["lang"]
        state["last_chunk"] = {"cut": reason != "silence", "lang": None, "arrived_at": arrived_at}

        t_seg0 = time.perf_counter()
        subsegments = await loop.run_in_executor(None, segment_turn, wav, SAMPLE_RATE, seq)
        seg_elapsed = time.perf_counter() - t_seg0
        if len(subsegments) > 1 and has_known_speakers(state["speaker_session"]):
            subsegments = await loop.run_in_executor(None, _merge_same_known_speaker, subsegments)
        is_multi = len(subsegments) > 1
        if is_multi:
            logger.info(f"[/ws/dialog] seq={seq} {len(subsegments)} qismga bo'lindi (seg took={seg_elapsed * 1000:.1f}ms)")

        for i, sub in enumerate(subsegments):
            sub_seq = seq if i == 0 else seq + i * 0.001
            sub_client_seqs = client_seqs if i == 0 else []
            await _process_subsegment(
                sub["wav"], sub_seq, sub_client_seqs, is_multi, sub["purity"],
                sub_index=i, sub_total=len(subsegments),
                continuation_lang=continuation_lang if i == 0 else None,
            )

    async def turn_worker():
        while True:
            item = await turn_queue.get()
            try:
                if item is None:
                    break
                wav, seq, client_seqs, reason, arrived_at = item
                await send_status("processing")
                await process_turn(wav, seq, client_seqs, reason, arrived_at)
            except Exception as e:
                logger.error(f"[/ws/dialog] turn_worker: process_turn raised unexpectedly: {e!r}")
            finally:
                turn_queue.task_done()

    worker_task = asyncio.create_task(turn_worker())

    def dispatch_pending_now(waited_s: float = 0.0):
        pending = state["pending"]
        state["pending"] = None
        logger.info(
            f"[MERGE] seq={pending['seq']} reason={pending.get('reason')} "
            f"waited={waited_s:.2f}s merged={pending.get('merge_count', 0) > 0}"
        )
        turn_queue.put_nowait((
            pending["wav"], pending["seq"], pending.get("client_seqs", []),
            pending.get("reason"), time.monotonic(),
        ))

    async def dispatch_after_delay(seq):
        try:
            await asyncio.sleep(MERGE_GAP_S)
        except asyncio.CancelledError:
            return
        pending = state["pending"]
        if pending is None or pending["seq"] != seq:
            return
        dispatch_pending_now(waited_s=MERGE_GAP_S)

    async def handle_audio_chunk(data: bytes, reason: str | None, client_seq=None):
        client_seqs_for_chunk = [client_seq] if client_seq is not None else []

        wav = _decode_pcm_bytes(data)
        if wav is None or wav.size == 0:
            logger.info("[/ws/dialog] turn skipped (empty audio)")
            await send_status("turn_skipped", reason="empty_audio", client_seqs=client_seqs_for_chunk)
            return

        duration_s = wav.size / SAMPLE_RATE
        pending = state["pending"]

        if pending is not None and duration_s < MERGE_CONTINUATION_MAX_DURATION_S:
            if pending.get("dispatch_task") is not None:
                pending["dispatch_task"].cancel()
            merged_wav = np.concatenate([pending["wav"], wav])
            seq = pending["seq"]
            merge_count = pending.get("merge_count", 0) + 1
            client_seqs = pending.get("client_seqs", []) + client_seqs_for_chunk
            logger.info(f"[/ws/dialog] merged short continuation (dur={duration_s:.2f}s) "
                f"into pending turn seq={seq} (new total dur="
                f"{merged_wav.size / SAMPLE_RATE:.2f}s)")
        else:
            merged_wav = wav
            client_seqs = client_seqs_for_chunk
            state["turn_seq"] += 1
            seq = state["turn_seq"]
            merge_count = 0

        if reason == "silence":
            state["pending"] = {
                "wav": merged_wav,
                "seq": seq,
                "dispatch_task": None,
                "reason": reason,
                "merge_count": merge_count,
                "client_seqs": client_seqs,
            }
            dispatch_pending_now(waited_s=0.0)
        else:
            dispatch_task = asyncio.create_task(dispatch_after_delay(seq))
            state["pending"] = {
                "wav": merged_wav,
                "seq": seq,
                "dispatch_task": dispatch_task,
                "reason": reason,
                "merge_count": merge_count,
                "client_seqs": client_seqs,
            }

    async def receiver():
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                if state["phase"] == "listening":
                    reason = state.pop("pending_turn_reason", None)
                    client_seq = state.pop("pending_turn_client_seq", None)
                    await handle_audio_chunk(data, reason, client_seq)
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

        if action == "turn_meta":
            state["pending_turn_reason"] = msg.get("reason")
            state["pending_turn_client_seq"] = msg.get("client_seq")

        elif action == "stop_dialog":
            await websocket.close()

        elif action == "correct_speaker" and state["phase"] == "listening":
            seq = msg.get("seq")
            target_speaker = msg.get("target_speaker")
            entry = state["turn_history"].get(seq)
            if entry is None:
                logger.info(f"[/ws/dialog] correct_speaker: seq={seq} topilmadi (eskirgan?)")
                return
            if target_speaker is None:
                logger.info(f"[/ws/dialog] correct_speaker: target_speaker berilmagan (seq={seq})")
                return

            old_speaker = entry["speaker"]
            if target_speaker == old_speaker:
                logger.info(f"[/ws/dialog] correct_speaker: seq={seq} target == joriy spiker ({old_speaker}) — no-op")
                return

            merged_seqs = entry.get("merged_seqs", [seq])
            result = await loop.run_in_executor(
                None,
                apply_speaker_correction,
                state["speaker_session"],
                entry["wav"],
                SAMPLE_RATE,
                merged_seqs,
                old_speaker,
                target_speaker,
            )
            new_speaker = result["speaker"]
            removed_speaker = result.get("removed_speaker")

            entry["speaker"] = new_speaker
            entry["uncertain"] = False
            if state["last_turn_by_speaker"].get(old_speaker, {}).get("seq") == seq:
                del state["last_turn_by_speaker"][old_speaker]
            state["last_turn_by_speaker"][new_speaker] = {
                "seq": seq,
                "lang": entry["source_lang"],
                "sent_at": time.monotonic(),
            }
            logger.info(f"[/ws/dialog] correct_speaker: seq={seq} {old_speaker} -> {new_speaker}")

            await _apply_merges_and_notify(result.get("merges", []))
            if removed_speaker is not None:
                state["last_turn_by_speaker"].pop(removed_speaker, None)
                await send_status("speaker_removed", speaker=removed_speaker)
            await _resend_turn(seq, entry)

        elif action == "delete_turn" and state["phase"] == "listening":
            seq = msg.get("seq")
            entry = state["turn_history"].pop(seq, None)
            if entry is None:
                logger.info(f"[/ws/dialog] delete_turn: seq={seq} topilmadi (eskirgan?)")
                return

            deleted_speaker = entry["speaker"]
            merged_seqs = entry.get("merged_seqs", [seq])
            result = await loop.run_in_executor(
                None, remove_turn_prototype, state["speaker_session"], deleted_speaker, merged_seqs
            )
            removed_speaker = result.get("removed_speaker")

            if state["last_turn_by_speaker"].get(deleted_speaker, {}).get("seq") == seq:
                del state["last_turn_by_speaker"][deleted_speaker]

            logger.info(
                f"[/ws/dialog] delete_turn: seq={seq} speaker={deleted_speaker} "
                f"speaker_removed={removed_speaker is not None}"
            )

            await _apply_merges_and_notify(result.get("merges", []))
            if removed_speaker is not None:
                state["last_turn_by_speaker"].pop(removed_speaker, None)
                await send_status("speaker_removed", speaker=removed_speaker)
            await send_status("turn_deleted", seq=seq)

        elif action == "rename_speaker":
            speaker = msg.get("speaker")
            name = msg.get("name")
            logger.info(f"[/ws/dialog] rename_speaker: speaker={speaker} name={name!r}")

        elif action == "merge_speakers" and state["phase"] == "listening":
            from_speaker = msg.get("from")
            into_speaker = msg.get("into")
            if from_speaker is None or into_speaker is None or from_speaker == into_speaker:
                logger.info(
                    f"[/ws/dialog] merge_speakers: noto'g'ri so'rov from={from_speaker} into={into_speaker}"
                )
                return

            result = await loop.run_in_executor(
                None, merge_speakers_manual, state["speaker_session"], from_speaker, into_speaker
            )
            if result.get("merged"):
                await _apply_merges_and_notify(result.get("merges", []))
            else:
                logger.info(
                    f"[/ws/dialog] merge_speakers: no-op (from={from_speaker} into={into_speaker} topilmadi?)"
                )

    await send_status("listening_started")

    try:
        await receiver()
    except Exception as e:
        logger.error(f"[/ws/dialog] receiver loop crashed: {e!r}")
    finally:
        if state["pending"] is not None:
            if state["pending"].get("dispatch_task") is not None:
                state["pending"]["dispatch_task"].cancel()
            dispatch_pending_now(waited_s=0.0)

        turn_queue.put_nowait(None)
        await worker_task

        if websocket.client_state.name != "DISCONNECTED":
            try:
                await websocket.close()
            except Exception:
                pass
