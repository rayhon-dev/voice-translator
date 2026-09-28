import asyncio
import json
import os
import uuid

import numpy as np
from fastapi import APIRouter, WebSocket

from app.models.stt import (
    transcribe_audio,
    _decode_waveform_bytes,
    _write_wav,
    SAMPLE_RATE,
)
from app.models.speaker_diart import create_session, get_raw_speaker_label, resolve_speaker
from app.models.translator import translate_text
from app.config import MIN_TURN_SECONDS

router = APIRouter()

# Asosiy shovqin/notanish-til filtri transcribe_audio() ICHIDA (global argmax
# tekshiruvi) amalga oshiriladi. Bu yerdagi chegara — ikkinchi, yengilroq
# himoya qatlami: real testda argmax'dan o'tib ketgan, lekin haqiqatda
# atrofdagi begona ovoz bo'lgan juda qisqa/past-ishonchli navbatlar kuzatildi
# (masalan 6-8 belgili matn, ishonch 0.16-0.35) — ular skriptdagi hech qanday
# gapga mos kelmasdi. 0.03 juda bo'sh bo'lib chiqdi. 0.75 esa (avvalgi holat)
# haqiqiy gaplarni ham rad etardi (real testda aniq mos til uchun ham 0.2-0.5
# chiqishi kuzatilgan). 0.15 — shu ikki haddan o'rtacha, real kuzatuvlarga
# asoslangan muvozanat.
LANGUAGE_MATCH_MIN_PROB = 0.15

# Qisqa matn uchun qo'shimcha, qattiqroq chegara (yuqoridagi izohga qarang).
SHORT_TEXT_MAX_LEN = 12
SHORT_TEXT_MIN_PROB = 0.5

# --- VAQTINCHALIK DIAGNOSTIKA (baland ovozli spikerlarda tasodifiy past-ishonchli
# noto'g'ri til aniqlanishi sababini — clipping/distorsiya ehtimolini — tekshirish
# uchun). Sinov tugagach shu blok va uni ishlatuvchi kod olib tashlansin. ---
DIAG_CLIP_PEAK_THRESHOLD = 0.98  # shundan yuqori peak amplituda — clipping belgisi
DIAG_LOW_CONF_THRESHOLD = 0.5  # shundan past lang_prob — "diagnostika uchun saqlash" chegarasi
DIAG_MAX_SAVED_TURNS = 20  # /tmp'ni cheksiz to'ldirib yubormaslik uchun
DIAG_DIR = "/tmp/dialog_diagnostics"
_diag_saved_count = 0


def _save_diagnostic_turn(wav: np.ndarray, detected_lang, lang_prob: float, peak: float, clipped: bool) -> None:
    global _diag_saved_count
    if _diag_saved_count >= DIAG_MAX_SAVED_TURNS:
        return
    os.makedirs(DIAG_DIR, exist_ok=True)
    _diag_saved_count += 1
    filename = (
        f"{DIAG_DIR}/{uuid.uuid4()}_lang={detected_lang}_prob={lang_prob:.2f}"
        f"_peak={peak:.3f}_clipped={clipped}.wav"
    )
    _write_wav(filename, wav)
    print(f"[/ws/dialog][DIAG] past-ishonchli turn saqlandi ({_diag_saved_count}/{DIAG_MAX_SAVED_TURNS}): {filename}", flush=True)


@router.websocket("/ws/dialog")
async def websocket_dialog(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    state = {
        "phase": "await_lang_a",
        "speaker_langs": {},
        "diart_session": create_session(),
    }
    turn_tasks = set()

    async def send_status(message: str, **extra):
        try:
            await websocket.send_json({"status": message, **extra})
        except Exception:
            pass


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

        # --- VAQTINCHALIK DIAGNOSTIKA: clipping (audio buzilishi) belgisi ---
        peak_amplitude = float(np.abs(wav).max())
        is_clipped = peak_amplitude >= DIAG_CLIP_PEAK_THRESHOLD
        if is_clipped:
            print(
                f"[/ws/dialog][DIAG] CLIPPING shubhasi: peak_amplitude={peak_amplitude:.4f} "
                f"(chegara={DIAG_CLIP_PEAK_THRESHOLD})",
                flush=True,
            )

        temp_filename = f"/tmp/{uuid.uuid4()}_turn.wav"
        await loop.run_in_executor(None, _write_wav, temp_filename, wav)

        try:
            # Avval diart (CPU'da yengil) — ovoz kimga tegishli ekanligini
            # (A/B klasteriga qay darajada o'xshashligini) aniqlaymiz.
            raw_label = await loop.run_in_executor(
                None, get_raw_speaker_label, state["diart_session"], wav, SAMPLE_RATE
            )

            if raw_label is None:
                print("[/ws/dialog] turn skipped (speaker=unknown)", flush=True)
                await send_status("turn_skipped", reason="unknown_speaker")
                return

            # Til HAR DOIM (ilgari bog'langan spikerlar uchun ham) haqiqiy
            # aniqlash orqali (`candidate_languages`) topiladi — hech qachon
            # ko'r-ko'rona majburlanmaydi. Sabab: til majburlab berilganda
            # Whisper aslida aniqlash bosqichini o'tkazib yuboradi va
            # `language_probability`ni har doim soxta ≈1.0 deb qaytaradi —
            # ya'ni ovoz-klasterlash xato qilib qo'ysa (real testda kuzatildi:
            # ingliz nutqi rus deb bog'langan klasterga tasodifan tushib qolgan),
            # tizim buni "100% ishonch bilan" noto'g'ri tilda dekod qilaverar
            # va hatto atrofdagi uchinchi odamning (masalan o'zbekcha) nutqini
            # ham A yoki B tiliga majburan "tarjima qilib" chiqarib yuborar edi.
            candidate_langs = list(dict.fromkeys(state["speaker_langs"].values()))
            try:
                text, detected_lang, lang_prob = await loop.run_in_executor(
                    None, transcribe_audio, temp_filename, None, candidate_langs
                )
            except Exception as e:
                print(f"[/ws/dialog] turn processing failed: {e!r}", flush=True)
                await send_status("turn_skipped", reason="processing_error")
                return
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        # --- VAQTINCHALIK DIAGNOSTIKA: past-ishonchli turnni qo'lda eshitish uchun saqlash ---
        if lang_prob < DIAG_LOW_CONF_THRESHOLD:
            await loop.run_in_executor(
                None, _save_diagnostic_turn, wav, detected_lang, lang_prob, peak_amplitude, is_clipped
            )

        text = (text or "").strip()
        if not text:
            print("[/ws/dialog] turn skipped (empty transcription)", flush=True)
            await send_status("turn_skipped", reason="empty_transcription")
            return

        # Real testda aniqlandi: juda QISQA matn (masalan 5-8 belgi) uchun
        # LANGUAGE_MATCH_MIN_PROB (0.15) yetarlicha qattiq emas — atrofdagi
        # begona ovoz ba'zan shunday qisqa, o'rtacha-past ishonch bilan (masalan
        # 0.16-0.44) "en"/"ru"dan birini tasodifan yutib qolishi mumkin (skriptga
        # mos kelmaydigan matn sifatida chiqib ketgan holatlar kuzatildi).
        # Uzunroq matnda buning ehtimoli ancha kam, chunki argmax ko'proq
        # kontent bilan ishlagan. Shuning uchun qisqa matn uchun qo'shimcha,
        # qattiqroq chegara qo'yiladi.
        if len(text) < SHORT_TEXT_MAX_LEN and lang_prob < SHORT_TEXT_MIN_PROB:
            print(
                f"[/ws/dialog] turn skipped (short_low_confidence: text={text!r} "
                f"len={len(text)} prob={lang_prob:.2f})",
                flush=True,
            )
            await send_status("turn_skipped", reason="short_low_confidence")
            return

        speaker, skip_reason = resolve_speaker(
            state["diart_session"],
            raw_label,
            state["speaker_langs"],
            detected_lang,
            lang_prob,
            min_prob=LANGUAGE_MATCH_MIN_PROB,
        )
        if speaker is None:
            print(f"[/ws/dialog] turn skipped ({skip_reason})", flush=True)
            await send_status("turn_skipped", reason=skip_reason)
            return

        # resolve_speaker() aniqlangan til ishonchi past yoki A/B tillariga mos
        # kelmasa allaqachon None qaytargan (shovqin/notanish spiker sifatida
        # rad etilgan) — shu yerda qo'shimcha tekshirish shart emas.
        speaker_lang = state["speaker_langs"].get(speaker, "en")

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
                if state["phase"] == "listening":
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

        if action == "set_language" and state["phase"] in ("await_lang_a", "await_lang_b"):
            if state["phase"] == "await_lang_a":
                state["speaker_langs"]["A"] = msg.get("lang", "en")
                state["phase"] = "await_lang_b"
                await send_status("awaiting_language", speaker="B")
            else:
                state["speaker_langs"]["B"] = msg.get("lang", "en")
                state["phase"] = "listening"
                await send_status("listening_started")

        elif action == "stop_dialog":
            await websocket.close()


    await send_status("awaiting_language", speaker="A")

    try:
        await receiver()
    except Exception:
        pass
    finally:
        for task in list(turn_tasks):
            task.cancel()
        if turn_tasks:
            await asyncio.gather(*turn_tasks, return_exceptions=True)

        if websocket.client_state.name != "DISCONNECTED":
            try:
                await websocket.close()
            except Exception:
                pass
