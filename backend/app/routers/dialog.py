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
from app.models.speaker_diart import create_session, get_raw_speaker_label, resolve_speaker
from app.models.translator import translate_text
from app.config import MIN_TURN_SECONDS, NLLB_LANG_CODES, SUPPORTED_LANGUAGES

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

# Shundan PASTROQ lang_prob bilan chiqqan matn ko'pincha Whisper'ning
# gallyutsinatsiyasi (haqiqiy nutqqa aloqasi yo'q, mantiqsiz/tasodifiy
# chiqish) bo'lib chiqadi — bunday holatda ASL matnni ham ko'rsatish
# noto'g'ri, chunki u o'zi ham ishonchli emas (LANGUAGE_MATCH_MIN_PROB'dan
# farqli — o'sha faqat TARJIMANI o'tkazib yuboradi, asl matn baribir
# ko'rsatilaveradi). Shuning uchun bu chegaradan past bo'lsa, turn
# UMUMAN ko'rsatilmaydi.
VERY_LOW_CONFIDENCE_LANG_PROB = 0.3


@router.websocket("/ws/dialog")
async def websocket_dialog(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    state = {
        "phase": "await_lang_a",
        "speaker_langs": {},
        "diart_session": create_session(),
        "turn_seq": 0,
    }
    turn_tasks = set()

    async def send_status(message: str, **extra):
        try:
            await websocket.send_json({"status": message, **extra})
        except Exception:
            pass


    async def process_turn(data: bytes, seq: int):
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
            # Avval diart (CPU'da yengil) — ovoz kimga tegishli ekanligini
            # (A/B klasteriga qay darajada o'xshashligini) aniqlaymiz.
            raw_label, sim_debug = await loop.run_in_executor(
                None, get_raw_speaker_label, state["diart_session"], wav, SAMPLE_RATE
            )

            if raw_label is None:
                sim_str = " ".join(f"vs {name}={sim:.3f}" for name, sim in sim_debug.items())
                print(
                    f"[/ws/dialog] turn skipped (speaker=unknown) [SPK][SIM] {sim_str} "
                    f"(til aniqlanmadi — transkripsiya o'tkazib yuborildi)",
                    flush=True,
                )
                await send_status("turn_skipped", reason="unknown_speaker")
                return

            # Til A/B'ning OLDINDAN TANLAGAN juftligiga (ya'ni shu suhbatdagi
            # ikkita tilga) CHEKLANMAYDI — Whisper'ning aniqlashi butun ilova
            # bo'ylab ruxsat etilgan SUPPORTED_LANGUAGES (en/ru/ko) ro'yxati
            # bilan cheklanadi, A/B tanlovidan qat'i nazar. Sabab ikkita
            # qatlamli:
            #  1) spiker KIM ekanligi butunlay ovoz orqali
            #     (`get_raw_speaker_label`/`resolve_speaker`) hal qilinadi —
            #     til bu yerda FAQAT tarjima YO'NALISHI uchun kerak, shuning
            #     uchun uni A/B tillari bilan cheklashning printsipial hojati
            #     yo'q (masalan odam ikkinchi tilga o'tib qolsa ham to'g'ri
            #     aniqlansin).
            #  2) SHUNGA QARAMAY endi 3 tilli SUPPORTED_LANGUAGES bilan
            #     cheklanadi — bu A/B juftligidan KENGROQ (ilgarigi "faqat A
            #     va B tanlagan 2 til" mexanizmidan farqli), shunchaki
            #     Whisper'ning 99 tilli to'liq avtomatik aniqlashi o'rniga
            #     uning adashish maydonini (masalan koreyschani turkchaga
            #     yoki mongolchaga adashtirishini) qisqartiradi. Bu ovoz
            #     asosidagi spiker-aniqlashga umuman TA'SIR QILMAYDI — faqat
            #     Whisper'ning IKKINCHI (til) bosqichini aniqroq qiladi.
            try:
                text, detected_lang, lang_prob = await loop.run_in_executor(
                    None, transcribe_audio, temp_filename, None, SUPPORTED_LANGUAGES
                )
            except Exception as e:
                print(f"[/ws/dialog] turn processing failed: {e!r}", flush=True)
                await send_status("turn_skipped", reason="processing_error")
                return

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        text = (text or "").strip()
        if not text:
            print("[/ws/dialog] turn skipped (empty transcription)", flush=True)
            await send_status("turn_skipped", reason="empty_transcription")
            return

        if lang_prob < VERY_LOW_CONFIDENCE_LANG_PROB:
            print(
                f"[/ws/dialog] turn skipped (very_low_confidence: text={text!r} "
                f"lang_prob={lang_prob:.2f} < {VERY_LOW_CONFIDENCE_LANG_PROB}) — "
                f"ehtimol Whisper gallyutsinatsiyasi, matn ko'rsatilmadi",
                flush=True,
            )
            await send_status("turn_skipped", reason="very_low_confidence")
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

        # Tarjima manbai — bu navbatda HAQIQATDA aniqlangan til (`detected_lang`),
        # spikerning oldindan tanlagan tili EMAS (masalan odam ikkinchi tilga
        # o'tib qolsa ham to'g'ri tarjima qilinsin uchun). Yo'nalish esa
        # suhbatdoshning (ovoz orqali aniqlangan speaker asosida) tanlagan
        # tili — u har doim shu ikki tildan biri bo'lgani uchun o'zgarishsiz.
        source_lang = detected_lang
        other_speaker = "B" if speaker == "A" else "A"
        target_lang = state["speaker_langs"].get(other_speaker, "en")

        if source_lang == target_lang:
            # Suhbatdosh shu tilni allaqachon tushunadi — tarjima shart emas.
            translation = text
        elif (
            source_lang not in NLLB_LANG_CODES
            or target_lang not in NLLB_LANG_CODES
            or lang_prob < LANGUAGE_MATCH_MIN_PROB
        ):
            # NLLB bu tilni qo'llamaydi yoki aniqlash ishonchi juda past —
            # noto'g'ri/buzilgan tarjima chiqarish o'rniga asl matn ko'rsatiladi.
            print(
                f"[/ws/dialog] translation skipped (source_lang={source_lang!r} "
                f"target_lang={target_lang!r} lang_prob={lang_prob:.2f}) — showing original text",
                flush=True,
            )
            translation = text
        else:
            try:
                translation = await loop.run_in_executor(
                    None, translate_text, text, target_lang, source_lang
                )
            except Exception as e:
                print(f"[/ws/dialog] translation failed: {e!r}", flush=True)
                return

        await send_status(
            "turn",
            seq=seq,
            speaker=speaker,
            original=text,
            translation=translation,
            source_lang=source_lang,
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
                    # Seq SHU YERDA (audio qabul qilingan paytda, task
                    # yaratilishidan OLDIN) beriladi — bu WebSocket orqali
                    # audio kelish tartibi, ya'ni aytilish tartibi. Har bir
                    # turn keyin process_turn() ichida PARALLEL (alohida
                    # asyncio task) qayta ishlanadi va STT/tarjima vaqti
                    # turn uzunligiga qarab farq qilgani uchun TUGASH tartibi
                    # bu bilan mos kelmasligi mumkin — shuning uchun frontend
                    # natijalarni kelish tartibida emas, shu seq bo'yicha
                    # saralaydi (useDialogSession.js'ga qarang).
                    state["turn_seq"] += 1
                    seq = state["turn_seq"]
                    task = asyncio.create_task(process_turn(data, seq))
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
