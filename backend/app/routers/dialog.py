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
    SAMPLE_RATE,
)
from app.models.speaker_diart import (
    create_session,
    identify_speaker,
    apply_speaker_correction,
    both_slots_bound,
    save_turn_label,
)
from app.models.translator import translate_text
from app.config import MIN_TURN_SECONDS, NLLB_LANG_CODES
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# DIALOG_PLAN.md 1-bosqich (B2/B3): avvalgi VERY_LOW_CONFIDENCE_LANG_PROB
# (0.3) va SHORT_TEXT_MIN_PROB (0.5) chegaralari OLIB TASHLANDI. Ular
# transcribe_audio()'dan qaytgan lang_prob'ga (xom, ~99-tilli taqsimotdan
# olingan, real mikrofon audiosida to'g'ri til uchun ham past/beqaror
# qiymat) tayangan edi — bunday mutlaq chegara ko'plab haqiqiy gaplarni
# ham rad etardi. Endi gallyutsinatsiyaga qarshi himoya transcribe_audio()
# ICHIDA (stt.py: no_speech_prob/avg_logprob/compression_ratio + fantom-
# ibora ro'yxati) amalga oshiriladi. Shu yerda qoladigan bitta chegara —
# pastda, tarjima qilish-qilmaslik qarori uchun ishlatiladi (2-bosqichdan
# keyin, lang_prob endi 2-til-normallashtirilgan bo'lgani uchun ANCHA
# ishonchli signal; speaker_diart.py'dagi CENTROID_UPDATE_MIN_LANG_PROB
# bilan ADASHTIRMASLIK kerak — u OVOZ profilini yangilash uchun, bu esa
# TARJIMA uchun alohida chegara).
LANGUAGE_MATCH_MIN_PROB = 0.15

# Frontend VAD gapning OXIRIDAGI tabiiy ovoz pasayishini ba'zan erta jimlik
# deb hisoblab, navbatni vaqtidan oldin to'xtatib yuborishi mumkin (frontend
# tarafda SILENCE_DURATION_MS/SPEECH_RMS_THRESHOLD biroz yumshatildi, lekin
# bu holat butunlay yo'q qilinishi kafolatlanmaydi) — natijada gapning
# so'nggi 3-4 so'zi alohida, juda qisqa va past energiyali audio bo'lak
# sifatida keladi va o'z holicha embedding/STT uchun yetarli bo'lmay
# unknown_speaker/too_short deb rad etiladi. Shu sababli: agar keyingi
# bo'lak oldingisidan shu GAP ichida kelsa VA o'zi QISQA bo'lsa, buni
# alohida navbat emas, oldingisining DAVOMI deb hisoblab, ikkalasi audio
# darajasida birlashtiriladi va BITTA STT/spiker-aniqlash jarayonidan
# o'tkaziladi. Bu har bir navbatga shu GAP miqdoricha kechikish qo'shadi
# (ehtimoliy davom bo'lagini kutish uchun) — jonli tarjima uchun qabul
# qilinadigan muvozanat.
MERGE_GAP_S = 0.3
MERGE_CONTINUATION_MAX_DURATION_S = 3.0

# TEZLIK TUZATISHI (maqsad: gap tugashi -> tarjima ~2s): MERGE_GAP_S HAR BIR
# navbat uchun doimiy qo'shiladigan kechikish edi (davom bo'lagini kutish
# uchun) — 0.5'dan 0.3'ga tushirildi. Davomi hali ham qisman o'tkazib
# yuborilsa, pastdagi RETRO_MERGE_WINDOW_S (STT/tarjimadan KEYIN, kechikish
# QO'SHMASDAN birlashtiradigan mexanizm) buni qoplaydi.
#
# RETROAKTIV BIRLASHTIRISH: agar BIR XIL spiker (BIR XIL tilda) oldingi
# turn frontendga yuborilgandan keyin shu vaqt ICHIDA yana gapirsa — buni
# YANGI, alohida kutish bilan emas, oldingi (allaqachon yuborilgan) turn
# matniga qo'shib, BUTUN matn uchun tarjimani qayta qilib, XUDDI SHU seq
# bilan yangilangan "turn" sifatida yuboramiz (frontend'dagi upsert
# mantiqi — useDialogSession.js — buni yangi bubble emas, mavjudini
# almashtiruvchi deb qabul qiladi). Bu MERGE_GAP_S'dan ANCHA uzoqroq oyna
# (2.5s) — chunki endi hech qanday qo'shimcha KUTISH yo'q, navbat DARHOL
# qayta ishlanadi, faqat natija KEYINROQ (agar kerak bo'lsa) birlashtiriladi.
RETRO_MERGE_WINDOW_S = 2.5

# Frontend VAD'ning SPEECH_RMS_THRESHOLD'i (useDialogSession.js) gap-oxiri
# so'zlarini kesib yubormasligi uchun atayin past qo'yilgan — buning oqibati,
# real suhbatda juda ko'p sonli, past-energiyali shovqin bo'lagi ham "nutq
# boshlandi" deb serverga yuborilishi (har biri trim/RMS/embedding/STT
# hisoblashini talab qiladi). Bunga backend tarafdan ikkita slot (A, B)
# ALLAQACHON ovoz profiliga ega bo'lgandan KEYIN (ya'ni yangi spiker endi
# umuman kutilmaydigan holatda) qo'yiladigan "fast path": agar audio ham
# QISQA ham JUDA PAST energiyali bo'lsa, STT/embedding'ga umuman
# yubormasdan, darhol shovqin deb rad etiladi. Maqsad — nutqni noto'g'ri
# rad etish emas, balki aniq-ravshan shovqinni protsessor/tarmoq
# sarflamasdan saralab tashlash, shuning uchun chegaralar ATAYLAB
# konservativ (juda past RMS, juda qisqa davomiylik) tanlangan — hatto
# past ovozli haqiqiy nutq ham bu yerdan xavfsiz o'tib ketishi uchun.
FAST_PATH_MAX_DURATION_S = 2.0
FAST_PATH_MAX_RMS = 0.01

# DIALOG_PLAN.md 4-bosqich: "spikerni almashtirish" tuzatishi uchun qancha
# oxirgi turn xotirada (audio bilan) saqlanadi — uzoq suhbatda xotira
# cheksiz o'smasligi uchun chegara.
MAX_TURN_HISTORY = 50


@router.websocket("/ws/dialog")
async def websocket_dialog(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    state = {
        "phase": "await_lang_a",
        "speaker_langs": {},
        "diart_session": create_session(),
        "turn_seq": 0,
        # Hali dispatch qilinmagan, davom bo'lagi kelishi mumkinligi uchun
        # kutilayotgan navbat: {"wav": np.ndarray, "seq": int, "dispatch_task": Task}
        "pending": None,
        # DIALOG_PLAN.md 4-bosqich: "spikerni almashtirish" tugmasi uchun —
        # muvaffaqiyatli ko'rsatilgan HAR BIR turnning audiosi/matni shu
        # yerda seq bo'yicha saqlanadi, correct_speaker action shundan
        # qayta tarjima qilish va markazni to'g'rilash uchun foydalanadi.
        # MAX_TURN_HISTORY'dan oshganda eng eskisi tashlab yuboriladi —
        # uzoq sessiyada xotira cheksiz o'smasligi uchun.
        "turn_history": {},
        # RETROAKTIV BIRLASHTIRISH uchun: har bir spiker uchun MUVAFFAQIYATLI
        # yuborilgan ("turn" statusi bilan) ENG SO'NGGI natija — {"seq":,
        # "lang":, "sent_at": time.monotonic()}. process_turn() yangi
        # navbatni shu bilan solishtirib, birlashtirish kerakligini
        # aniqlaydi (qarang RETRO_MERGE_WINDOW_S izohi yuqorida).
        "last_turn_by_speaker": {},
    }
    # DIALOG_PLAN.md 1-bosqich (D1): navbatlar endi shu YAGONA FIFO navbat
    # orqali, turn_worker() tomonidan KETMA-KET qayta ishlanadi — qarang
    # turn_worker() va dispatch_pending_now() izohlariga.
    turn_queue: asyncio.Queue = asyncio.Queue()

    async def send_status(message: str, **extra):
        try:
            await websocket.send_json({"status": message, **extra})
        except Exception:
            pass


    async def process_turn(wav, seq: int):
        # TEZLIK TUZATISHI (6-band): butun navbatning "audio keldi ->
        # frontendga natija yuborildi" vaqtini o'lchash uchun — pastdagi
        # yakuniy [TURN] logiga qarang.
        t_turn_start = time.perf_counter()
        stt_elapsed = 0.0
        mt_elapsed = 0.0

        duration_s = wav.size / SAMPLE_RATE
        if duration_s < MIN_TURN_SECONDS:
            logger.info(f"[/ws/dialog] turn skipped (too short: {duration_s:.2f}s)")
            await send_status("turn_skipped", reason="too_short")
            return

        if duration_s < FAST_PATH_MAX_DURATION_S and both_slots_bound(state["diart_session"]):
            rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
            if rms < FAST_PATH_MAX_RMS:
                logger.info(f"[/ws/dialog] turn skipped (fast_path_noise: duration={duration_s:.2f}s "
                    f"rms={rms:.4f} < {FAST_PATH_MAX_RMS}, A/B slots allaqachon band) — "
                    f"STT/embedding chaqirilmadi")
                await send_status("turn_skipped", reason="unknown_speaker")
                return

        temp_filename = f"/tmp/{uuid.uuid4()}_turn.wav"
        await loop.run_in_executor(None, _write_wav, temp_filename, wav)

        # DIALOG_PLAN.md 2-bosqich: spiker aniqlash endi TIL va OVOZ
        # signallarining birlashgan balli orqali hal qilinadi
        # (speaker_diart.identify_speaker) — shuning uchun STT endi HAR
        # DOIM, speaker-aniqlashdan OLDIN ishga tushadi (eski koddagi
        # "faqat founding-sample tekshiruvi uchun shartli STT" mexanizmi
        # endi kerak emas, shu bilan birga stt_cache ham olib tashlandi).
        #
        # DIALOG_PLAN.md 1-bosqich (B1): butun ilovaning SUPPORTED_LANGUAGES
        # (en/ru/ko) ro'yxati o'rniga, endi FAQAT shu sessiyaning ikki tili
        # ([langA, langB]) nomzod sifatida beriladi. `state["speaker_langs"]`
        # bu yerga kelguncha ALLAQACHON ikkalasi ham to'ldirilgan bo'ladi
        # (phase="listening"ga faqat ikkisi ham tanlangandan keyin
        # o'tiladi — handle_client_message'ga qarang).
        session_langs = [state["speaker_langs"]["A"], state["speaker_langs"]["B"]]

        t_stt0 = time.perf_counter()
        try:
            try:
                text, detected_lang, lang_prob, unsupported_language = await loop.run_in_executor(
                    None, transcribe_audio, temp_filename, None, session_langs
                )
            except Exception as e:
                logger.error(f"[/ws/dialog] turn processing failed: {e!r}")
                await send_status("turn_skipped", reason="processing_error")
                return
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        stt_elapsed = time.perf_counter() - t_stt0

        # REAL TESTDA TOPILGAN MUAMMO (2-band): tekshiruv `empty_transcription`
        # dan OLDIN — stt.py bu holatda ATAYLAB bo'sh matn qaytaradi (qimmat
        # to'liq decode'ga umuman o'tmay), shuning uchun pastdagi `if not text`
        # bunga "Hech narsa eshitilmadi" deb noto'g'ri sabab berib qo'yardi.
        # Bu yerda identify_speaker() UMUMAN chaqirilmaydi — demak hech bir
        # spiker centroidi bu turndan ta'sirlanmaydi.
        if unsupported_language:
            logger.info(
                f"[/ws/dialog] turn skipped (unsupported_language: session_langs={session_langs})"
            )
            await send_status("turn_skipped", reason="unsupported_language")
            return

        text = (text or "").strip()
        if not text:
            logger.info("[/ws/dialog] turn skipped (empty transcription)")
            await send_status("turn_skipped", reason="empty_transcription")
            return

        debug_label: dict = {}
        speaker, score_debug = await loop.run_in_executor(
            None,
            identify_speaker,
            state["diart_session"],
            wav,
            SAMPLE_RATE,
            state["speaker_langs"],
            detected_lang,
            lang_prob,
            debug_label,
        )

        if speaker == "ambiguous":
            # A2 (margin): ikki spikerning balli yetarlicha farqlanmadi —
            # kim gapirgani ANIQ emas. Bu yerda ATAYLAB taxmin QILINMAYDI;
            # o'rniga frontendga ALOHIDA sabab ("ambiguous_speaker") +
            # qo'shimcha ma'lumot (asl matn, seq) yuboriladi — kelajakda
            # frontend shu yerga "A yoki B?" so'raydigan tanlash tugmasini
            # qo'shishi mumkin (hozircha faqat skip qilinadi, qarang
            # useDialogSession.js'dagi SKIP_REASON_MESSAGES).
            logger.info(f"[/ws/dialog] turn ambiguous (score_debug={score_debug})")
            save_turn_label(debug_label.get("wav_filename"), speaker=None, lang=detected_lang, text=text)
            await send_status(
                "turn_skipped",
                reason="ambiguous_speaker",
                seq=seq,
                original=text,
                source_lang=detected_lang,
            )
            return

        save_turn_label(debug_label.get("wav_filename"), speaker=speaker, lang=detected_lang, text=text)

        # Tarjima manbai — bu navbatda HAQIQATDA aniqlangan til (`detected_lang`),
        # spikerning oldindan tanlagan tili EMAS (masalan odam ikkinchi tilga
        # o'tib qolsa ham to'g'ri tarjima qilinsin uchun). Yo'nalish esa
        # suhbatdoshning (ovoz orqali aniqlangan speaker asosida) tanlagan
        # tili — u har doim shu ikki tildan biri bo'lgani uchun o'zgarishsiz.
        source_lang = detected_lang
        other_speaker = "B" if speaker == "A" else "A"
        target_lang = state["speaker_langs"].get(other_speaker, "en")

        # RETROAKTIV BIRLASHTIRISH (3-band, tezlik tuzatishi): bu navbatni
        # KUTIB TURMASDAN (hech qanday qo'shimcha delay qo'shilmaydi, STT
        # yuqorida ALLAQACHON tugadi) — agar AYNAN shu spiker AYNAN shu
        # tilda, oldingi (allaqachon frontendga yuborilgan) natijadan keyin
        # RETRO_MERGE_WINDOW_S ichida yana gapirgan bo'lsa, bu "yangi gap"
        # emas, oldingisining DAVOMI deb hisoblanadi: matnlar qo'shiladi,
        # tarjima BUTUN (birlashtirilgan) matn uchun qayta qilinadi va
        # natija YANGI bubble emas, ESKI seq bilan "turn" sifatida qaytadi
        # (useDialogSession.js'dagi upsert shuni mavjud bubble'ni
        # almashtiradi deb qabul qiladi).
        merge_seq = None
        merge_prev_entry = None
        last_turn = state["last_turn_by_speaker"].get(speaker)
        if (
            last_turn is not None
            and last_turn["lang"] == source_lang
            and (time.monotonic() - last_turn["sent_at"]) <= RETRO_MERGE_WINDOW_S
        ):
            candidate_entry = state["turn_history"].get(last_turn["seq"])
            if candidate_entry is not None:
                merge_seq = last_turn["seq"]
                merge_prev_entry = candidate_entry

        if merge_prev_entry is not None:
            out_seq = merge_seq
            out_text = f"{merge_prev_entry['text']} {text}".strip()
            out_wav = np.concatenate([merge_prev_entry["wav"], wav])
            logger.info(
                f"[/ws/dialog] retro-merging seq={seq} into seq={out_seq} "
                f"(speaker={speaker} lang={source_lang})"
            )
        else:
            out_seq = seq
            out_text = text
            out_wav = wav

        if source_lang == target_lang:
            # Suhbatdosh shu tilni allaqachon tushunadi — tarjima shart emas.
            translation = out_text
        elif (
            source_lang not in NLLB_LANG_CODES
            or target_lang not in NLLB_LANG_CODES
            or lang_prob < LANGUAGE_MATCH_MIN_PROB
        ):
            # NLLB bu tilni qo'llamaydi yoki aniqlash ishonchi juda past —
            # noto'g'ri/buzilgan tarjima chiqarish o'rniga asl matn ko'rsatiladi.
            logger.info(f"[/ws/dialog] translation skipped (source_lang={source_lang!r} "
                f"target_lang={target_lang!r} lang_prob={lang_prob:.2f}) — showing original text")
            translation = out_text
        else:
            t_mt0 = time.perf_counter()
            try:
                translation = await loop.run_in_executor(
                    None, translate_text, out_text, target_lang, source_lang
                )
            except Exception as e:
                # DIALOG_PLAN.md 1-bosqich (D2): ilgari bu yerda shunchaki
                # `return` qilinardi — frontendga HECH NARSA yuborilmasdan,
                # turn "yo'qolib" qolardi (foydalanuvchi hatto "eshitilmadi"
                # degan xabarni ham ko'rmasdi). Endi boshqa skip-sabablar
                # bilan BIR XIL tarzda turn_skipped yuboriladi.
                logger.error(f"[/ws/dialog] translation failed: {e!r}")
                await send_status("turn_skipped", reason="translation_error")
                return
            mt_elapsed = time.perf_counter() - t_mt0

        # DIALOG_PLAN.md 4-bosqich: "spikerni almashtirish" tugmasi uchun
        # — foydalanuvchi keyinroq shu turnni tuzatsa, qayta tarjima
        # qilish/markazni to'g'rilash uchun kerak bo'ladigan hamma narsa
        # shu yerda saqlanadi (qarang handle_client_message'dagi
        # `correct_speaker`). Birlashtirilgan holatda ESKI seq ustidan
        # YOZILADI (yangi seq uchun alohida yozuv OCHILMAYDI).
        state["turn_history"][out_seq] = {
            "wav": out_wav,
            "speaker": speaker,
            "text": out_text,
            "detected_lang": source_lang,
        }
        history = state["turn_history"]
        while len(history) > MAX_TURN_HISTORY:
            del history[min(history)]

        # Keyingi davomlar shu (yangilangan) seq'ga qarab zanjirlansin —
        # 3-chi bo'lak ham 2.5s ichida kelsa, u ham AYNAN shu bubble'ga
        # qo'shiladi.
        state["last_turn_by_speaker"][speaker] = {
            "seq": out_seq,
            "lang": source_lang,
            "sent_at": time.monotonic(),
        }

        logger.info(
            f"[TURN] seq={out_seq} audio={duration_s:.2f}s stt={stt_elapsed:.2f}s "
            f"mt={mt_elapsed:.2f}s total={time.perf_counter() - t_turn_start:.2f}s "
            f"merged={merge_prev_entry is not None}"
        )

        await send_status(
            "turn",
            seq=out_seq,
            speaker=speaker,
            original=out_text,
            translation=translation,
            source_lang=source_lang,
            target_lang=target_lang,
            # REAL TESTDA TOPILGAN MUAMMO (3-band): til va ovoz ZID bo'lsa
            # (identify_speaker'dagi VOICE_CONFLICT_MARGIN tekshiruvi),
            # natija hali ham ko'rsatiladi (til bo'yicha), lekin frontend
            # bu bayroqni ko'rib bubble'da "⚠ spiker noaniq" belgisini
            # chiqaradi — foydalanuvchi kerak bo'lsa "spikerni almashtirish"
            # orqali qo'lda to'g'rilashi mumkin.
            conflict=bool(score_debug.get("conflict", False)),
        )


    async def turn_worker():
        """DIALOG_PLAN.md 1-bosqich (D1): navbatlar endi SESSIYA BO'YICHA
        KETMA-KET, shu YAGONA worker tomonidan qayta ishlanadi. Ilgari har
        bir navbat alohida `asyncio.create_task()` bilan PARALLEL ishga
        tushirilardi — bu klaster yaratish/A-B bog'lash tartibini qaysi
        navbat birinchi TUGASHIGA (STT/tarjima vaqti uzunlikka qarab farq
        qiladi) bog'liq qilib qo'yardi, natija har safar boshqacha chiqishi
        mumkin edi; GPU baribir bitta bo'lgani uchun parallellikdan tezlik
        yutug'i ham yo'q edi. Endi `turn_queue`ga navbatlar KELISH tartibida
        qo'yiladi (dispatch_pending_now) va shu funksiya ularni process_turn()
        to'liq tugaguncha keyingisini boshlamasdan, bir-birining ortidan
        qayta ishlaydi — `None` navbatga qo'yilsa (sessiya yopilayotganda),
        worker navbatda qolgan hammasini tugatib, so'ng to'xtaydi."""
        while True:
            item = await turn_queue.get()
            try:
                if item is None:
                    break
                wav, seq = item
                # DIALOG_PLAN.md 4-bosqich: frontend shu statusni ko'rib
                # "Qayta ishlanmoqda…" deb ko'rsatadi (qarang
                # useDialogSession.js) — STT/tarjima bir necha soniya
                # davom etishi mumkin, foydalanuvchi shu payt hech narsa
                # ko'rinmayotganidan "ishlamayapti" deb o'ylab qolmasligi
                # uchun. process_turn() tugashi (natija "turn" yoki
                # "turn_skipped" bilan) shu holatni frontend tarafda
                # avtomatik tugatadi.
                await send_status("processing")
                await process_turn(wav, seq)
            except Exception as e:
                logger.error(f"[/ws/dialog] turn_worker: process_turn raised unexpectedly: {e!r}")
            finally:
                turn_queue.task_done()

    worker_task = asyncio.create_task(turn_worker())

    def dispatch_pending_now(waited_s: float = 0.0):
        """`state['pending']`da turgan (davom bo'lagi endi kelmaydi deb hisoblangan)
        navbatni darhol `turn_queue`ga qo'yadi — qayta ishlov esa turn_worker()
        tomonidan ketma-ket amalga oshiriladi.

        TEZLIK TUZATISHI (MERGE_GAP_S'ni faqat kerak bo'lganda qo'llash): har bir
        dispatch qarori uchun bitta [MERGE] log qatori — `waited_s` chaqiruvchi
        tomonidan beriladi (0.0 — kutilmasdan darhol, yoki MERGE_GAP_S — to'liq
        kutish muddati o'tgandan keyin), `merged` esa shu pending umri davomida
        kamida bitta davom-bo'lagi audio darajasida qo'shilganini bildiradi."""
        pending = state["pending"]
        state["pending"] = None
        logger.info(
            f"[MERGE] seq={pending['seq']} reason={pending.get('reason')} "
            f"waited={waited_s:.2f}s merged={pending.get('merge_count', 0) > 0}"
        )
        turn_queue.put_nowait((pending["wav"], pending["seq"]))

    async def dispatch_after_delay(seq):
        try:
            await asyncio.sleep(MERGE_GAP_S)
        except asyncio.CancelledError:
            return
        pending = state["pending"]
        # `pending` shu funksiya kutayotgan paytda ALLAQACHON davom bo'lagi
        # bilan birlashtirilib, YANGI dispatch_after_delay bilan almashtirilgan
        # bo'lishi mumkin (shu holda bu eski task cancel qilingan va bu yergacha
        # yetib kelmaydi) — shuning uchun seq moslig'i tekshiriladi, ikki marta
        # dispatch qilib yubormaslik uchun.
        if pending is None or pending["seq"] != seq:
            return
        dispatch_pending_now(waited_s=MERGE_GAP_S)

    async def handle_audio_chunk(data: bytes, reason: str | None):
        # DIALOG_PLAN.md 3-bosqich (C3): endi xom Float32 PCM (frontend
        # AudioWorkletNode orqali yuboradi) — webm/ffmpeg decode shart
        # emas, shuning uchun bu sinxron, tezkor chaqiruv executor'ga
        # chiqarilmaydi (eski _decode_waveform_bytes sekin edi, chunki
        # tempfile yozib, keyin decode_audio/ffmpeg orqali o'qirdi).
        wav = _decode_pcm_bytes(data)
        if wav is None or wav.size == 0:
            logger.info("[/ws/dialog] turn skipped (empty audio)")
            await send_status("turn_skipped", reason="empty_audio")
            return

        duration_s = wav.size / SAMPLE_RATE
        pending = state["pending"]

        if pending is not None and duration_s < MERGE_CONTINUATION_MAX_DURATION_S:
            # Frontend VAD gap oxiridagi tabiiy ovoz pasayishini erta jimlik
            # deb kesib yuborgan bo'lishi mumkin — bu bo'lak oldingi
            # (hali dispatch qilinmagan) navbatning DAVOMI, shuning uchun
            # audio darajasida birlashtiriladi, SEQ esa asosiy (oldingi)
            # navbatnikicha qoladi. Bu birlashtirish mantig'ining o'zi
            # o'ZGARTIRILMAGAN — faqat pastda, ENDI YARATILAYOTGAN
            # (birlashtirilgan) pending uchun KUTISH KERAKMI degan qaror
            # YANGI, shu CHUNK'ning o'z `reason`iga qarab beriladi.
            if pending.get("dispatch_task") is not None:
                pending["dispatch_task"].cancel()
            merged_wav = np.concatenate([pending["wav"], wav])
            seq = pending["seq"]
            merge_count = pending.get("merge_count", 0) + 1
            logger.info(f"[/ws/dialog] merged short continuation (dur={duration_s:.2f}s) "
                f"into pending turn seq={seq} (new total dur="
                f"{merged_wav.size / SAMPLE_RATE:.2f}s)")
        else:
            merged_wav = wav
            # Seq SHU YERDA (audio qabul qilingan paytda, dispatch qilinishidan
            # OLDIN) beriladi — bu WebSocket orqali audio kelish tartibi, ya'ni
            # aytilish tartibi. Navbatlar endi turn_worker() tomonidan shu
            # KELISH tartibida, KETMA-KET qayta ishlanadi (D1), shuning uchun
            # natijalar odatda tugash tartibida HAM shu bilan mos keladi —
            # frontend'dagi seq bo'yicha saralash (useDialogSession.js) shunga
            # qaramay xavfsizlik to'ri sifatida saqlab qolingan.
            state["turn_seq"] += 1
            seq = state["turn_seq"]
            merge_count = 0

        # TEZLIK TUZATISHI: MERGE_GAP_S ILGARI HAR DOIM, so'zsiz qo'llanardi —
        # lekin "davomi" (continuation) faqat frontend gapni MAJBURAN
        # (max_duration) kesganda real ehtimol; odam o'zi JIMLIK bilan
        # to'xtaganda (reason="silence") frontend buni ALLAQACHON
        # SILENCE_DURATION_MS kutib finalize qilgan — shu holatda yana
        # MERGE_GAP_S kutish birlashtirishga HECH QACHON yordam bermaydi
        # (davomi kelsa, u allaqachon SILENCE_DURATION_MS ichida YANGI,
        # alohida audio sifatida kelib bo'lardi), faqat sof kechikish
        # qo'shardi. Shu sababli: reason="silence" — kutmasdan darhol
        # navbatga; reason="max_duration" YOKI reason yo'q (eski frontend,
        # xavfsiz default) — eskicha MERGE_GAP_S kutiladi. Birlashtirishning
        # o'zi (yuqoridagi audio-concatenate) O'ZGARISHSIZ qoldi.
        if reason == "silence":
            state["pending"] = {
                "wav": merged_wav,
                "seq": seq,
                "dispatch_task": None,
                "reason": reason,
                "merge_count": merge_count,
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
            }

    async def receiver():
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                if state["phase"] == "listening":
                    # Frontend turn_meta ("action": "turn_meta", "reason": ...)
                    # matnli xabarini AYNAN shu binary chunk'dan OLDIN yuboradi
                    # (useDialogSession.js) — WebSocket bir ulanishda xabarlarni
                    # KETMA-KET saqlagani uchun, handle_client_message() bu
                    # sababni `state["pending_turn_reason"]`ga allaqachon
                    # yozib qo'ygan bo'ladi.
                    reason = state.pop("pending_turn_reason", None)
                    await handle_audio_chunk(data, reason)
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

        elif action == "turn_meta":
            # TEZLIK TUZATISHI: frontend har bir turn audio'sidan OLDIN shuni
            # yuboradi (useDialogSession.js'dagi sendTurnAudio) — keyingi
            # binary chunk uchun "reason" (silence/max_duration) shu yerda
            # vaqtincha saqlanadi, receiver() uni handle_audio_chunk()ga
            # uzatishdan oldin o'qib oladi.
            state["pending_turn_reason"] = msg.get("reason")

        elif action == "stop_dialog":
            await websocket.close()

        elif action == "correct_speaker" and state["phase"] == "listening":
            # DIALOG_PLAN.md 4-bosqich: har bir bubble'dagi "⇄ spikerni
            # almashtirish" tugmasi — foydalanuvchi "bu A emas, B edi"
            # deb bosganda shu yerga keladi. Saqlangan turn_history
            # yozuvidan (process_turn'ga qarang) spikerni almashtiramiz,
            # YO'NALISHI teskari bo'lgani uchun tarjimani QAYTA bajaramiz
            # va to'g'ri spikerning ovoz markazini yangilaymiz — natija
            # frontendga XUDDI SHU `seq` bilan "turn" sifatida qayta
            # yuboriladi (useDialogSession.js buni yangi bubble emas,
            # MAVJUD bubble'ning o'rnini bosuvchi deb qabul qiladi).
            seq = msg.get("seq")
            entry = state["turn_history"].get(seq)
            if entry is None:
                logger.info(f"[/ws/dialog] correct_speaker: seq={seq} topilmadi (eskirgan?)")
                return

            new_speaker = "B" if entry["speaker"] == "A" else "A"
            other_speaker = "A" if new_speaker == "B" else "B"
            source_lang = entry["detected_lang"]
            target_lang = state["speaker_langs"].get(other_speaker, "en")
            text = entry["text"]

            if source_lang == target_lang:
                translation = text
            elif source_lang not in NLLB_LANG_CODES or target_lang not in NLLB_LANG_CODES:
                translation = text
            else:
                try:
                    translation = await loop.run_in_executor(
                        None, translate_text, text, target_lang, source_lang
                    )
                except Exception as e:
                    logger.error(f"[/ws/dialog] correct_speaker: translation failed: {e!r}")
                    return

            entry["speaker"] = new_speaker
            await loop.run_in_executor(
                None, apply_speaker_correction, state["diart_session"], entry["wav"], SAMPLE_RATE, new_speaker
            )

            await send_status(
                "turn",
                seq=seq,
                speaker=new_speaker,
                original=text,
                translation=translation,
                source_lang=source_lang,
                target_lang=target_lang,
            )


    await send_status("awaiting_language", speaker="A")

    try:
        await receiver()
    except Exception as e:
        # Bu bare except ilgari hech narsa chiqarmasdan (silent) hammasini
        # yutib yuborardi — masalan _decode_pcm_bytes() ValueError
        # ko'tarsa (frontend eski/mos kelmaydigan format yuborsa), butun
        # sessiya HECH QANDAY log qoldirmasdan o'lib qolardi. Debug qilishni
        # osonlashtirish uchun endi kamida bitta qator chiqariladi.
        logger.error(f"[/ws/dialog] receiver loop crashed: {e!r}")
    finally:
        # Session yopilayotganda hali dispatch qilinmagan (davom bo'lagi
        # kutilayotgan) oxirgi navbat bo'lsa — uni shunchaki tashlab
        # yubormasdan, navbatga qo'shib yuboramiz, aks holda suhbatning
        # oxirgi gapi butunlay yo'qolib qolardi.
        if state["pending"] is not None:
            if state["pending"].get("dispatch_task") is not None:
                state["pending"]["dispatch_task"].cancel()
            dispatch_pending_now(waited_s=0.0)

        # turn_worker()ga to'xtash signali — navbatda ENDI turgan hammasi
        # (yuqoridagi flush ham kiradi) KETMA-KET to'liq qayta ishlanib
        # bo'lingach, worker shu `None`ni ko'rib o'zi to'xtaydi.
        turn_queue.put_nowait(None)
        await worker_task

        if websocket.client_state.name != "DISCONNECTED":
            try:
                await websocket.close()
            except Exception:
                pass
