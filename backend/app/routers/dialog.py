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
    save_turn_label,
)
from app.models.translator import translate_text
from app.models.segmenter import segment_turn
from app.config import MIN_TURN_SECONDS, NLLB_LANG_CODES
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Til tanlash bosqichi olib tashlangani uchun sessiya tillari doim FAQAT
# shu ikkisi — STT'ga (Whisper yo'lida) nomzod sifatida beriladi,
# pair_mass/lang_prob filtri shu ikkisidan TASHQARI tilni rad etadi
# (qarang stt.py).
SESSION_LANGS = ["en", "ru"]

# DIALOG_PLAN.md 1-bosqich (B2/B3): avvalgi VERY_LOW_CONFIDENCE_LANG_PROB
# (0.3) va SHORT_TEXT_MIN_PROB (0.5) chegaralari OLIB TASHLANDI. Ular
# transcribe_audio()'dan qaytgan lang_prob'ga (xom, ~99-tilli taqsimotdan
# olingan, real mikrofon audiosida to'g'ri til uchun ham past/beqaror
# qiymat) tayangan edi — bunday mutlaq chegara ko'plab haqiqiy gaplarni
# ham rad etardi. Endi gallyutsinatsiyaga qarshi himoya transcribe_audio()
# ICHIDA (stt.py: no_speech_prob/avg_logprob/compression_ratio + fantom-
# ibora ro'yxati + MIN_PAIR_MASS/MIN_LANG_PROB) amalga oshiriladi. Shu
# yerda qoladigan bitta chegara — pastda, tarjima qilish-qilmaslik qarori
# uchun ishlatiladi.
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
# RETROAKTIV BIRLASHTIRISH: agar BIR XIL spiker (OVOZ QARORI bo'yicha, BIR
# XIL tilda) oldingi turn frontendga yuborilgandan keyin shu vaqt ICHIDA
# yana gapirsa — buni YANGI, alohida kutish bilan emas, oldingi
# (allaqachon yuborilgan) turn matniga qo'shib, BUTUN matn uchun tarjimani
# qayta qilib, XUDDI SHU seq bilan yangilangan "turn" sifatida yuboramiz
# (frontend'dagi upsert mantiqi — useDialogSession.js — buni yangi bubble
# emas, mavjudini almashtiruvchi deb qabul qiladi). Bu MERGE_GAP_S'dan
# ANCHA uzoqroq oyna (2.5s) — chunki endi hech qanday qo'shimcha KUTISH
# yo'q, navbat DARHOL qayta ishlanadi, faqat natija KEYINROQ (agar kerak
# bo'lsa) birlashtiriladi.
RETRO_MERGE_WINDOW_S = 2.5

# Mentor demo'da topilgan 3-muammo: real testda bitta bubble'ga 15+ bo'lak
# (456 belgi) qo'shilib ketgan edi — retroaktiv birlashtirish hech qanday
# chegaraga ega emas edi. Ikki mustaqil himoya:
#   1. uncertain=True YOKI speaker=None ("unknown") bo'lgan HAR QANDAY
#      tomon (joriy bo'lak HAM, oldingi bubble HAM) uchun birlashtirish
#      UMUMAN qo'llanmaydi — spiker ANIQ bo'lmagan holatda "bu aynan
#      shu odamning davomi" degan asos o'zi ham noaniq.
#   2. Bubble MAX_MERGED_CHARS (matn) yoki MAX_MERGED_AUDIO_S (audio)dan
#      oshsa — qo'shilmaydi, YANGI bubble boshlanadi (garchi spiker/til/
#      vaqt oynasi hammasi mos kelsa ham).
MAX_MERGED_CHARS = int(os.environ.get("MAX_MERGED_CHARS", "300"))
MAX_MERGED_AUDIO_S = float(os.environ.get("MAX_MERGED_AUDIO_S", "25"))

# Frontend VAD'ning SPEECH_RMS_THRESHOLD'i (useDialogSession.js) gap-oxiri
# so'zlarini kesib yubormasligi uchun atayin past qo'yilgan — buning oqibati,
# real suhbatda juda ko'p sonli, past-energiyali shovqin bo'lagi ham "nutq
# boshlandi" deb serverga yuborilishi (har biri trim/RMS/embedding/STT
# hisoblashini talab qiladi). Bunga backend tarafdan KAMIDA BITTA spiker
# uchun ovoz profili ALLAQACHON mavjud bo'lgandan KEYIN qo'yiladigan "fast
# path": agar audio ham QISQA ham JUDA PAST energiyali bo'lsa, STT/
# embedding'ga umuman yubormasdan, darhol shovqin deb rad etiladi. Maqsad —
# nutqni noto'g'ri rad etish emas, balki aniq-ravshan shovqinni protsessor/
# tarmoq sarflamasdan saralab tashlash, shuning uchun chegaralar ATAYLAB
# konservativ (juda past RMS, juda qisqa davomiylik) tanlangan — hatto
# past ovozli haqiqiy nutq ham bu yerdan xavfsiz o'tib ketishi uchun.
FAST_PATH_MAX_DURATION_S = 2.0
FAST_PATH_MAX_RMS = 0.01

# DIALOG_PLAN.md 4-bosqich: "spikerni almashtirish"/delete tuzatishi uchun
# qancha oxirgi turn xotirada (audio bilan) saqlanadi — uzoq suhbatda
# xotira cheksiz o'smasligi uchun chegara.
MAX_TURN_HISTORY = 50

# ---------------------------------------------------------------------------
# Mentor demo'da topilgan 1-muammo (shovqinli xona, fonda o'zbekcha
# gaplashuv): Silero VAD fondagi nutqni HAM to'g'ri "nutq" deb tutadi —
# frontend'dagi near-field qoidasi (useDialogSession.js, NEAR_FIELD_RATIO)
# FAQAT FAOL TURN ICHIDA ishlaydi (davom etayotgan gapni fon bilan
# aralashtirmaslik uchun), lekin turn BOSHLANISHIDA (yangi, alohida bo'lak
# sifatida) hech qanday masofa-filtri yo'q — fondagi gap ham, agar til
# filtridan tasodifan o'tib ketsa, to'liq bo'lak sifatida qabul qilinardi.
#
# Shu sababli BACKEND tarafida IKKINCHI, mustaqil qatlam: sessiya davomida
# QABUL QILINGAN (spikerga berilgan) bo'laklarning nutq-RMS'lari (qarang
# compute_speech_rms, stt.py) oxirgi NEAR_REF_MAX_SAMPLES tasi saqlanadi —
# `near_ref` ularning medianasi, "foydalanuvchining odatiy (mikrofonga
# yaqin) ovoz darajasi" degani. Yangi bo'lakning nutq-RMS'i shundan
# FAR_FIELD_TURN_RATIO ulushidan PAST bo'lsa — bu bo'lak (near_ref'ga
# nisbatan) "uzoqda" turgan, demak fondagi odam, deb hisoblanadi.
#
# "Umumiy" talab (sovuq start/fon yo'q holatda eski xatti-harakat
# o'zgarmasin): kamida NEAR_REF_MIN_TURNS ta namuna to'planmaguncha filtr
# UMUMAN qo'llanmaydi (pastga qarang).
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
        # Hali dispatch qilinmagan, davom bo'lagi kelishi mumkinligi uchun
        # kutilayotgan navbat: {"wav": np.ndarray, "seq": int, "dispatch_task": Task}
        "pending": None,
        # "spikerni almashtirish"/delete_turn tugmalari uchun — muvaffaqiyatli
        # ko'rsatilgan HAR BIR turnning audiosi/matni/tarjimasi shu yerda seq
        # bo'yicha saqlanadi. MAX_TURN_HISTORY'dan oshganda eng eskisi
        # tashlab yuboriladi — uzoq sessiyada xotira cheksiz o'smasligi uchun.
        "turn_history": {},
        # RETROAKTIV BIRLASHTIRISH uchun: har bir spiker (ID) uchun
        # MUVAFFAQIYATLI yuborilgan ("turn" statusi bilan) ENG SO'NGGI
        # natija — {"seq":, "lang":, "sent_at": time.monotonic()}.
        # process_turn() yangi navbatni shu bilan solishtirib, birlashtirish
        # kerakligini aniqlaydi (qarang RETRO_MERGE_WINDOW_S izohi yuqorida).
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


    def _trim_turn_history():
        history = state["turn_history"]
        while len(history) > MAX_TURN_HISTORY:
            del history[min(history)]


    async def _resend_turn(seq: int, entry: dict) -> None:
        """Mavjud (turn_history'da saqlangan) bir turnni XUDDI SHU seq
        bilan qayta yuboradi — frontend'dagi upsert mantiqi buni yangi
        bubble emas, mavjudining o'rnini bosuvchi deb qabul qiladi.
        correct_speaker'da ishlatiladi — tarjima/matn O'ZGARMAYDI, FAQAT
        spiker/uncertain bayrog'i yangilanadi."""
        await send_status(
            "turn",
            seq=seq,
            speaker=entry["speaker"],
            original=entry["text"],
            translation=entry["translation"],
            source_lang=entry["source_lang"],
            target_lang=entry["target_lang"],
            uncertain=entry.get("uncertain", False),
            # Tarjima: DeepL (asosiy) + NLLB-600M (zaxira) — frontend shuni
            # ko'rib tarjima yonida juda kichik "DeepL"/"lokal" belgisini
            # (yoki IKKISI HAM muvaffaqiyatsiz bo'lsa — "Tarjima vaqtincha
            # mavjud emas" kulrang matnini) ko'rsatadi.
            mt_provider=entry.get("mt_provider"),
            translation_error=entry.get("mt_provider") == "error",
            # Mentor demo'da topilgan 4-muammo: frontend har bir yuborilgan
            # audio bo'lagi uchun o'zining `client_seq`ini kuzatib,
            # yuborilgani bilan darhol "…" placeholder ko'rsatadi — bu
            # yerda shu bo'lak(lar)ga mos client_seq'lar ("merge"/retro-merge
            # bo'lsa bir nechtasi) qaytarilib, frontend mos placeholder(lar)ni
            # haqiqiy bubble bilan almashtira oladi (useDialogSession.js).
            client_seqs=entry.get("client_seqs", []),
        )


    async def _apply_merges_and_notify(merges: list[tuple[int, int, float]]) -> None:
        """speaker_diart.py'dagi identify_speaker/apply_speaker_correction/
        remove_turn_prototype — har biri avtomatik birlashishlarni
        ("merges", qarang SPK_MERGE_SIM) qaytarishi mumkin: ikkita ID
        aslida BITTA odam ekani aniqlanib, kichik raqamligiga birlashtirilgan.
        Bu yerda ular KETMA-KET (zanjir bo'lsa ham to'g'ri) qo'llaniladi:
        turn_history'dagi BARCHA mos yozuvlar (shu jumladan avvalgi
        qadamlarda ALLAQACHON yangilanganlar — shu sababli zanjir to'g'ri
        "ta'qib qilinadi") qayta tayinlanadi, so'ng frontendga
        "speakers_merged" yuboriladi — u o'z navbatida BARCHA "from"
        bubble'larini "into"ga ko'chiradi."""
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
        """Mentor demo'da topilgan 8-muammo: haqiqiy tarjima chaqiruvi
        (DeepL — tarmoq, yoki NLLB-600M — GPU) endi turn_worker()ni
        BLOKlamaydi — _process_subsegment() bu funksiyani `asyncio.
        create_task()` orqali ishga tushirib, DARHOL qaytadi (navbatdagi
        bo'lak GPU'ga — STT/embedding — kutilmasdan o'tadi). Natija
        tayyor bo'lganda shu funksiya turn_history yozuvini to'ldirib,
        "turn"ni (ENDI haqiqiy tarjima bilan) yuboradi.

        POYGA HOLATI TUZATISHI: shu orada (tarjima kutilayotgan paytda)
        `out_seq` RETRO-MERGE orqali boshqa matn bilan yangilangan
        bo'lishi mumkin (_process_subsegment o'zi HAR DOIM yangi matn
        uchun YANGI tarjima so'rovini boshlaydi) — shu sababli natija
        qo'llanishidan OLDIN entry["text"] HALI HAM shu funksiya
        tarjima qilgan AYNAN `out_text`ga teng ekani tekshiriladi;
        mos kelmasa (matn allaqachon o'zgargan/merge bo'lgan, yoki
        bubble o'chirilgan) — bu ESKIRGAN natija, qo'llanmaydi (uni
        allaqachon o'zgartirgan tomon o'ZINING tarjimasiga javobgar)."""
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
        sub_index: int = 0, sub_total: int = 1,
    ):
        """Bitta QISM (sub-segment, qarang segmenter.segment_turn) uchun
        to'liq quvur — STT, spiker aniqlash, retro-merge, tarjima. Bitta
        turn FAQAT bitta qismdan iborat bo'lsa (ODDIY, eski holat),
        process_turn uni AYNAN shu funksiya orqali, bir marta chaqiradi —
        natija (va vaqt sarfi) 5/6-bandgacha bo'lgan kod bilan bir xil.

        `is_multi` — shu turn bir nechta qismga bo'lingan bo'lsa True
        (6-band: begona-til skip sababi "foreign_subsegment" bo'ladi,
        "unsupported_language" o'rniga — frontend buni alohida xabar
        bilan ko'rsatishi mumkin). `purity` — segment_turn() qaytargan,
        shu qismning o'ZI qancha "toza" (bitta ovozga tegishli) ekanligi
        bahosi — 7-band: identify_speaker() shu orqali "iflos" (ehtimol
        ikki ovoz aralashgan) qismdan profilni YANGI prototip bilan
        ifloslantirmaydi."""
        # Mentor demo'da topilgan 4-muammo: shu turnga tegishli frontend
        # `client_seq`lar (bitta bo'lak — bitta qiymat, merge/retro-merge
        # bo'lsa bir nechtasi) — HAR BIR qaytariladigan "turn"/"turn_skipped"
        # xabariga qo'shiladi, frontend shu orqali mos placeholder(lar)ni
        # topadi (qarang _resend_turn va pastdagi send_status chaqiruvlari).
        client_seqs = client_seqs or []

        # TEZLIK TUZATISHI (6-band, ESKI izoh): "audio keldi -> natija
        # yuborildi" vaqtini o'lchash uchun. Mentor demo'da topilgan
        # 8-muammo: ENDI bu vaqt tarjima (ayniqsa DeepL tarmoq so'rovi)
        # vaqtini O'Z ICHIGA OLMAYDI — shu bosqich endi fonda, alohida
        # task sifatida ishlaydi (qarang _finish_translation).
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
                    None, transcribe_audio, temp_filename, None, SESSION_LANGS
                )
            except Exception as e:
                logger.error(f"[/ws/dialog] turn processing failed: {e!r}")
                await send_status("turn_skipped", reason="processing_error", client_seqs=client_seqs)
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
        # spiker profili bu turndan ta'sirlanmaydi.
        if unsupported_language:
            # Mentor demo'da topilgan 6-muammo: bitta turn bir nechta
            # qismga bo'lingan bo'lsa, FAQAT shu (begona tilli) QISM
            # tashlab yuboriladi — qolgan qismlar MUSTAQIL davom etadi
            # (process_turn'dagi tsiklga qarang). Shu sababli sabab
            # nomi ham alohida — frontend buni "butun gap tashlandi"
            # emas, "bir qismi tashlandi" sifatida talqin qila oladi.
            reason = "foreign_subsegment" if is_multi else "unsupported_language"
            logger.info(f"[/ws/dialog] turn skipped (reason={reason}: session_langs={SESSION_LANGS})")
            await send_status("turn_skipped", reason=reason, client_seqs=client_seqs)
            return

        text = (text or "").strip()
        if not text:
            logger.info("[/ws/dialog] turn skipped (empty transcription)")
            await send_status("turn_skipped", reason="empty_transcription", client_seqs=client_seqs)
            return

        # SPIKER ANIQLASH — endi FAQAT ovoz asosida, cheklanmagan sonli
        # onlayn klasterlash (qarang speaker_diart.identify_speaker).
        # `detected_lang` bu yerga FAQAT [SPK] logiga chiqarish uchun
        # uzatiladi — qarorga hech qanday ta'sir qilmaydi.
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

        # Shu turnni qayta ishlash davomida ikkita (yoki zanjir bo'lsa
        # undan ko'p) spiker AVTOMATIK bitta spikerga birlashtirilgan
        # bo'lishi mumkin (qarang speaker_diart.SPK_MERGE_SIM) — eski
        # turn_history yozuvlari va frontend shu yerda yangilanadi.
        # `speaker` o'zi identify_speaker ICHIDA ALLAQACHON remap qilingan
        # (post-merge, yakuniy ID).
        await _apply_merges_and_notify(spk_result.get("merges", []))

        # Mentor demo'da topilgan 1-muammo: "unknown" buferi (speaker_diart.py)
        # shu chaqiruv ichida BOSHQA (joriy seq'dan tashqari, oldin
        # "Aniqlanmoqda…" sifatida allaqachon frontendga yuborilgan
        # bo'lishi mumkin bo'lgan) bo'laklarni ham qayta tayinlagan bo'lishi
        # mumkin — ularning turn_history yozuvi (bo'lsa) yangilanadi va
        # "turn" sifatida qayta yuboriladi (xuddi correct_speaker kabi).
        for r_seq, r_speaker in spk_result.get("reassigned", []):
            r_entry = state["turn_history"].get(r_seq)
            if r_entry is None:
                continue
            r_entry["speaker"] = r_speaker
            r_entry["uncertain"] = False
            # "unknown" turnlar last_turn_by_speaker'ga HECH QACHON
            # yozilmaydi (pastga qarang) — shuning uchun bu yerda eski
            # yozuvni tozalash shart emas, faqat YANGI (endi ma'lum)
            # spiker uchun qo'shiladi.
            state["last_turn_by_speaker"][r_speaker] = {
                "seq": r_seq,
                "lang": r_entry["source_lang"],
                "sent_at": time.monotonic(),
            }
            logger.info(f"[/ws/dialog] unknown buffer reassigned seq={r_seq} -> speaker={r_speaker}")
            await _resend_turn(r_seq, r_entry)

        save_turn_label(debug_label.get("wav_filename"), speaker=speaker, lang=detected_lang, text=text)

        # Tarjima yo'nalishi ENDI spiker/sessiya tiliga EMAS, FAQAT shu
        # turnda HAQIQATDA aniqlangan tilga bog'liq — ikki sessiya tili
        # ([en, ru]) bo'lgani uchun yo'nalish oddiy: en bo'lsa → ru,
        # ru bo'lsa → en (STT'ning pair_mass/lang_prob filtri source_lang
        # har doim shu ikkisidan BIRI bo'lishini kafolatlaydi — aks holda
        # yuqorida `unsupported_language` bilan ALLAQACHON qaytarilgan).
        source_lang = detected_lang
        target_lang = "ru" if source_lang == "en" else "en"

        # RETROAKTIV BIRLASHTIRISH (3-band, tezlik tuzatishi): bu navbatni
        # KUTIB TURMASDAN (hech qanday qo'shimcha delay qo'shilmaydi, STT
        # yuqorida ALLAQACHON tugadi) — agar AYNAN shu spiker (OVOZ QARORI
        # bo'yicha) AYNAN shu tilda, oldingi (allaqachon frontendga
        # yuborilgan) natijadan keyin RETRO_MERGE_WINDOW_S ichida yana
        # gapirgan bo'lsa, bu "yangi gap" emas, oldingisining DAVOMI deb
        # hisoblanadi: matnlar qo'shiladi, tarjima BUTUN (birlashtirilgan)
        # matn uchun qayta qilinadi va natija YANGI bubble emas, ESKI seq
        # bilan "turn" sifatida qaytadi.
        merge_seq = None
        merge_prev_entry = None
        # Mentor demo'da topilgan 3-muammo: speaker=None ("unknown") uchun
        # retro-merge UMUMAN ko'rib chiqilmaydi — "aynan shu odamning
        # davomi" degan asosning o'zi noaniq bo'lgan holatda birlashtirish
        # ma'nosiz (va xavfli — keyin to'g'ri spikerga ajratish qiyinlashadi).
        last_turn = state["last_turn_by_speaker"].get(speaker) if speaker is not None else None
        if (
            last_turn is not None
            and last_turn["lang"] == source_lang
            and (time.monotonic() - last_turn["sent_at"]) <= RETRO_MERGE_WINDOW_S
            # Mentor demo'da topilgan 3-muammo: ikkala tomon ham ANIQ
            # (uncertain=False) bo'lishi SHART — yoki JORIY bo'lak, yoki
            # OLDINGI bubble noaniq bo'lsa, "bu aynan shu odam" degan
            # taxmin o'zi ishonchsiz, birlashtirish xato spikerni
            # "mustahkamlab" qo'yishi mumkin.
            and not uncertain
        ):
            candidate_entry = state["turn_history"].get(last_turn["seq"])
            if (
                candidate_entry is not None
                and not candidate_entry.get("uncertain", False)
                # Mentor demo'da topilgan 3-muammo: bubble hajm chegarasi —
                # MAX_MERGED_CHARS/MAX_MERGED_AUDIO_S'dan oshsa, qo'shish
                # o'rniga YANGI bubble boshlanadi (bir bubble'ga 15+ bo'lak,
                # 456 belgi qo'shilib ketgan real-test nosozligi).
                and len(candidate_entry["text"]) < MAX_MERGED_CHARS
                and (candidate_entry["wav"].size / SAMPLE_RATE) < MAX_MERGED_AUDIO_S
            ):
                merge_seq = last_turn["seq"]
                merge_prev_entry = candidate_entry

        if merge_prev_entry is not None:
            out_seq = merge_seq
            out_text = f"{merge_prev_entry['text']} {text}".strip()
            out_wav = np.concatenate([merge_prev_entry["wav"], wav])
            # REAL TESTDA TOPILGAN XATO: bu yozuv asosiy seq (`merge_seq`)
            # TEGI bilan qoladi, lekin shu `seq` ham (identify_speaker ichida)
            # O'ZINING prototipini qo'shgan bo'lishi mumkin edi — shu
            # prototip ESKI, asosiy seq bilan hech qanday bog'liqligi
            # bo'lmagani uchun correct_speaker/delete_turn uni topolmay,
            # eski spikerda "osilib qolardi". Shu sababli bubble'ga
            # tegishli BARCHA seq'lar (zanjir bo'lsa ham) shu yerda
            # to'planadi.
            merged_seqs = list(merge_prev_entry.get("merged_seqs", [out_seq]))
            merged_seqs.append(seq)
            # Mentor demo'da topilgan 4-muammo: retro-birlashtirilgan
            # bo'lakning OWN client_seq'lari ham oldingi bubble'ning
            # ro'yxatiga qo'shiladi — shu bo'lak uchun alohida yuborilgan
            # placeholder ham ENDI shu (eski) bubble bilan birga yopiladi.
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

        # Mentor demo'da topilgan 8-muammo: haqiqiy tarjima chaqiruvi
        # (pastdagi `else` — DeepL tarmoq so'rovi yoki NLLB-600M GPU
        # inference) ENDI `await` QILINMAYDI — alohida fon-task sifatida
        # ishga tushiriladi (_finish_translation), shunda turn_worker()
        # DARHOL navbatdagi bo'lakka o'tadi (GPU bosqichlari — STT,
        # embedding — bitta worker'da ketma-ket qoladi, FAQAT tarjima
        # mustaqil). "Tezkor" (tarmoq/GPU chaqiruvisiz) holatlar — til
        # bir xil, yoki NLLB qo'llamaydigan til, yoki past lang_prob —
        # BIR ZUMDA ma'lum bo'lgani uchun o'zgarishsiz, sinxron qoladi.
        mt_provider = None
        translation_pending = False
        if source_lang == target_lang:
            # Nazariy jihatdan amalda bo'lmasligi kerak (SESSION_LANGS ikki
            # XIL til), lekin xavfsizlik uchun saqlanadi.
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
            # Hali noma'lum — frontend bu seq uchun ALLAQACHON ko'rsatgan
            # placeholder ("…") shu natija kelgunga qadar davom etadi
            # (_resend_turn bu yerda ATAYLAB chaqirilMAYDI).
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
            # DeepL (asosiy)/NLLB-600M (zaxira)/None (tarjima umuman
            # chaqirilmagan, masalan source==target) — qarang _resend_turn.
            "mt_provider": mt_provider,
            # Mentor demo'da topilgan 4-muammo: frontend placeholder'larini
            # topish uchun — qarang _resend_turn.
            "client_seqs": out_client_seqs,
        }
        _trim_turn_history()

        # Keyingi davomlar shu (yangilangan) seq'ga qarab zanjirlansin —
        # 3-chi bo'lak ham 2.5s ichida kelsa, u ham AYNAN shu bubble'ga
        # qo'shiladi. Mentor demo'da topilgan 1/3-muammolar: speaker=None
        # ("unknown") uchun BU YOZILMAYDI — noaniq spiker hech qachon
        # "oxirgi gapirgan" fallback manbai ham, retro-merge nishoni ham
        # bo'lmasligi kerak.
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
            # Mentor demo'da topilgan 8-muammo: worker shu yerda TO'XTAMAYDI
            # — tarjima fonda tugaganda _finish_translation o'zi "turn"ni
            # (haqiqiy tarjima bilan) yuboradi.
            asyncio.create_task(_finish_translation(out_seq, out_text, source_lang, target_lang))
        else:
            await _resend_turn(out_seq, state["turn_history"][out_seq])


    async def process_turn(wav, seq: int, client_seqs: list | None = None):
        """Mentor demo'da topilgan 5/6-muammolar: bitta turn'ning XOM
        audiosini (kerak bo'lsa) segmenter.segment_turn() orqali bir
        nechta MUSTAQIL qismga ajratib, HAR BIRINI alohida
        _process_subsegment() orqali qayta ishlaydi. Bitta qism chiqsa
        (ODDIY, ko'pchilik holat) — natija AYNAN eski (segmentatsiyasiz)
        yo'l bilan bir xil (qarang segmenter.py'dagi FAST_PATH_MAX_DURATION_S).

        `seq`/`client_seqs` — FAQAT BIRINCHI qismga tegishli (frontend
        shu AYNAN shu ikkisi bilan bitta placeholder yuborgan edi, qarang
        useDialogSession.js'dagi sendTurnAudio). Qolgan qism(lar) uchun
        YANGI seq kerak — lekin ATAYLAB `state["turn_seq"]`DAN (ya'ni
        handle_audio_chunk HAR BIR YANGI jismoniy bo'lak kelganda
        oshiradigan, ARALASH hisoblagich) emas, balki ASOSIY `seq`ning
        o'zidan KICHIK KASR qo'shib hosil qilinadi (`seq + i*0.001`).
        SABAB — POYGA HOLATI: turn_worker() navbatlarni KETMA-KET qayta
        ishlaydi, lekin receiver() YANGI jismoniy bo'laklarni BUNGA
        qarashmasdan qabul qilib, `state["turn_seq"]`ni OSHIRIB borishda
        davom etadi — ya'ni shu turnning segmentatsiyasi/qayta ishlanishi
        DAVOM etayotganda, KEYINGI jismoniy bo'lak ALLAQACHON KATTAROQ
        seq olgan bo'lishi mumkin. Agar shu umumiy hisoblagichdan foydalansak,
        2-qism (hali shu ESKI bo'lakka tegishli, demak vaqt bo'yicha
        OLDINROQ) KEYINGI jismoniy bo'lakdan KATTAROQ seq olib qolishi
        mumkin edi — "vaqt tartibi" buzilardi. `seq + i*0.001` esa HECH
        QANDAY umumiy holatga bog'liq emas — har doim `seq < seq+i*0.001
        < seq+1 <= keyingi_seq` (amalda bir turn hech qachon 999 qismga
        bo'linmaydi), demak tartib HAR DOIM to'g'ri saqlanadi."""
        client_seqs = client_seqs or []
        t_seg0 = time.perf_counter()
        subsegments = await loop.run_in_executor(None, segment_turn, wav, SAMPLE_RATE, seq)
        seg_elapsed = time.perf_counter() - t_seg0
        is_multi = len(subsegments) > 1
        if is_multi:
            logger.info(f"[/ws/dialog] seq={seq} {len(subsegments)} qismga bo'lindi (seg took={seg_elapsed * 1000:.1f}ms)")

        for i, sub in enumerate(subsegments):
            sub_seq = seq if i == 0 else seq + i * 0.001
            sub_client_seqs = client_seqs if i == 0 else []
            await _process_subsegment(
                sub["wav"], sub_seq, sub_client_seqs, is_multi, sub["purity"],
                sub_index=i, sub_total=len(subsegments),
            )


    async def turn_worker():
        """DIALOG_PLAN.md 1-bosqich (D1): navbatlar endi SESSIYA BO'YICHA
        KETMA-KET, shu YAGONA worker tomonidan qayta ishlanadi. Ilgari har
        bir navbat alohida `asyncio.create_task()` bilan PARALLEL ishga
        tushirilardi — bu klaster yaratish/spiker bog'lash tartibini qaysi
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
                wav, seq, client_seqs = item
                # DIALOG_PLAN.md 4-bosqich: frontend shu statusni ko'rib
                # "Qayta ishlanmoqda…" deb ko'rsatadi (qarang
                # useDialogSession.js) — STT/tarjima bir necha soniya
                # davom etishi mumkin, foydalanuvchi shu payt hech narsa
                # ko'rinmayotganidan "ishlamayapti" deb o'ylab qolmasligi
                # uchun. process_turn() tugashi (natija "turn" yoki
                # "turn_skipped" bilan) shu holatni frontend tarafda
                # avtomatik tugatadi.
                await send_status("processing")
                await process_turn(wav, seq, client_seqs)
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
        turn_queue.put_nowait((pending["wav"], pending["seq"], pending.get("client_seqs", [])))

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

    async def handle_audio_chunk(data: bytes, reason: str | None, client_seq=None):
        # Mentor demo'da topilgan 4-muammo: frontend shu bo'lak uchun
        # o'zining `client_seq`ini yuboradi (useDialogSession.js,
        # sendTurnAudio) — pastda pending/navbat bo'yicha to'planib,
        # oxir-oqibat "turn"/"turn_skipped" bilan birga qaytariladi.
        client_seqs_for_chunk = [client_seq] if client_seq is not None else []

        # DIALOG_PLAN.md 3-bosqich (C3): endi xom Float32 PCM (frontend
        # AudioWorkletNode orqali yuboradi) — webm/ffmpeg decode shart
        # emas, shuning uchun bu sinxron, tezkor chaqiruv executor'ga
        # chiqarilmaydi (eski _decode_waveform_bytes sekin edi, chunki
        # tempfile yozib, keyin decode_audio/ffmpeg orqali o'qirdi).
        wav = _decode_pcm_bytes(data)
        if wav is None or wav.size == 0:
            logger.info("[/ws/dialog] turn skipped (empty audio)")
            await send_status("turn_skipped", reason="empty_audio", client_seqs=client_seqs_for_chunk)
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
            # Shu davomiy bo'lakning client_seq'i ham ASOSIY (pending)
            # yozuvning ro'yxatiga qo'shiladi — qarang client_seqs_for_chunk
            # izohi yuqorida.
            client_seqs = pending.get("client_seqs", []) + client_seqs_for_chunk
            logger.info(f"[/ws/dialog] merged short continuation (dur={duration_s:.2f}s) "
                f"into pending turn seq={seq} (new total dur="
                f"{merged_wav.size / SAMPLE_RATE:.2f}s)")
        else:
            merged_wav = wav
            client_seqs = client_seqs_for_chunk
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
                    # Frontend turn_meta ("action": "turn_meta", "reason": ...)
                    # matnli xabarini AYNAN shu binary chunk'dan OLDIN yuboradi
                    # (useDialogSession.js) — WebSocket bir ulanishda xabarlarni
                    # KETMA-KET saqlagani uchun, handle_client_message() bu
                    # sababni `state["pending_turn_reason"]`ga allaqachon
                    # yozib qo'ygan bo'ladi.
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
            # TEZLIK TUZATISHI: frontend har bir turn audio'sidan OLDIN shuni
            # yuboradi (useDialogSession.js'dagi sendTurnAudio) — keyingi
            # binary chunk uchun "reason" (silence/max_duration) shu yerda
            # vaqtincha saqlanadi, receiver() uni handle_audio_chunk()ga
            # uzatishdan oldin o'qib oladi. Mentor demo'da topilgan
            # 4-muammo: `client_seq` ham xuddi shunday — frontend
            # placeholder'ini keyinroq "turn"/"turn_skipped" bilan
            # topish uchun (qarang handle_audio_chunk/_resend_turn).
            state["pending_turn_reason"] = msg.get("reason")
            state["pending_turn_client_seq"] = msg.get("client_seq")

        elif action == "stop_dialog":
            await websocket.close()

        elif action == "correct_speaker" and state["phase"] == "listening":
            # Har bir bubble'dagi "spikerni almashtirish" kichik menyusi —
            # foydalanuvchi mavjud spikerlardan BIRINI yoki "Yangi spiker"ni
            # tanlaganda shu yerga keladi. `target_speaker` — mavjud spiker
            # ID (int) yoki literal "new" satri (frontend shuni yuboradi).
            # Tarjima yo'nalishi ENDI spikerga emas, FAQAT aniqlangan tilga
            # bog'liq bo'lgani uchun — bu yerda tarjima QAYTA QILINMAYDI,
            # faqat spiker yorlig'i/ovoz profili yangilanadi.
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
                # Foydalanuvchi joriy spikerning o'ziga "tuzatish" yuborgan
                # — frontend menyusi buni allaqachon ko'rsatmasligi kerak
                # (joriy spiker ro'yxatdan chiqarib tashlangan), lekin
                # himoya sifatida bu yerda ham no-op.
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
            # Har bir bubble'dagi 🗑 tugmasi — turn butunlay o'chiriladi.
            # Agar shu turn ovoz profiliga prototip QO'SHGAN bo'lsa, o'sha
            # AYNAN O'ZI (seq bo'yicha) olib tashlanadi — spikerda ENDI
            # birorta ham prototip qolmasa, spiker profilining o'zi ham
            # o'chiriladi (aks holda "hech qanday ovoz namunasiga ega
            # bo'lmagan" spiker sessiya oxirigacha xotirada osilib qolardi).
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
            # Frontend'dagi ✏️ tahrirlash — spiker ismlari FAQAT frontend
            # holatida (speaker ID -> ism xaritasi) saqlanadi, backend
            # mantig'iga (identify_speaker/klasterlash qaroriga) HECH QANDAY
            # ta'sir qilmaydi. Bu yerda FAQAT diagnostika uchun log
            # qilinadi — shu orqali [SPK] loglaridagi spiker ID'larni
            # keyinroq (masalan chegaralarni sozlashda) inson ismiga
            # bog'lash mumkin bo'ladi.
            speaker = msg.get("speaker")
            name = msg.get("name")
            logger.info(f"[/ws/dialog] rename_speaker: speaker={speaker} name={name!r}")

        elif action == "merge_speakers" and state["phase"] == "listening":
            # Frontend (DialogSessionView.jsx) foydalanuvchi ikkita spikerga
            # BIR XIL ism bergandan va "bitta odam sifatida birlashtiraymi?"
            # so'roviga "Ha" deb javob berganidan keyin yuboradi. `from` —
            # endi tahrirlangan spiker, `into` — ismi ALLAQACHON mavjud
            # bo'lgan spiker. Avtomatik birlashtirishdan (SPK_MERGE_SIM)
            # farqli — bu yerda o'xshashlik chegarasi TEKSHIRILMAYDI,
            # foydalanuvchi ANIQ tasdiqlagan.
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
