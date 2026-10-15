import difflib
import os
import time
import traceback

import numpy as np
import requests
import soundfile as sf
import torch  # noqa: F401 — ctranslate2 uchun libcublas.so.12'ni oldindan yuklaydi
from faster_whisper import WhisperModel
from faster_whisper.vad import get_speech_timestamps
from app.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    STT_PROVIDER,
    ELEVENLABS_API_KEY,
    SUPPORTED_LANGUAGES,
)
import logging

logger = logging.getLogger(__name__)


NO_SPEECH_PROB_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0
RMS_SILENCE_THRESHOLD = 0.003
USE_VAD_FILTER = True

# REAL TESTDA TOPILGAN MUAMMO (TIL FILTRIDAGI TESHIK): avvalgi shart
# `pair_mass < MIN_PAIR_MASS AND raw_top_lang not in candidates` edi — bu
# "VA" tufayli teshik qoldirgan: o'zbekcha nutq Whisper'ning xom (~99-tilli)
# taqsimotida TASODIFAN ru/en'ga yaqin tovushlanib, `raw_top_lang`NING O'ZI
# ham (noto'g'ri) candidates ICHIGA tushib qolgan hollarda, past pair_mass
# BO'LISHIGA QARAMASDAN rad etilmay, GALLYUTSINATSIYA qilingan ru/en matn
# chiqib ketgan edi. Real logda kuzatilgan: pair_mass=0.508 (raw_top=ru,
# ya'ni candidates ICHIDA) → o'tib ketdi → "Успех, я вам хочу снять
# карту..." (haqiqatda o'zbekcha gap edi). Haqiqiy EN/RU turnlarda
# pair_mass odatda 0.809-0.993 oralig'ida — demak past pair_mass O'ZI
# YETARLI, mustaqil rad etish sababi: `raw_top_lang`ning candidates'da
# bor-yo'qligidan QAT'IY NAZAR. Default 0.5 → 0.65'ga oshirildi (real
# logdagi gallyutsinatsiya holatlari 0.5-0.6 oralig'ida edi).
MIN_PAIR_MASS = float(os.environ.get("MIN_PAIR_MASS", "0.65"))

# IKKINCHI, MUSTAQIL TEShIK: pair_mass yuqori bo'lsa ham (ikki nomzod
# orasida XOM ehtimollik yig'indisi katta), ikki-tillik QAYTA NORMALLASH-
# TIRILGAN `lang_prob` (p_A yoki p_B) o'zi PAST bo'lishi mumkin — ya'ni
# ikkala nomzod ham, bir-biriga nisbatan, taxminan TENG ehtimollikda
# ko'rinadi (masalan 0.50-0.60), bu esa Whisper chindan HAQIQIY tilni
# ANIQ FARQLAY OLMAGANINI bildiradi. Haqiqiy EN/RU turnlarda ishonch
# odatda 0.82+ (Whisper) / 0.82+ (ElevenLabs language_probability). Shu
# sababli MIN_LANG_PROB'dan past bo'lsa ham mustaqil ravishda rad etiladi
# — ElevenLabs yo'lida HAM (transcribe_audio_elevenlabs'ga qarang),
# chunki u o'zining `language_probability`sini to'g'ridan-to'g'ri xuddi
# shu ma'noda ishlatadi.
MIN_LANG_PROB = float(os.environ.get("MIN_LANG_PROB", "0.75"))

# ARALASH TILLAR (A1): turn darajasidagi pair_mass (yuqoridagi ikkisi —
# MIN_PAIR_MASS/MIN_LANG_PROB — allaqachon o'tgan, ya'ni turn UMUMAN rad
# etilmagan) shu qiymatdan PAST bo'lsa, turn "aralash" (asosiy til + ichida
# begona til qismi bor) ko'rinishda hisoblanadi va HAR BIR segment uchun
# ALOHIDA til tekshiruvi ishga tushadi (qarang _segment_language_filter).
# Toza EN/RU turnlarda (pair_mass odatda 0.97+) bu qo'shimcha (GPU talab
# qiladigan) bosqich UMUMAN ishlamaydi — tezlikka ta'sir qilmaydi.
SEG_LANG_CHECK_BELOW = float(os.environ.get("SEG_LANG_CHECK_BELOW", "0.97"))
# Shundan QISQAROQ segment tekshirilmaydi (bitta so'z/atama bo'lishi mumkin
# — "gap ichidagi alohida begona so'zlar aytilganicha qolsin" talabi shu
# orqali ta'minlanadi, A2'dagi initial_prompt bilan birgalikda).
SEG_LANG_MIN_S = float(os.environ.get("SEG_LANG_MIN_S", "1.0"))
# Shundan PAST segment pair_mass'i — segment ikki sessiya tilidan ALOHIDA,
# BEGONA til (masalan o'zbekcha) deb hisoblanadi va tashlab yuboriladi.
# Turn darajasidagi MIN_PAIR_MASS'dan (0.65) PASTROQ ataylab — segment
# o'zi qisqa bo'lgani uchun Whisper'ning langID boshi tabiiy ravishda
# kamroq ishonchli, qattiqroq chegara haqiqiy (lekin qisqa) EN/RU
# segmentlarni ham noto'g'ri tashlab yuborishi mumkin edi.
SEG_MIN_PAIR_MASS = float(os.environ.get("SEG_MIN_PAIR_MASS", "0.5"))

# OpenAI Whisper'ning o'zining standart qiymati (compression_ratio_threshold=2.4)
# — segment matni g'ayrioddiy "siqiladigan" (ya'ni bir xil so'z/ibora ko'p marta
# takrorlangan) bo'lsa, bu odatda gallyutsinatsiya/stuck-loop belgisi (REPETITION_
# PENALTY/NO_REPEAT_NGRAM_SIZE pastda shu holatni KAMAYTIRADI, lekin har doim
# oldini olmaydi — shu metrika qolgan holatlarni ushlaydi).
COMPRESSION_RATIO_THRESHOLD = 2.4

# DIALOG_PLAN.md 1-bosqich: Whisper (ayniqsa rus/koreys tilida ko'p o'rgatilgan
# YouTube subtitri/outro matnlaridan) past-energiya/nojonli audio'da HAQIQIY
# nutqqa aloqasi yo'q, lekin lingvistik jihatdan "to'g'ri" ko'ringan matn
# generatsiya qilishi ma'lum — no_speech_prob/avg_logprob/compression_ratio
# buni har doim ushlay olmaydi (matn o'zi "ishonchli" talaffuz qilingan bo'lishi
# mumkin). Shu sababli qo'shimcha, matn-darajasidagi qora ro'yxat.
#
# MUHIM — har bir yozuv TO'LIQ, aniq boilerplate IBORA, bitta umumiy KALIT
# SO'Z emas: real testda "субтитры" so'zining o'zi (fragment sifatida)
# ro'yxatda turgani uchun "Для вас есть субтитры, как обычно, здесь..." va
# "Где можно включить субтитры" kabi HAQIQIY, mazmunli ruscha gaplar ham
# substring moslik orqali noto'g'ri fantom deb topilib, butunlay
# bo'shatib yuborilgan edi (ikkisi ham "субтитры" so'zini tabiiy
# ishlatgan, lekin Whisper'ning klassik "Субтитры создавал DimaTorzok"
# gallyutsinatsiyasi EMAS). Shu sababli solishtirish endi substring
# (`in`) emas, difflib.SequenceMatcher orqali FUZZY-RATIO — matn fantom
# ibora bilan DEYARLI AYNAN bir xil (normallashtirilgandan keyin)
# bo'lgandagina fantom deb hisoblanadi. Uzun, mazmunli gap (hatto fantom
# so'zni o'z ichiga olsa ham) qisqa boilerplate iboradan sezilarli uzunroq
# bo'lgani uchun ratio avtomatik past chiqadi — maxsus uzunlik tekshiruvi
# kerak emas.
PHANTOM_PHRASES = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to my channel",
    "subtitles by the amara.org community",
    "субтитры создавал dimatorzok",
    "редактор субтитров а.семкин",
    "корректор а.egorova",
    "продолжение следует",
]

# >= shu qiymatda ikki matn "deyarli bir xil" deb hisoblanadi (difflib
# ratio: 1.0 — aynan bir xil, 0.0 — umuman mos emas). 0.85 — qisqa
# boilerplate'ning kichik imlo/tinish belgisi farqlarini (masalan
# "DimaTorzok" turlicha yozilishi) hali ham ushlaydi, lekin uzunroq,
# boshqacha mazmunli gaplarni ushlamaydi (pastga, PHANTOM_PHRASES
# izohiga qarang).
PHANTOM_PHRASE_FUZZY_THRESHOLD = 0.85


def _is_phantom_phrase(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    return any(
        difflib.SequenceMatcher(None, normalized, phrase).ratio() >= PHANTOM_PHRASE_FUZZY_THRESHOLD
        for phrase in PHANTOM_PHRASES
    )

# TEZLIK TUZATISHI (maqsad: gap tugashi -> tarjima ~2s): ILGARI
# TRANSCRIBE_TEMPERATURE=(0.0, 0.4) edi — temp=0.0 compression_ratio/logprob
# mezonlaridan o'tmasa, faster-whisper BITTA marta temp=0.4 bilan QAYTA DECODE
# qilardi (butun segmentni ikkinchi marta, to'liq qayta hisoblash — sezilarli
# kechikish manbai). Endi FALLBACK BUTUNLAY O'CHIRILGAN (bitta temperature,
# qayta urinishsiz) va beam_size=1 (greedy decode, beam_size>1'dan tezroq) —
# ikkisi ham .env orqali sozlanadi, agar aniqlik (WER) sezilarli yomonlashsa
# qaytarish uchun.
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))
WHISPER_TEMPERATURE = float(os.environ.get("WHISPER_TEMPERATURE", "0.0"))
WHISPER_CONDITION_ON_PREVIOUS_TEXT = (
    os.environ.get("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false").strip().lower() == "true"
)

# ARALASH TILLAR (A2): "begona so'zlarni aytilganicha saqlash". Whisper'ga
# decode TILI ALLAQACHON ma'lum bo'lganda (detected_lang — restricted
# candidate yo'lida, yoki aniq `language` berilganda) shu tilga mos
# initial_prompt beriladi — bu Whisper'ga "asosiy til shu, lekin ichida
# boshqa til so'zlari bo'lishi mumkin, ularni TARJIMA qilmasdan, AYTILGANICHA
# yoz" deb "ko'rsatma" beradi (initial_prompt — Whisper decode'ni majburlamaydi,
# faqat KONTEKST sifatida ta'sir qiladi). .env orqali O'CHIRISH mumkin —
# initial_prompt ba'zan (ayniqsa qisqa/notinch audio'da) gallyutsinatsiyani
# KUCHAYTIRISHI ham mumkin, shuning uchun A/B solishtirish uchun tezkor
# yoqish/o'chirish kaliti sifatida qoldirildi.
WHISPER_INITIAL_PROMPT_ENABLED = (
    os.environ.get("WHISPER_INITIAL_PROMPT_ENABLED", "true").strip().lower() == "true"
)
WHISPER_INITIAL_PROMPTS = {
    "en": os.environ.get(
        "WHISPER_INITIAL_PROMPT_EN",
        "The speaker mainly speaks English but may use some Russian or Uzbek words. "
        "Write those words exactly as spoken, without translating them.",
    ),
    "ru": os.environ.get(
        "WHISPER_INITIAL_PROMPT_RU",
        "Говорящий в основном говорит по-русски, но может использовать отдельные "
        "английские или узбекские слова. Записывай их так, как они произнесены, без перевода.",
    ),
}

# Uzun (10-13s) kesmalarda Whisper ba'zan BIR XIL jumlani 10-20 marta ketma-ket
# takrorlab chiqarishi (decode "tutilib qolishi") kuzatildi — condition_on_previous_text
# allaqachon False (pastga qarang), lekin bu yolgiz bu turdagi takrorlanishni to'xtatmadi.
# repetition_penalty>1 oldingi tokenlarni qayta tanlashni qimmatlashtiradi,
# no_repeat_ngram_size esa N so'zlik ketma-ketlikni so'zma-so'z qaytarilishini
# taqiqlaydi — ikkisi birga shu "stuck loop" gallyutsinatsiyasini kamaytiradi.
REPETITION_PENALTY = 1.3
NO_REPEAT_NGRAM_SIZE = 3

# TEZLIK TUZATISHI (4-band): butunlay shovqin/jimlik bo'lgan navbatlar ham
# (til-aniqlash + to'liq decode bosqichigacha borib) GPU vaqtini sarflardi —
# real testda 5 navbatdan 3 tasi shovqin bo'lib chiqgan, har biri 0.7-2.2s
# GPU vaqti olib, navbatdagi HAQIQIY keyingi gapni kuttirgan (bitta navbat,
# ketma-ket worker — dialog.py). Endi Silero VAD (faster_whisper.vad, allaqachon
# onnxruntime orqali ishlaydi — pastdagi _ONNX_AVAILABLE) orqali ENG BOSHIDA,
# tilni aniqlashdan OLDIN, haqiqiy NUTQ necha soniya ekani tez hisoblanadi —
# agar shu qiymat MIN_SPEECH_S'dan kam bo'lsa, STT butunlay chaqirilmaydi.
MIN_SPEECH_S = float(os.environ.get("MIN_SPEECH_S", "0.3"))

# SHOVQINLI MUHIT TUZATISHI: mikrofonga ENG YAQIN turgan odam (deyarli
# doim) turn ichidagi ENG BALAND RMS'li Whisper segmentiga ega bo'ladi —
# signal kuchi masofa kvadratiga teskari proporsional kamayadi. Shu
# sababli atrofdagi odamlar (mikrofondan uzoqroq) gapirgan segmentlar,
# frontend'dagi near-field VAD (useDialogSession.js, NEAR_FIELD_RATIO)
# ulardan ba'zilarini o'zib ketishi mumkin bo'lsa ham, bu yerda IKKINCHI,
# mustaqil himoya qatlami sifatida yana tekshiriladi: segment RMS'i shu
# TURN ichidagi eng baland segment RMS'ining SEGMENT_REL_RMS ulushidan
# past bo'lsa — "fon" (far_field) deb hisoblanib, STT natijasiga umuman
# kiritilmaydi (_transcribe_audio_whisper'ga qarang). Bitta-segmentli
# turnlarda max_rms = segmentning o'zi, demak hech qachon filtrlanmaydi —
# shu sababli ODDIY (fon shovqinsiz) suhbatda xatti-harakat o'zgarmaydi.
# .env orqali sozlanishi mumkin (xona akustikasi/mikrofon joylashuviga
# qarab qayta deploy qilmasdan moslashtirish uchun).
SEGMENT_REL_RMS = float(os.environ.get("SEGMENT_REL_RMS", "0.3"))

DEBUG_STT = False

SAMPLE_RATE = 16000

# ElevenLabs Scribe v2 ("Create transcript") — Whisper'ga MUQOBIL (ikkinchi)
# STT provayder, config.STT_PROVIDER="elevenlabs" bo'lsa ishlatiladi.
# Rasmiy Python SDK o'rniga to'g'ridan-to'g'ri REST so'rov (requests)
# ishlatildi — loyihada `requests` allaqachon (transitive dependency
# sifatida) mavjud, alohida SDK qo'shish shart emas.
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_MODEL_ID = "scribe_v2"
ELEVENLABS_TIMEOUT_S = 30

# ElevenLabs tilni UCH harfli ISO 639-3 kod bilan qaytaradi (masalan "eng",
# "rus", "kor"), lekin tizimning qolgan qismi (dialog.py'dagi SESSION_LANGS,
# config.SUPPORTED_LANGUAGES) Whisper qaytaradigan IKKI harfli ISO 639-1
# kodni kutadi ("en", "ru", "ko") — shu moslik uchun. Faqat SUPPORTED_
# LANGUAGES'dagi tillar kifoya: boshqa (xaritalanmagan) kod ham pastdagi
# SUPPORTED_LANGUAGES cheklovi orqali baribir to'g'ri (past ishonch bilan)
# rad etiladi, shuning uchun to'liq ISO 639-3 lug'ati kerak emas.
ELEVENLABS_LANG_CODE_MAP = {
    "eng": "en",
    "rus": "ru",
    "kor": "ko",
}

_model = None


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  

        return True
    except Exception:
        return False


_ONNX_AVAILABLE = _onnxruntime_available()


def get_model():
    global _model
    if _model is None:
        # Loaded once per process 
        t0 = time.perf_counter()
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        logger.info(f"[STT] Whisper model loaded in {time.perf_counter() - t0:.2f}s "
            f"(size={WHISPER_MODEL_SIZE!r}, device={WHISPER_DEVICE!r}, "
            f"compute_type={WHISPER_COMPUTE_TYPE!r})")
        if USE_VAD_FILTER and not _ONNX_AVAILABLE:
            logger.info("[STT] vad_filter requested but onnxruntime is not installed; "
                "running without VAD")
    return _model


def warmup(run_inference: bool = True) -> None:
    model = get_model()
    if not run_inference:
        return
    try:
        segments, _ = model.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            language="en",
            vad_filter=USE_VAD_FILTER and _ONNX_AVAILABLE,
            condition_on_previous_text=False,
        )
        for _ in segments:  
            pass
        logger.info("[STT] warm-up inference done")
    except Exception as e: 
        logger.info(f"[STT] warm-up inference skipped: {e!r}")


def _decode_waveform(audio_path: str):
    try:
        from faster_whisper.audio import decode_audio

        return decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
    except Exception:
        try:
            import librosa

            wav, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            return wav.astype(np.float32)
        except Exception as e:
            logger.info(f"[STT] could not pre-decode audio for RMS check: {e!r}")
            return None


def _decode_waveform_bytes(chunk: bytes):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
        tmp.write(chunk)
        tmp.flush()
        return _decode_waveform(tmp.name)


def _decode_pcm_bytes(chunk: bytes) -> np.ndarray:
    """DIALOG_PLAN.md 3-bosqich (C3): dialog.py endi WebM/MediaRecorder
    o'rniga, frontend AudioWorkletNode orqali uzluksiz ushlagan XOM 16kHz
    mono Float32 PCM'ni to'g'ridan-to'g'ri qabul qiladi (qarang
    useDialogSession.js) — konteyner yo'q, ffmpeg/decode_audio orqali
    decode qilish SHART EMAS, shuning uchun bu `_decode_waveform_bytes`dan
    (webm, tempfile+decode_audio talab qiladi) ANCHA tezroq va soddaroq.

    Bayt tartibi: JS'dagi Float32Array'ning xom ArrayBuffer'i haqiqiy
    maqsadli platformalarning BARCHASIDA (x86/ARM) kichik-endian — bu esa
    numpy'ning native `float32`siga (shu platformalarda ham kichik-endian)
    mos keladi, shuning uchun aniq '<f4' belgilash shart emas.
    `np.frombuffer` natijasi FAQAT O'QISH uchun (read-only) view qaytaradi
    — `.copy()` shuni yozilishi mumkin bo'lgan, mustaqil massivga aylantiradi."""
    return np.frombuffer(chunk, dtype=np.float32).copy()


def _write_wav(path: str, wav: np.ndarray) -> None:
    sf.write(path, wav, SAMPLE_RATE)


def _run_transcribe(model, source, language: str | None, use_vad: bool, initial_prompt: str | None = None):
    """model.transcribe() ni chaqiradi; vad_filter xato bersa (masalan qisqa
    audio'da faster-whisper'ning ichki til aniqlash kodi bo'sh to'plamda max()
    chaqirishi mumkin) — VAD'siz BITTA marta qayta uriniladi.

    `initial_prompt` — A2 (begona so'zlarni aytilganicha saqlash): chaqiruvchi
    (`_transcribe_audio_whisper`) tomonidan, decode tili ALLAQACHON ma'lum
    bo'lganda WHISPER_INITIAL_PROMPTS'dan tanlab beriladi (qarang yuqoridagi
    izoh) — WHISPER_INITIAL_PROMPT_ENABLED=false bo'lsa yoki til noma'lum
    bo'lsa, `None` sifatida uzatiladi (Whisper'ning standart xatti-harakati)."""
    t0 = time.perf_counter()
    try:
        segments, info = model.transcribe(
            source,
            language=language,
            vad_filter=use_vad,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            temperature=WHISPER_TEMPERATURE,
            beam_size=WHISPER_BEAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            initial_prompt=initial_prompt,
        )
        return list(segments), info
    except Exception as e:
        first_attempt_elapsed = time.perf_counter() - t0
        if not use_vad:
            raise
        logger.error(f"[STT][VAD_RETRY] vad_filter failed after {first_attempt_elapsed:.2f}s "
            f"(language={language!r}, exception={e!r}); retrying without VAD\n"
            f"{traceback.format_exc()}")
        t_retry0 = time.perf_counter()
        segments, info = model.transcribe(
            source,
            language=language,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            temperature=WHISPER_TEMPERATURE,
            beam_size=WHISPER_BEAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            initial_prompt=initial_prompt,
        )
        seg_list = list(segments)
        logger.info(f"[STT][VAD_RETRY] retry (no VAD) took {time.perf_counter() - t_retry0:.2f}s "
            f"(first attempt had already spent {first_attempt_elapsed:.2f}s)")
        return seg_list, info


def _detect_language_restricted(
    model, audio: np.ndarray, candidates: list[str]
) -> tuple[str, float, dict]:
    """Whisper'ning o'zining til-aniqlash (language-ID) boshini ishlatadi va
    faqat berilgan nomzod tillar orasidan eng ehtimolini tanlaydi.

    DIALOG_PLAN.md 1-bosqich: dialog rejimida `candidates` HAR DOIM aynan
    sessiyaning ikki tili ([langA, langB]) — shu sababli bu endi "qaysi til
    to'g'ri" emas, balki "shu IKKISIDAN qaysi biri" savoli. Ehtimollik shu
    ikkisi orasida QAYTA NORMALLASHTIRILADI: p_A = P(A) / (P(A) + P(B)).
    Buning fo'ydasi: xom ~99-tilli taqsimotdagi qiymatlar real mikrofon
    audiosida to'g'ri til uchun ham past/beqaror chiqadi (masalan aniq
    inglizcha gap uchun 0.2-0.5 kabi) — ikkita nomzod orasidagi NISBIY ulush
    esa ancha barqaror va to'g'ridan-to'g'ri foydali signal beradi.

    ESKI xatti-harakat (olib tashlandi): agar to'liq ~99-tilli taqsimotning
    GLOBAL argmaxi candidates'dan tashqarida bo'lsa, ishonch 0.0'ga
    tushirilardi. Bu endi KERAK EMAS — ikki-tillik renormalizatsiya o'zi
    shu holatni to'g'ri aks ettiradi (agar haqiqiy til uchinchi bo'lsa,
    P(A) va P(B) ikkisi ham past bo'ladi, demak p_A 0.5'ga yaqin chiqadi —
    "ishonchsiz" degani, aynan shunday bo'lishi kerak ham, mutlaqo 0.0'ga
    majburlash shart emas).

    MUHIM: bu funksiya HECH QACHON None qaytarmaydi — doim `candidates`dan
    BITTASINI tanlab qaytaradi, chunki qaytarilgan til chaqiruvchida
    (`transcribe_audio`) HAQIQIY decode bosqichiga `language=` sifatida
    to'g'ridan-to'g'ri uzatiladi — decode har doim candidates bilan
    CHEKLANGAN bo'lib qolishi kerak (aks holda matn candidates'dan butunlay
    tashqari tilda chiqib ketishi mumkin — real testda kuzatilgan).

    REAL TESTDA TOPILGAN MUAMMO: ikki-tillik normallashtirish (p_A =
    P(A)/(P(A)+P(B))) o'zi bo'yicha UCHINCHI tilni (masalan sessiyada
    tanlanmagan o'zbekcha) HECH QACHON ko'ra olmaydi — agar haqiqiy til
    uchinchi bo'lsa ham, P(A) va P(B) ORASIDA nisbiy ulush baribir
    hisoblanadi va ba'zan tasodifan "ishonchli" (masalan 0.97) chiqib
    ketadi. Shu sababli endi XOM (candidates bilan cheklanmagan) taqsimot
    ham qaytariladi — chaqiruvchi (`_transcribe_audio_whisper`) shu orqali
    "pair_mass" (ikki nomzodning XOM ehtimolliklar yig'indisi) va haqiqiy
    global eng yuqori tilni (`raw_top_lang`) alohida tekshirib, agar
    ikkisi mos kelmasa va pair_mass past bo'lsa, turnni "unsupported_
    language" deb belgilashi mumkin."""
    features = model.feature_extractor(audio)
    segment = features[:, : model.feature_extractor.nb_max_frames]
    encoder_output = model.encode(segment)
    results = model.model.detect_language(encoder_output)[0]
    all_probs = {token[2:-2]: prob for token, prob in results}
    cand_probs = {lang: all_probs.get(lang, 0.0) for lang in candidates}

    if not all_probs:
        diagnostics = {"pair_mass": 0.0, "raw_top_lang": None, "raw_top_prob": 0.0, "top5": []}
        return candidates[0], 1.0 / len(candidates), diagnostics

    sorted_all = sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True)
    raw_top_lang, raw_top_prob = sorted_all[0]
    diagnostics = {
        "pair_mass": sum(cand_probs.values()),
        "raw_top_lang": raw_top_lang,
        "raw_top_prob": raw_top_prob,
        "top5": sorted_all[:5],
    }

    best_lang = max(cand_probs, key=cand_probs.get)
    total = diagnostics["pair_mass"]
    if total <= 0:
        # Ikkala nomzod ham modelning (odatda qisqartirilgan, to'liq 99
        # tilni qamramaydigan) chiqishida umuman yo'q — teng taqsimlab,
        # chaqiruvchiga past ishonch signalini qoldiramiz.
        return best_lang, 1.0 / len(candidates), diagnostics

    return best_lang, cand_probs[best_lang] / total, diagnostics


def _segment_language_filter(
    model, audio: np.ndarray, seg_list, candidates: list[str]
) -> tuple[list[bool], list[dict | None]]:
    """A1 — ARALASH TILLAR: turn darajasidagi pair_mass yetarli past bo'lib
    (qarang SEG_LANG_CHECK_BELOW) turn "aralash" ko'ringanda chaqiriladi —
    HAR BIR segment uchun ALOHIDA, shu segmentning O'Z audio qismi bo'yicha
    til ehtimolligi hisoblanadi (xuddi _detect_language_restricted kabi,
    lekin butun turn emas, bitta segment uchun).

    SEG_LANG_MIN_S'dan QISQA segmentlar TEKSHIRILMAYDI (bitta so'z/atama
    bo'lishi mumkin — "gap ichidagi alohida begona so'zlar aytilganicha
    qolsin" talabi shu orqali, A2'dagi initial_prompt bilan birgalikda,
    ta'minlanadi).

    Qaytaradi: (foreign_flags, diagnostics) — ikkisi ham `seg_list` bilan
    BIR XIL uzunlik/tartibda. `foreign_flags[i] = True` — shu segment
    pair_mass'i SEG_MIN_PAIR_MASS'dan past, ya'ni ikki sessiya tilidan
    ALOHIDA, BEGONA til (masalan uzun o'zbekcha bo'lak) bo'lib chiqdi
    degani — chaqiruvchi (`_transcribe_audio_whisper`dagi asosiy filtrlash
    tsikli) buni "foreign_segment" sababi bilan tashlab yuboradi."""
    foreign = [False] * len(seg_list)
    diagnostics: list[dict | None] = [None] * len(seg_list)
    for i, seg in enumerate(seg_list):
        seg_start_s = getattr(seg, "start", None)
        seg_end_s = getattr(seg, "end", None)
        if seg_start_s is None or seg_end_s is None:
            continue
        if seg_end_s - seg_start_s < SEG_LANG_MIN_S:
            continue
        start_idx = max(0, int(seg_start_s * SAMPLE_RATE))
        end_idx = min(audio.size, int(seg_end_s * SAMPLE_RATE))
        if end_idx <= start_idx:
            continue
        seg_audio = audio[start_idx:end_idx]
        _, _, seg_diag = _detect_language_restricted(model, seg_audio, candidates)
        diagnostics[i] = seg_diag
        if seg_diag["pair_mass"] < SEG_MIN_PAIR_MASS:
            foreign[i] = True
    return foreign, diagnostics


def _fast_vad_speech_duration(audio: np.ndarray) -> float:
    """Silero VAD (faster_whisper.vad.get_speech_timestamps) orqali audio'dagi
    JAMI nutq davomiyligini (soniyada) hisoblaydi — model.encode()/transcribe()
    kabi og'ir, GPU-bog'liq bosqichlarga umuman kirmasdan, faqat kichik VAD
    modeli (onnxruntime, CPU) orqali. `transcribe_audio()`dagi tezkor
    "bo'sh/shovqin" tekshiruvi uchun ishlatiladi (MIN_SPEECH_S bilan
    solishtiriladi). Xato bo'lsa (masalan model fayli topilmasa), xavfsiz
    tomonga — cheksiz ("yetarli nutq bor", oddiy yo'lga o'tkazadi) qaytariladi."""
    try:
        segments = get_speech_timestamps(audio)
    except Exception as e:
        logger.info(f"[STT][VAD] get_speech_timestamps failed: {e!r} — tekshiruv o'tkazib yuborildi")
        return float("inf")
    return sum(seg["end"] - seg["start"] for seg in segments) / SAMPLE_RATE


def transcribe_audio(
    audio_path: str,
    language: str | None = None,
    candidate_languages: list[str] | None = None,
) -> tuple[str, str | None, float, bool, dict]:
    """config.STT_PROVIDER'ga qarab Whisper (lokal) yoki ElevenLabs Scribe v2
    (API) orqali transkripsiya qiladi — har doim BIR XIL (text, detected_lang,
    lang_prob, unsupported_language, stt_diag) formatini qaytaradi, shuning
    uchun dialog.py/speaker_diart.py provayderdan QAT'IY NAZAR o'zgarishsiz
    ishlaydi. `stt_diag` — {"seg_lang_check": bool, "seg_lang_check_ms": float}
    (A1, qarang _segment_language_filter) — ElevenLabs yo'lida va segment
    tekshiruvi ishlamagan hollarda doim {"seg_lang_check": False,
    "seg_lang_check_ms": 0.0}, dialog.py buni [TURN] logiga qo'shadi.
    `unsupported_language=True` — `candidate_languages` berilgan (Dialog
    rejimi) VA haqiqiy nutq ikkalasidan ham BOSHQA tilda bo'lgani (yetarlicha
    ishonch bilan) aniqlangan holatda; chaqiruvchi bu holda speaker/tarjima
    mantig'iga umuman kirmasligi kerak (qarang dialog.py).

    `candidate_languages` (dialog rejimida — sessiyaning aynan ikki tili,
    [langA, langB]) ikki provayderda IKKI XIL ishlatiladi, chunki ElevenLabs
    API'si Whisper'dan farqli ravishda bitta ro'yxatni emas, bitta "hint"ni
    qabul qiladi (pastga, transcribe_audio_elevenlabs izohiga qarang):
      - Whisper: _transcribe_audio_whisper() ICHIDA, _detect_language_restricted()
        orqali decode OLDINDAN shu ro'yxat bilan cheklanadi va ehtimollik shu
        ikkisi orasida normallashtiriladi.
      - ElevenLabs: so'rov OLDINDAN cheklanmaydi (bitta tilni "hint" qilib
        bo'lmaydi — qaysi ikkisidan biri ekani oldindan noma'lum), natija
        QAYTGANDAN KEYIN shu ro'yxatga qarshi tekshiriladi.

    TEZLIK TUZATISHI (4-band): provayderdan QAT'IY NAZAR (Whisper HAM,
    ElevenLabs HAM) — tilni aniqlash/API-chaqiruvga o'tishdan OLDIN, tez
    (faqat Silero VAD, GPU/tarmoq ishlatmaydi) nutq-davomiyligi tekshiruvi
    o'tkaziladi. Agar haqiqiy nutq MIN_SPEECH_S'dan kam bo'lsa (butun kesma
    shovqin/jimlik), darhol bo'sh natija qaytariladi — qimmat bosqichlarga
    umuman o'tilmaydi."""
    if _ONNX_AVAILABLE:
        audio_for_vad = _decode_waveform(audio_path)
        if audio_for_vad is not None and audio_for_vad.size:
            t_vad0 = time.perf_counter()
            speech_s = _fast_vad_speech_duration(audio_for_vad)
            if speech_s < MIN_SPEECH_S:
                logger.info(
                    f"[STT][VAD] speech={speech_s:.2f}s — skipped in "
                    f"{(time.perf_counter() - t_vad0) * 1000:.0f}ms"
                )
                return "", None, 0.0, False, {"seg_lang_check": False, "seg_lang_check_ms": 0.0}

    if STT_PROVIDER == "elevenlabs":
        return transcribe_audio_elevenlabs(audio_path, language_hint=language, candidate_languages=candidate_languages)
    return _transcribe_audio_whisper(audio_path, language, candidate_languages)


def transcribe_audio_elevenlabs(
    audio_path: str,
    language_hint: str | None = None,
    candidate_languages: list[str] | None = None,
) -> tuple[str, str | None, float, bool, dict]:
    """ElevenLabs Scribe v2 ("Create transcript", POST /v1/speech-to-text)
    orqali transkripsiya qiladi. Natija transcribe_audio() bilan BIR XIL
    (text, detected_lang, lang_prob, unsupported_language, stt_diag) formatiga
    moslashtiriladi.

    A3 — ARALASH TILLAR, ElevenLabs yo'li: rasmiy hujjat (SpeechToTextWordResponseModel,
    elevenlabs.io/docs/api-reference/speech-to-text) tekshirildi — `words`
    massividagi har bir elementda FAQAT `text`/`type`/`logprob`/`start`/`end`/
    `speaker_id`/`channel_index` bor, SO'Z DARAJASIDA til kodi YO'Q. Til
    (`language_code`/`language_probability`) FAQAT BUTUN transkript/chunk
    uchun BIR MARTA qaytariladi. Demak A1'dagi (Whisper yo'li) segment-
    darajasidagi til-tekshiruv mantig'ini bu yo'lda TAKRORLASH IMKONSIZ —
    shuning uchun bu yerga QO'SHILMAGAN, `stt_diag` doim
    {"seg_lang_check": False, "seg_lang_check_ms": 0.0} qaytaradi.

    ESLATMA (2-band, unsupported_language): ElevenLabs API'si Whisper'ning
    language-ID boshi kabi TO'LIQ (~99 tilli) XOM taqsimot qaytarmaydi —
    faqat BITTA (o'zi "eng ishonchli" deb tanlagan) til + unga tegishli
    `language_probability`. Shu sababli Whisper yo'lidagi kabi "pair_mass"/
    "raw_top5" hisoblash BU YERDA matematik jihatdan mumkin emas. Buning
    o'rniga ElevenLabs'ning o'z eng-yuqori tanlovi (`detected_lang`) to'g'ridan-
    to'g'ri "raw_top_lang" vazifasini bajaradi deb qaraladi: agar u ikki
    sessiya tilidan birortasiga ham to'g'ri kelmasa — bu allaqachon ikkisidan
    "tashqarida" degani, shuning uchun `unsupported_language=True` qilib
    qaytariladi (pastga qarang).

    Tarmoq xatosi, limit tugashi, API kaliti sozlanmagani yoki boshqa HTTP
    xatoligi holatida tizim QULAMAYDI — xatolik log qilinadi va bo'sh
    natija ("", None, 0.0, False) qaytariladi. Chaqiruvchi tomon (dialog.py'dagi
    `empty_transcription` tekshiruvi, speaker_diart.py'dagi founding-speech
    tekshiruvi) buni allaqachon bo'sh transkripsiya sifatida to'g'ri
    boshqaradi — xatolik va "haqiqatan ham jim/shovqin" holati bir xil
    (xavfsiz) natija bilan yakunlanadi."""
    empty_diag = {"seg_lang_check": False, "seg_lang_check_ms": 0.0}

    if not ELEVENLABS_API_KEY:
        logger.info("[STT][ELEVENLABS] ELEVENLABS_API_KEY sozlanmagan (.env'ni tekshiring) — "
            "bo'sh natija qaytarilyapti")
        return "", None, 0.0, False, empty_diag

    # Whisper yo'lidagi kabi: butunlay jim/shovqin audio uchun tarmoq
    # so'rovini umuman yubormaslik (API chaqiruvini/kreditni tejash uchun).
    audio = _decode_waveform(audio_path)
    if audio is not None and audio.size:
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
        if rms < RMS_SILENCE_THRESHOLD:
            logger.info(f"[STT][ELEVENLABS] audio below RMS threshold (rms={rms:.5f} < "
                f"{RMS_SILENCE_THRESHOLD}) — skipping transcription")
            return "", None, 0.0, False, empty_diag

    t0 = time.perf_counter()
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            data = {"model_id": ELEVENLABS_MODEL_ID}
            if language_hint:
                data["language_code"] = language_hint
            response = requests.post(
                ELEVENLABS_STT_URL,
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                files=files,
                data=data,
                timeout=ELEVENLABS_TIMEOUT_S,
            )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        logger.info(f"[STT][ELEVENLABS] so'rov muvaffaqiyatsiz bo'ldi: {e!r} — bo'sh natija "
            f"qaytarilyapti (tizim davom etadi)")
        return "", None, 0.0, False, empty_diag

    # SHOVQINLI MUHIT TUZATISHI: Whisper yo'lida (_transcribe_audio_whisper)
    # qo'llanilgan segment-darajasidagi far_field filtri BU YERDA (hozircha)
    # QO'LLANILMAGAN. Sabab: hozirgi so'rov (`data` yuqorida) faqat
    # "model_id"/"language_code" yuboradi — ElevenLabs'dan so'z-darajasidagi
    # vaqt belgilari (word-level timestamps) UMUMAN SO'RALMAYDI, `result`
    # faqat yakuniy, BUTUN turn uchun bitta `text` qaytaradi (segment/so'z
    # chegaralari yo'q). Shu sababli "turn ichidagi eng baland segment"ga
    # nisbatan solishtirish BU YO'LDA matematik jihatdan mumkin emas — buni
    # amalga oshirish uchun so'rovga vaqt-belgilari parametri (ElevenLabs
    # Scribe API hujjatiga qarang, masalan "timestamps_granularity": "word")
    # qo'shish va `result`dan qaytgan har bir so'z/segment uchun start/end
    # vaqt oraliqlarini ajratib olish kerak bo'ladi — bu ishning ko'lamidan
    # tashqarida qoldirildi (hozircha loyihada STT_PROVIDER="whisper" default,
    # ElevenLabs yo'li faqat qo'shimcha/muqobil sifatida mavjud).
    text = (result.get("text") or "").strip()
    raw_lang_code = result.get("language_code") or None
    # ElevenLabs'ning UCH harfli (ISO 639-3) kodini Whisper formatidagi IKKI
    # harfli (ISO 639-1) kodga aylantiramiz — qolgan tizim (dialog.py'dagi
    # SESSION_LANGS, NLLB_LANG_CODES) shu formatni kutadi (qarang
    # ELEVENLABS_LANG_CODE_MAP izohi).
    detected_lang = ELEVENLABS_LANG_CODE_MAP.get(raw_lang_code, raw_lang_code) if raw_lang_code else None
    lang_prob = float(result.get("language_probability") or 0.0)

    # DIALOG_PLAN.md 1-bosqich (B1/B4), ElevenLabs uchun moslashtirilgan:
    # Whisper'da (_detect_language_restricted) decode'ning O'ZI oldindan
    # candidate_languages bilan cheklanadi va ehtimollik shu ikkisi orasida
    # QAYTA NORMALLASHTIRILADI (p_A = P(A)/(P(A)+P(B))). ElevenLabs API'si
    # buni OLDINDAN qila olmaydi — `language_code` so'rov parametri BITTA
    # tilni hint/majburlash uchun, Whisper'ning candidate RO'YXATI kabi emas,
    # va qaysi ikkisidan biri gapirilayotgani OLDINDAN noma'lum (shu sababli
    # bitta tilni "hint" qilib bo'lmaydi — ikkalasini ham hint qilish
    # mumkin emas). ElevenLabs'ning o'zi TO'LIQ til ro'yxati bo'yicha
    # avtomatik aniqlaydi va natija allaqachon BITTA tanlangan til +
    # UNGA tegishli `language_probability` ko'rinishida qaytadi — Whisper'dan
    # farqli, bu yerda ikkita nomzod orasidagi alohida-alohida ehtimollik
    # yo'q (faqat bitta, ElevenLabs tanlagan til uchun), shuning uchun
    # Whisper'dagi kabi ikki-tillik QAYTA NORMALLASHTIRISH shu yerda
    # matematik jihatdan mumkin emas (faqat bitta nuqtali baho bor).
    #
    # Shu sababli post-hoc qaror: natija QAYTGANDAN KEYIN tekshiriladi —
    # agar `detected_lang` candidate_languages (sessiyaning ikki tili)
    # ICHIDA bo'lsa, ElevenLabs'ning o'z `language_probability`'si TO'G'RIDAN
    # TO'G'RI ishonch sifatida ishlatiladi (ElevenLabs — buning uchun maxsus
    # o'qitilgan model, Whisper'ning yon-mahsulot langID boshidan ko'ra
    # ishonchliroq). Agar `detected_lang` IKKALASIGA ham mos kelmasa (masalan
    # real logda kuzatilgan "por" kabi uchinchi til) — bu A/B'dan qaysi biri
    # ekanini ElevenLabs'ning o'zi ayta olmaydi, shuning uchun ishonch 0.0'ga
    # tushiriladi (xuddi Whisper yo'lidagi kabi) va qaror pastroq oqimdagi
    # mavjud mexanizmlarga (identify_speaker'ning til+ovoz birlashgan
    # balli, NLLB_LANG_CODES'da yo'q tilni tarjima qilmasdan asl matnni
    # ko'rsatish) qoldiriladi. `candidate_languages` berilmagan bo'lsa (masalan
    # live_transcribe.py'dagi bitta-tilli chaqiruv), eski, kengroq
    # SUPPORTED_LANGUAGES chegarasiga qaytiladi.
    valid_langs = candidate_languages if candidate_languages else SUPPORTED_LANGUAGES
    # 2-band: ElevenLabs'ning o'z eng-yuqori tanlovi (`detected_lang`) bu
    # yerda Whisper'dagi `raw_top_lang`ning vazifasini bajaradi — agar u
    # sessiyaning ikki tilidan (candidate_languages berilgan bo'lsa)
    # BIRIGA ham to'g'ri kelmasa, bu "unsupported_language" (pair_mass
    # hisoblanmaydi — izohga qarang, ElevenLabs to'liq taqsimot bermaydi).
    unsupported_language = False
    if detected_lang not in valid_langs:
        logger.info(f"[STT][ELEVENLABS] detected_lang={detected_lang!r} (raw={raw_lang_code!r}) "
            f"{valid_langs}'dan tashqarida — ishonch 0.0'ga tushirildi")
        lang_prob = 0.0
        if candidate_languages:
            unsupported_language = True
            logger.info(
                f"[STT][LANG][ELEVENLABS] unsupported_language reason=out_of_pair detected_lang={detected_lang!r} "
                f"candidate_languages={candidate_languages} ichida yo'q — turn tashlab yuborildi"
            )
    elif candidate_languages and lang_prob < MIN_LANG_PROB:
        # REAL TESTDA TOPILGAN MUAMMO (TIL FILTRIDAGI TESHIK): `detected_lang`
        # candidate juftlik ICHIDA (masalan "en") bo'lishi mumkin, lekin
        # ElevenLabs'ning O'ZINING ishonchi (`language_probability`) past —
        # bu odatda haqiqiy til candidate juftlikdan TASHQARI (masalan
        # o'zbekcha) bo'lib, ElevenLabs shunchaki eng yaqin candidate'ni
        # (past ishonch bilan) "taxmin qilgan" degani. Real logda kuzatilgan:
        # detected_lang=en (0.33) → "Och, herre Jesus. Ja" (gallyutsinatsiya),
        # detected_lang=en (0.12) → bitta belgi qolib ketgan. Haqiqiy EN/RU
        # turnlarda `language_probability` odatda 0.82+. Yuqoridagi
        # "detected_lang not in valid_langs" tekshiruvi buni USHLAMAYDI
        # (chunki "en" candidate ICHIDA), shuning uchun bu MUSTAQIL,
        # qo'shimcha tekshiruv.
        unsupported_language = True
        logger.info(
            f"[STT][LANG][ELEVENLABS] unsupported_language reason=low_lang_prob lang_prob={lang_prob:.3f} "
            f"< {MIN_LANG_PROB} detected_lang={detected_lang!r} — turn tashlab yuborildi"
        )
    # MUHIM: bu endi mustaqil `if`, yuqoridagi til-tekshiruviga `elif` EMAS —
    # fantom-ibora HAR DOIM tekshirilishi kerak, til candidate'dan tashqari
    # chiqgan holatda ham (aks holda VERY_LOW_CONFIDENCE_LANG_PROB filtri
    # dialog.py'dan olib tashlangani uchun, bog'langan spiker uchun bunday
    # fantom matn tekshirilmasdan foydalanuvchiga ko'rsatilib ketishi mumkin).
    if _is_phantom_phrase(text):
        logger.info(f"[STT][ELEVENLABS] phantom-ibora aniqlandi (text={text!r}) — matn bo'shatildi")
        text = ""

    elapsed = time.perf_counter() - t0
    logger.info(f"[STT][ELEVENLABS] {len(text)} chars in {elapsed:.2f}s "
        f"detected_lang={detected_lang} (raw={raw_lang_code!r}) ({lang_prob:.2f})")
    return text, detected_lang, lang_prob, unsupported_language, empty_diag


def _transcribe_audio_whisper(
    audio_path: str,
    language: str | None = None,
    candidate_languages: list[str] | None = None,
) -> tuple[str, str | None, float, bool, dict]:
    """`candidate_languages` berilsa (va `language` berilmagan bo'lsa), Whisper'ning
    ~99 tilli avtomatik aniqlashi o'rniga faqat shu ro'yxatdagi tillar orasida
    (Whisper'ning language-ID boshi orqali) tanlanadi, so'ng transkripsiya shu
    bitta tanlangan til bilan majburlanadi. Dialog rejimida so'zlovchilar tilini
    oldindan tanlagani uchun bu — qisqa/shovqinli navbatlarda avtomatik aniqlash
    tasodifiy boshqa tilni (masalan gruzincha yoki bengalcha) tanlab qo'yishining
    oldini oladi.

    A1/A2 (ARALASH TILLAR): turn "aralash" ko'ringanda (pair_mass past, qarang
    SEG_LANG_CHECK_BELOW) segment-darajasida qo'shimcha til tekshiruvi
    o'tkaziladi (qarang _segment_language_filter) va decode'ga asosiy tilga
    mos initial_prompt beriladi (WHISPER_INITIAL_PROMPTS) — begona (masalan
    ruscha/o'zbekcha) ALOHIDA so'zlar tarjima qilinmasdan, aytilganicha
    saqlanishi uchun."""

    model = get_model()

    empty_diag = {"seg_lang_check": False, "seg_lang_check_ms": 0.0}

    audio = _decode_waveform(audio_path)
    rms = None
    if audio is not None and audio.size:
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
        if rms < RMS_SILENCE_THRESHOLD:
            logger.info(f"[STT] audio below RMS threshold (rms={rms:.5f} < {RMS_SILENCE_THRESHOLD}) — skipping transcription")
            return "", None, 0.0, False, empty_diag


    source = audio if (audio is not None and audio.size) else audio_path

    use_vad = USE_VAD_FILTER and _ONNX_AVAILABLE
    t0 = time.perf_counter()

    candidates = [c for c in dict.fromkeys(candidate_languages or []) if c]

    if language is None and len(candidates) >= 2 and audio is not None and audio.size:
        detected_lang, lang_prob, lang_diag = _detect_language_restricted(model, audio, candidates)

        pair_mass = lang_diag["pair_mass"]
        raw_top_lang = lang_diag["raw_top_lang"]
        raw_top_prob = lang_diag["raw_top_prob"]
        top5_str = ", ".join(f"{lang}={p:.3f}" for lang, p in lang_diag["top5"])
        logger.info(
            f"[STT][LANG] pair=({','.join(candidates)}) pair_mass={pair_mass:.3f} "
            f"raw_top={raw_top_lang}({raw_top_prob:.3f}) top5=[{top5_str}]"
        )

        # REAL TESTDA TOPILGAN MUAMMO (TIL FILTRIDAGI TESHIK — qarang
        # MIN_PAIR_MASS izohi yuqorida): eski shart `... AND raw_top_lang
        # not in candidates` edi — bu "VA" aynan teshikning o'zi bo'lgan,
        # chunki `raw_top_lang` TASODIFAN candidates ICHIGA tushib qolganda
        # (masalan o'zbekcha nutq Whisper'ning xom taqsimotida ru'ga eng
        # yaqin chiqqanda), past pair_mass'ga QARAMASDAN bu tekshiruv
        # o'tkazib yuborilardi. Endi past pair_mass O'ZI YETARLI, mustaqil
        # sabab — `raw_top_lang`ning candidates'da bor-yo'qligi ENDI
        # UMUMAN TEKSHIRILMAYDI.
        if pair_mass < MIN_PAIR_MASS:
            logger.info(
                f"[STT][LANG] unsupported_language reason=low_pair_mass pair_mass={pair_mass:.3f} "
                f"< {MIN_PAIR_MASS} raw_top={raw_top_lang!r}({raw_top_prob:.3f}) "
                f"pair=({','.join(candidates)}) — turn tashlab yuborildi"
            )
            return "", None, 0.0, True, empty_diag

        # IKKINCHI, MUSTAQIL TESHIK (qarang MIN_LANG_PROB izohi yuqorida):
        # pair_mass yetarli bo'lsa ham, ikki nomzod orasidagi QAYTA
        # NORMALLASHTIRILGAN nisbat (`lang_prob`) o'zi past (noaniq)
        # bo'lishi mumkin — bu ham mustaqil ravishda rad etiladi.
        if lang_prob < MIN_LANG_PROB:
            logger.info(
                f"[STT][LANG] unsupported_language reason=low_lang_prob lang_prob={lang_prob:.3f} "
                f"< {MIN_LANG_PROB} detected_lang={detected_lang!r} pair_mass={pair_mass:.3f} "
                f"pair=({','.join(candidates)}) — turn tashlab yuborildi"
            )
            return "", None, 0.0, True, empty_diag

        # A1: shu turn uchun "aralash-til tekshiruvi" kerak-kerakmasligini
        # keyinroq (seg_list tayyor bo'lgandan keyin) hal qilish uchun
        # saqlanadi — toza EN/RU turnlarda (pair_mass yuqori) `None` bo'lib
        # qolib, pastdagi tekshiruv UMUMAN ishlamaydi.
        turn_pair_mass = pair_mass
        # A2: decode tili ALLAQACHON ma'lum (`detected_lang`) — shu tilga
        # mos "begona so'zlarni aytilganicha saqla" ko'rsatmasi beriladi.
        initial_prompt = WHISPER_INITIAL_PROMPTS.get(detected_lang) if WHISPER_INITIAL_PROMPT_ENABLED else None
        seg_list, info = _run_transcribe(model, source, detected_lang, use_vad, initial_prompt=initial_prompt)
    else:
        turn_pair_mass = None
        initial_prompt = (
            WHISPER_INITIAL_PROMPTS.get(language) if (WHISPER_INITIAL_PROMPT_ENABLED and language) else None
        )
        seg_list, info = _run_transcribe(model, source, language, use_vad, initial_prompt=initial_prompt)
        detected_lang = getattr(info, "language", None)
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)

    elapsed = time.perf_counter() - t0

    # A1: turn "aralash" ko'ringanda (pair_mass past) HAR BIR segment uchun
    # ALOHIDA til tekshiruvi — qarang _segment_language_filter va
    # SEG_LANG_CHECK_BELOW izohi. Toza EN/RU turnlarda (`turn_pair_mass`
    # None YOKI yuqori) bu GPU talab qiladigan bosqich UMUMAN ishlamaydi.
    seg_foreign: list[bool] = [False] * len(seg_list)
    seg_lang_diag: list[dict | None] = [None] * len(seg_list)
    seg_lang_check_performed = False
    seg_lang_check_ms = 0.0
    if (
        turn_pair_mass is not None
        and turn_pair_mass < SEG_LANG_CHECK_BELOW
        and audio is not None
        and audio.size
    ):
        t_seg0 = time.perf_counter()
        seg_foreign, seg_lang_diag = _segment_language_filter(model, audio, seg_list, candidates)
        seg_lang_check_performed = True
        seg_lang_check_ms = (time.perf_counter() - t_seg0) * 1000
        if any(seg_foreign):
            logger.info(
                f"[STT][LANG] seg_lang_check turn_pair_mass={turn_pair_mass:.3f} "
                f"dropped {sum(seg_foreign)}/{len(seg_list)} segment(s) in {seg_lang_check_ms:.1f}ms"
            )

    # SHOVQINLI MUHIT TUZATISHI: turn ichidagi HAR BIR segmentning o'z RMS'i
    # oldindan hisoblanadi — pastdagi asosiy tsiklda har bir segment shu
    # TURN ICHIDAGI eng baland segmentga nisbatan solishtiriladi (qarang
    # SEGMENT_REL_RMS izohi). `audio` yo'q/bo'sh bo'lsa (masalan decode
    # muvaffaqiyatsiz bo'lib, `source` to'g'ridan-to'g'ri audio_path bo'lgan
    # holatda) — RMS hisoblab bo'lmaydi, filtr shunchaki qo'llanmaydi (hamma
    # segment uchun `None`).
    seg_rms_list: list[float | None] = []
    if audio is not None and audio.size:
        for seg in seg_list:
            seg_start_s = getattr(seg, "start", None)
            seg_end_s = getattr(seg, "end", None)
            if seg_start_s is None or seg_end_s is None:
                seg_rms_list.append(None)
                continue
            start_idx = max(0, int(seg_start_s * SAMPLE_RATE))
            end_idx = min(audio.size, int(seg_end_s * SAMPLE_RATE))
            if end_idx <= start_idx:
                seg_rms_list.append(None)
                continue
            seg_audio = audio[start_idx:end_idx]
            seg_rms_list.append(float(np.sqrt(np.mean(np.square(seg_audio.astype(np.float64))))))
    else:
        seg_rms_list = [None] * len(seg_list)

    valid_seg_rms = [r for r in seg_rms_list if r is not None]
    max_seg_rms = max(valid_seg_rms) if valid_seg_rms else None

    kept_parts = []
    raw_parts = []
    # Pastdagi "hammasi filtrlandi, xomdan foydalanamiz" fallback'i uchun —
    # nsp/alp/compression_ratio asosida filtrlangan segment haqiqiy nutq
    # bo'lib chiqishi mumkin (noto'g'ri-pozitiv), lekin FANTOM ibora HECH
    # QACHON bu fallback orqali qaytarilmasligi kerak (aks holda butun
    # filtrning ma'nosi qolmaydi — qisqa klip FAQAT bitta fantom segmentdan
    # iborat bo'lsa, u "yagona segment" sifatida qayta kiritilib ketar edi).
    # FAR_FIELD segmentlar xuddi shu sababdan fallback'ga HECH QACHON
    # kiritilmaydi — ular fon deb aniqlangan, "hammasi filtrlandi" degani
    # ularni qayta tiklash kerak degani emas.
    fallback_candidates = []
    dropped = 0
    for idx, (seg, seg_rms) in enumerate(zip(seg_list, seg_rms_list)):
        nsp = getattr(seg, "no_speech_prob", 0.0)
        alp = getattr(seg, "avg_logprob", 0.0)
        cr = getattr(seg, "compression_ratio", 0.0)
        seg_start = getattr(seg, "start", None)
        seg_end = getattr(seg, "end", None)
        seg_text = seg.text.strip()
        raw_parts.append(seg_text)

        # SHOVQINLI MUHIT TUZATISHI: bu tekshiruv ATAYLAB eng BOSHIDA,
        # nsp/alp/compression/fantom tekshiruvlaridan OLDIN turadi — fon
        # segmenti ko'pincha o'z-o'zidan ham "past ishonchli" chiqishi
        # mumkin, lekin SABAB sifatida aniq "far_field" ko'rsatish
        # diagnostika uchun foydaliroq.
        is_far_field = (
            seg_rms is not None
            and max_seg_rms is not None
            and max_seg_rms > 0
            and seg_rms < SEGMENT_REL_RMS * max_seg_rms
        )
        if is_far_field:
            logger.info(
                f"[STT][DROP] reason=far_field text={seg_text!r} rms={seg_rms:.4f} "
                f"max_rms={max_seg_rms:.4f} start={seg_start} end={seg_end}"
            )
            dropped += 1
            continue

        # A1: ARALASH TILLAR — shu segment (yetarlicha uzun bo'lsa)
        # _segment_language_filter orqali ikki sessiya tilidan ALOHIDA,
        # BEGONA til deb aniqlangan bo'lsa, far_field'dan KEYINGI eng
        # BIRINCHI sabab sifatida tashlab yuboriladi (fallback'ga ham
        # KIRMAYDI — qisman noto'g'ri bo'lsa ham, begona til matnini
        # "fallback" sifatida qayta tiklash ma'nosiz).
        if seg_foreign[idx]:
            seg_diag = seg_lang_diag[idx] or {}
            logger.info(
                f"[STT][DROP] reason=foreign_segment text={seg_text!r} "
                f"seg_pair_mass={seg_diag.get('pair_mass', 0.0):.3f} "
                f"raw_top={seg_diag.get('raw_top_lang')!r}({seg_diag.get('raw_top_prob', 0.0):.3f})"
            )
            dropped += 1
            continue

        is_phantom = _is_phantom_phrase(seg_text)
        if not is_phantom:
            fallback_candidates.append(seg_text)

        # REAL TESTDA TOPILGAN MUAMMO TUZATISHI: ilgari no_speech_prob VA
        # avg_logprob ALOHIDA-ALOHIDA (OR) tekshirilardi — shuning uchun
        # haqiqiy, tushunarli aytilgan segment ham, ikkisidan FAQAT BITTASI
        # (masalan ozgina baland no_speech_prob, lekin aslida yaxshi
        # avg_logprob) chegaradan o'tsa, noto'g'ri tashlab yuborilardi
        # (ayniqsa ruscha turnlarda ko'p kuzatilgan). Whisper'ning o'zi bu
        # ikkisini BIRGA (AND) qarab hal qiladi — endi shu yerda ham xuddi
        # shunday: FAQAT ikkalasi BIRGA chegaradan o'tganda "past ishonchli"
        # deb hisoblanadi. compression_ratio va fantom-ibora — bular MUSTAQIL
        # (har biri o'zi yolg'iz ham yetarli) sabablar, o'zgarishsiz qoldi.
        is_low_confidence = nsp > NO_SPEECH_PROB_THRESHOLD and alp < AVG_LOGPROB_THRESHOLD
        is_long_compression = cr > COMPRESSION_RATIO_THRESHOLD
        if is_low_confidence or is_long_compression or is_phantom:
            reasons = []
            if is_low_confidence:
                reasons.append(
                    f"low_confidence (no_speech_prob={nsp:.3f} > {NO_SPEECH_PROB_THRESHOLD} "
                    f"AND avg_logprob={alp:.3f} < {AVG_LOGPROB_THRESHOLD})"
                )
            if is_long_compression:
                reasons.append(f"compression_ratio {cr:.2f} > {COMPRESSION_RATIO_THRESHOLD}")
            if is_phantom:
                reasons.append("phantom_phrase")
            # Har doim (DEBUG_STT'dan qat'i nazar) chiqariladi — haqiqiy
            # segmentlar noto'g'ri tashlab yuborilayotganini real ishlatishda
            # ham kuzatib borish uchun.
            logger.info(
                f"[STT][DROP] text={seg_text!r} no_speech_prob={nsp:.3f} avg_logprob={alp:.3f} "
                f"compression_ratio={cr:.2f} start={seg_start} end={seg_end} "
                f"reason=({'; '.join(reasons)})"
            )
            dropped += 1
            continue

        if DEBUG_STT:
            logger.info(f"[STT][DEBUG] segment KEPT    [{seg_start}-{seg_end}] text={seg_text!r} "
                f"no_speech_prob={nsp:.3f} avg_logprob={alp:.3f}")
        kept_parts.append(seg_text)

    text = "".join(kept_parts).strip()

    if not text and fallback_candidates:
        fallback_text = "".join(fallback_candidates).strip()
        if fallback_text:
            if DEBUG_STT:
                logger.info("[STT][DEBUG] fallback used: all segments filtered, using raw transcript")
            text = fallback_text

    # A1: agar natijada hali ham hech narsa qolmagan bo'lsa VA buning
    # sababi (kamida qisman) begona-til segmentlari bo'lsa — bu "shovqin"
    # emas, BUTUN turn chindan begona tilda bo'lib chiqdi degani (turn
    # darajasidagi tekshiruv buni ko'rolmagan, chunki pair_mass yetarli
    # EDI, lekin segmentlarning HAMMASI, alohida tekshirilganda, begona
    # bo'lib chiqdi). `fallback_candidates` begona segmentlarni ATAYLAB
    # o'z ichiga OLMAYDI (yuqoridagi `continue`), shuning uchun bu yerga
    # faqat haqiqatan ham "qolgan hech narsa yo'q" holatida yetib keladi.
    if not text and seg_lang_check_performed and any(seg_foreign):
        logger.info(
            f"[STT][LANG] unsupported_language reason=all_segments_foreign "
            f"pair=({','.join(candidates)}) — turn tashlab yuborildi"
        )
        return (
            "",
            None,
            0.0,
            True,
            {"seg_lang_check": seg_lang_check_performed, "seg_lang_check_ms": seg_lang_check_ms},
        )

    # Yuqoridagi `_is_phantom_phrase` tekshiruvi HAR BIR SEGMENT uchun
    # alohida ishlaydi — agar fantom ibora Whisper tomonidan IKKI segmentga
    # bo'lib yuborilgan bo'lsa (masalan "Thank you" + " for watching"), har
    # biri o'z holicha to'liq iborani o'z ichiga olmaydi va shu tekshiruvdan
    # o'tib ketishi mumkin. Shu sababli YAKUNIY (birlashtirilgan) matnga ham
    # qo'shimcha, mustaqil tekshiruv — transcribe_audio_elevenlabs()'dagi
    # yakuniy-matn tekshiruvi bilan BIR XIL maqsadda.
    if _is_phantom_phrase(text):
        logger.info(f"[STT] final text is a phantom phrase (text={text!r}) — emptied")
        text = ""

    msg = f"[STT] {len(text)} chars in {elapsed:.2f}s detected_lang={detected_lang} ({lang_prob:.2f})"
    if dropped:
        # `dropped` endi low-confidence/compression/phantom/foreign_segment
        # BILAN BIRGA far_field (fon-segment) sabablarini ham o'z ichiga
        # oladi — har birining ANIQ sababi yuqoridagi [STT][DROP] qatorlarida
        # ko'rinadi.
        msg += f" ({dropped} segment(s) filtered)"
    if seg_lang_check_performed:
        msg += f" seg_lang_check={seg_lang_check_ms:.1f}ms"
    logger.info(msg)

    return (
        text,
        detected_lang,
        lang_prob,
        False,
        {"seg_lang_check": seg_lang_check_performed, "seg_lang_check_ms": seg_lang_check_ms},
    )
