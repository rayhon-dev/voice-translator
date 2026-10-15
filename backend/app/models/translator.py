import logging
import os
import re
import threading
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import (
    TRANSLATION_MODEL_NAME,
    MT_PROVIDER,
    DEEPL_API_KEY,
    MT_FALLBACK_PRELOAD,
    NLLB_LANG_CODES,
)

logger = logging.getLogger(__name__)


def _report_vram() -> str:
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            return f"{allocated:.2f}GB allocated / {reserved:.2f}GB reserved"
    except Exception:
        pass
    return "n/a"


# ---------------------------------------------------------------------------
# MATNNI TAYYORLASH — ikkala provayder (DeepL va NLLB) UCHUN UMUMIY. Faqat
# "himoyalangan atamalar"ni QANDAY belgilash (raqamli placeholder vs XML
# teg) ikkisida FARQ qiladi — qarang pastdagi _protect_terms_nllb/_deepl.
# ---------------------------------------------------------------------------

_FILLER_WORDS = ["uh", "um", "uhm", "mm", "hmm", "э", "эм", "ммм", "хм"]
_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _FILLER_WORDS) + r")\b,?",
    re.IGNORECASE,
)
_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
# "--" yoki yolg'iz "-" bilan tugaydigan, BO'SHLIQ/qator oxiridan OLDIN
# turgan so'z — kesilgan ("Это не рабо--") fragment. O'rtada turgan
# ("well-known" kabi) chiziqchalar TEGILMAYDI, chunki ulardan keyin
# bo'shliq/qator oxiri EMAS, yana harf keladi.
_TRUNCATED_WORD_RE = re.compile(r"\b[^\s-]+-{1,2}(?=\s|$)")

# REAL TESTDA TOPILGAN XATO: Whisper ba'zan segmentlarni BO'SHLIQSIZ ulab
# yuboradi (gap chegarasi segment chegarasiga to'g'ri kelganda), natijada
# "again.So", "are.Pretty", "news.Then" kabi ketma-ketliklar paydo bo'ladi
# — bular keyinroq (himoyalashda) SOXTA "domen" deb noto'g'ri qabul
# qilinishi mumkin edi. Himoyalashdan OLDIN — kichik harf + [.!?] + BOSH
# HARF orasiga bo'shliq qaytariladi. Haqiqiy domenlar ("listenaminute.com
# Social...") ALLAQACHON bo'shliq bilan yozilgan, shuning uchun bu qoida
# ularga ta'sir qilmaydi.
_MISSING_SPACE_RE = re.compile(r"(?<=[a-zа-яё])([.!?])(?=[A-ZА-ЯЁ])")


def _fix_missing_sentence_space(text: str) -> str:
    return _MISSING_SPACE_RE.sub(r"\1 ", text)


def _clean_text(text: str) -> str:
    """To'ldiruvchi so'zlar, [...]/(...)  teglar va kesilgan so'z
    fragmentlarini olib tashlaydi, so'ng bo'shliq/tinish belgilarini
    tozalaydi."""
    text = _BRACKET_RE.sub(" ", text)
    text = _FILLER_RE.sub("", text)
    text = _TRUNCATED_WORD_RE.sub("", text)
    text = re.sub(r"\s*,\s*,", ",", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"^[\s,]+", "", text)
    text = re.sub(r",\s*([.!?])", r"\1", text)
    return text.strip()


# Himoyalanadigan atamalar — brend/mahsulot nomlari, hech qachon tarjima
# qilinmaydi. .env orqali to'liq almashtirish mumkin.
MT_PROTECTED_TERMS = [
    t.strip()
    for t in os.environ.get("MT_PROTECTED_TERMS", "Whisper,ElevenLabs,NLLB,API,Claude,GPU").split(",")
    if t.strip()
]

# Domen/URL — FAQAT taniqli TLD bilan tugasa (yoki http(s)://, www. bilan
# boshlansa) hisoblanadi. REAL TESTDA TOPILGAN XATO: ESKI versiya HAR
# QANDAY "so'z.SO'Z" ketma-ketligini domen deb olardi (qarang
# _MISSING_SPACE_RE izohi yuqorida — "again.So" kabi) — endi domen FAQAT
# shu ro'yxatdagi TLD bilan tugasa qabul qilinadi.
_URL_TLDS = [
    "com", "org", "net", "io", "ai", "dev", "ru", "uz", "edu", "gov", "co", "me", "app", "info",
]
_URL_RE = re.compile(
    r"\b(?:https?://|www\.)\S+"
    r"|\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+(?:" + "|".join(_URL_TLDS) + r")\b(?:/\S*)?",
    re.IGNORECASE,
)
# O'zbek kirill alifbosidagi qo'shimcha harflar (ў, қ, ғ, ҳ) ham kiritilgan —
# EN matn ichida ruscha HAM, o'zbekcha-kirillcha HAM so'z chiqib qolishi
# mumkin (speaker aralash gapirsa).
_CYRILLIC_WORD_RE = re.compile(r"\b[А-Яа-яЁёЎўҚқҒғҲҳ]+\b")
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z]+\b")


def _find_and_wrap_protected(text: str, source_lang: str, wrap_fn) -> str:
    """URL/domenlar, MT_PROTECTED_TERMS va "begona yozuvdagi" so'zlarni
    (EN matnda kirill, RU matnda lotin — begona so'zlar tarjimada ham
    aytilganicha qolishi uchun) topib, HAR BIRI uchun `wrap_fn(matched_text)`
    chaqirib, natijaga almashtiradi. `wrap_fn` provayderga qarab farq
    qiladi — NLLB uchun raqamli placeholder (qarang _protect_terms_nllb),
    DeepL uchun `<x>...</x>` tegi (qarang _protect_terms_deepl)."""
    result = _URL_RE.sub(lambda m: wrap_fn(m.group(0)), text)

    if MT_PROTECTED_TERMS:
        terms_pattern = "|".join(re.escape(t) for t in MT_PROTECTED_TERMS)
        result = re.sub(rf"\b(?:{terms_pattern})\b", lambda m: wrap_fn(m.group(0)), result, flags=re.IGNORECASE)

    if source_lang == "en":
        result = _CYRILLIC_WORD_RE.sub(lambda m: wrap_fn(m.group(0)), result)
    elif source_lang == "ru":
        result = _LATIN_WORD_RE.sub(lambda m: wrap_fn(m.group(0)), result)

    return result


# ---------------------------------------------------------------------------
# DEEPL — asosiy provayder (MT_PROVIDER="deepl", default)
# ---------------------------------------------------------------------------

DEEPL_SOURCE_LANG = {"en": "EN", "ru": "RU"}
DEEPL_TARGET_LANG = {"en": "EN-US", "ru": "RU"}

DEEPL_TIMEOUT_S = float(os.environ.get("DEEPL_TIMEOUT_S", "3.0"))
# Ketma-ket shuncha marta DeepL muvaffaqiyatsiz bo'lsa — "circuit" ochiladi
# (qarang DEEPL_CIRCUIT_COOLDOWN_S) — DeepL vaqtincha BUTUNLAY so'ralmaydi,
# har bir turn uchun 3 soniya kutib keyin NLLB'ga o'tish o'rniga, darhol
# NLLB'ga o'tiladi.
DEEPL_CIRCUIT_FAIL_THRESHOLD = int(os.environ.get("DEEPL_CIRCUIT_FAIL_THRESHOLD", "3"))
DEEPL_CIRCUIT_COOLDOWN_S = float(os.environ.get("DEEPL_CIRCUIT_COOLDOWN_S", "60"))

_deepl_client = None  # deepl.Translator | None (None = kalit yo'q yoki hali yaratilmagan)
_deepl_consecutive_failures = 0
_deepl_circuit_open_until = 0.0  # time.monotonic() vaqt belgisi
# Diagnostika: shu server-jarayoni DAVOMIDA DeepL'ga yuborilgan belgilar
# yig'indisi — DeepL Free (500000/oy) kvotasiga qancha yaqinlashganini
# [MT] logida kuzatib borish uchun (DeepL'ning o'z get_usage()'i — aniqroq
# — FAQAT startup'da so'raladi, bu yerda esa HAR BIR chaqiruvda ko'rinadi).
_deepl_session_chars = 0


def _load_deepl_client():
    """`deepl.Translator`ni BIR MARTA yaratadi (server ishga tushganda —
    warmup() orqali — yoki birinchi kerak bo'lgan paytda). Kalit sozlanmagan
    bo'lsa yoki MT_PROVIDER != "deepl" bo'lsa — None qaytaradi, DeepL
    UMUMAN chaqirilmaydi (xatosiz, to'g'ridan-to'g'ri NLLB)."""
    global _deepl_client
    if _deepl_client is not None:
        return _deepl_client
    if MT_PROVIDER != "deepl":
        return None
    if not DEEPL_API_KEY:
        logger.info("[MT] DEEPL_API_KEY sozlanmagan (.env'ni tekshiring) — FAQAT NLLB-600M ishlatiladi")
        return None

    import deepl
    import deepl.http_client as deepl_http_client

    # Rasmiy `deepl` kutubxonasi Translator.__init__'da ochiq timeout/retry
    # parametrini QABUL QILMAYDI — o'zining ichki backoff/retry mexanizmi
    # shu IKKI modul-darajasidagi konstantani o'qiydi, shuning uchun talab
    # qilingan "3 soniya timeout, 1 qayta urinish" ANIQ shular orqali
    # sozlanadi (SDK'ning o'z mexanizmiga QARSHI chiqmasdan).
    deepl_http_client.min_connection_timeout = DEEPL_TIMEOUT_S
    deepl_http_client.max_network_retries = 1

    _deepl_client = deepl.Translator(DEEPL_API_KEY)

    try:
        usage = _deepl_client.get_usage()
        if usage.character.valid:
            logger.info(f"[MT] DeepL usage: {usage.character.count} / {usage.character.limit}")
    except Exception as e:
        logger.info(f"[MT] DeepL usage so'rovi muvaffaqiyatsiz bo'ldi: {e!r}")

    return _deepl_client


def _protect_terms_deepl(text: str, source_lang: str) -> str:
    """Himoyalangan atamalarni `<x>...</x>` tegiga o'raydi — DeepL'ga
    `tag_handling="xml", ignore_tags=["x"]` bilan yuborilganda, shu teg
    ICHIDAGI matn TARJIMA QILINMAYDI, aytilganicha qaytadi."""
    return _find_and_wrap_protected(text, source_lang, lambda original: f"<x>{original}</x>")


def _translate_via_deepl(text: str, source_lang: str, target_lang: str) -> str:
    """Muvaffaqiyatsiz bo'lsa ISTISNO ko'taradi — chaqiruvchi (_deepl_attempt)
    buni "zaxiraga o'tish" signaliga aylantiradi."""
    global _deepl_session_chars

    wrapped = _protect_terms_deepl(text, source_lang)
    result = _deepl_client.translate_text(
        wrapped,
        source_lang=DEEPL_SOURCE_LANG[source_lang],
        target_lang=DEEPL_TARGET_LANG[target_lang],
        tag_handling="xml",
        ignore_tags=["x"],
    )
    _deepl_session_chars += len(wrapped)
    return re.sub(r"</?x>", "", result.text).strip()


def _deepl_attempt(text: str, source_lang: str, target_lang: str) -> tuple[str | None, str | None]:
    """Qaytaradi: (tarjima, None) muvaffaqiyatli bo'lsa, (None, sabab)
    DeepL ishlatilmagan/muvaffaqiyatsiz bo'lsa — ikkinchi holatda
    chaqiruvchi (translate_text) NLLB'ga o'tadi."""
    global _deepl_consecutive_failures, _deepl_circuit_open_until

    client = _load_deepl_client()
    if client is None:
        return None, "no_api_key"

    now = time.monotonic()
    if now < _deepl_circuit_open_until:
        return None, "circuit_open"

    try:
        translation = _translate_via_deepl(text, source_lang, target_lang)
        _deepl_consecutive_failures = 0
        return translation, None
    except Exception as e:
        _deepl_consecutive_failures += 1
        reason = type(e).__name__
        if _deepl_consecutive_failures >= DEEPL_CIRCUIT_FAIL_THRESHOLD:
            _deepl_circuit_open_until = time.monotonic() + DEEPL_CIRCUIT_COOLDOWN_S
            _deepl_consecutive_failures = 0
            logger.info(
                f"[MT][FALLBACK] reason={reason} — {DEEPL_CIRCUIT_FAIL_THRESHOLD} ketma-ket xato, "
                f"DeepL {DEEPL_CIRCUIT_COOLDOWN_S:.0f}s davomida chaqirilmaydi"
            )
        return None, reason


# ---------------------------------------------------------------------------
# NLLB-600M — zaxira provayder (MT_PROVIDER="nllb" bo'lsa — YAGONA provayder)
# ---------------------------------------------------------------------------

MT_NUM_BEAMS = int(os.environ.get("MT_NUM_BEAMS", "4"))
# Qisqarib qolgan gapni qayta tarjima qilishda ishlatiladigan kattaroq
# beam_size (sekinroq, lekin FAQAT muvaffaqiyatsiz chiqgan kam sonli
# gaplar uchun, shuning uchun umumiy tezlikka sezilarli ta'sir qilmaydi).
MT_RETRY_NUM_BEAMS = int(os.environ.get("MT_RETRY_NUM_BEAMS", "6"))
# Shundan PAST uzunlik nisbati (tarjima_uzunligi / kirish_uzunligi) — gap
# TUSHIB QOLGAN bo'lishi mumkin deb hisoblanadi.
MT_RETRY_MIN_RATIO = float(os.environ.get("MT_RETRY_MIN_RATIO", "0.4"))
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("MT_NO_REPEAT_NGRAM_SIZE", "3"))
MT_MAX_LENGTH_CAP = int(os.environ.get("MT_MAX_LENGTH_CAP", "256"))

_PLACEHOLDER_BASE = 900000

_nllb_model = None
_nllb_tokenizer = None
_nllb_loaded = False

# DIALOG_PLAN.md 4-bosqich (D3): `tokenizer.src_lang = ...` global, ULASHILGAN
# tokenizer holatini o'zgartiradi — shuning uchun src_lang o'rnatish +
# tokenize + generate BUTUNLAY shu lock ostida (bir vaqtning o'zida faqat
# BITTA NLLB so'rovi bajariladi). DeepL chaqiruvlari bu lock'ga KIRMAYDI —
# DeepL stateless HTTP mijoz, ulashilgan mutable holatga ega emas.
_translate_lock = threading.Lock()


def _load_nllb_model() -> None:
    global _nllb_model, _nllb_tokenizer, _nllb_loaded
    if _nllb_loaded:
        return
    t0 = time.perf_counter()
    _nllb_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME, torch_dtype=torch.float16)
    _nllb_model.to("cuda")
    _nllb_loaded = True
    logger.info(
        f"[MT] NLLB zaxira modeli yuklandi: {TRANSLATION_MODEL_NAME!r} "
        f"{time.perf_counter() - t0:.2f}s vram=({_report_vram()})"
    )


def _protect_terms_nllb(text: str, source_lang: str) -> tuple[str, dict[str, str]]:
    """Himoyalangan atamalarni RAQAMLI placeholder'ga almashtiradi — NLLB
    ularni "ko'rmaydi", shuning uchun tarjima qilmaydi/buzmaydi.
    `_restore_placeholders()` tarjimadan KEYIN ularni asl so'zga qaytaradi.
    Placeholder — FAQAT RAQAMLAR: NLLB (en/ru yo'nalishida, ikkisi ham arab
    raqamlaridan foydalanadi) sof raqamlarni DEYARLI HAR DOIM o'zgarishsiz
    qoldiradi."""
    placeholder_map: dict[str, str] = {}
    counter = 0

    def _make_placeholder(original: str) -> str:
        nonlocal counter
        counter += 1
        placeholder = str(_PLACEHOLDER_BASE + counter)
        placeholder_map[placeholder] = original
        return placeholder

    result = _find_and_wrap_protected(text, source_lang, _make_placeholder)
    return result, placeholder_map


def _placeholder_pattern(placeholder: str) -> re.Pattern:
    """Ruscha tarjimada ko'p xonali raqamlar ko'pincha guruhlab yoziladi —
    "900001" o'rniga "900 001" yoki "900,001" chiqishi mumkin, aniq-mos
    qidiruv buni TOPOLMAYDI. Shu sababli raqamlarning HAR BIR juftligi
    orasida IXTIYORIY bitta bo'shliq/vergul/nuqta (yoki hech narsa)
    bo'lishi mumkin deb qidiriladi."""
    sep = r"[\s,.]?"
    body = sep.join(re.escape(ch) for ch in placeholder)
    return re.compile(r"(?<!\d)" + body + r"(?!\d)")


def _insert_near_correct_spot(text: str, original: str) -> str:
    """Tiklanmagan placeholder uchun — gap OXIRIGA ko'r-ko'rona qo'shish
    o'rniga, YAKUNIY tinish belgisidan (., !, ?) OLDIN qo'yiladi."""
    text = text.rstrip()
    m = re.search(r"[.!?]+$", text)
    if m:
        return text[: m.start()].rstrip() + f" {original}" + text[m.start() :]
    return text + f" {original}"


def _restore_placeholders(text: str, placeholder_map: dict[str, str]) -> str:
    restored = []
    missing = []
    for placeholder, original in placeholder_map.items():
        pattern = _placeholder_pattern(placeholder)
        match = pattern.search(text)
        if match:
            text = text[: match.start()] + f" {original} " + text[match.end() :]
            restored.append(original)
        else:
            missing.append(original)
            text = _insert_near_correct_spot(text, original)
    if restored or missing:
        logger.info(f"[MT][PLACEHOLDER] restored={restored} missing={missing}")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return text


# Gaplarga bo'lishda "Dr.", "3.5", "J.K." kabilarni BUZMASLIK uchun — bu
# nuqtalar vaqtincha placeholder belgisiga almashtiriladi, bo'lingandan
# KEYIN qaytariladi.
_SENTENCE_DOT_PLACEHOLDER = "\x01"
_ABBREVIATIONS = ["Dr", "Mr", "Mrs", "Ms", "St", "Prof", "Jr", "Sr", "vs", "etc", "e.g", "i.e"]
_ABBREV_RE = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.", re.IGNORECASE)
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_INITIALS_RE = re.compile(r"(?:[A-ZА-ЯЁ]\.){2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_SENTENCE_END_RE = re.compile(r"[.!?…\"'”’)\]]$")


def _split_sentences(text: str) -> list[tuple[str, bool]]:
    """Matnni gaplarga bo'ladi, "Dr."/"3.5"/"J.K." kabilarni buzmasdan.
    Qaytaradi: [(gap, biz_nuqta_qoshdikmi), ...] — `biz_nuqta_qoshdikmi=True`
    bo'lsa, tarjimadan keyin shu gapning OXIRIDAGI nuqta olib tashlanadi
    (asl matnda tinish belgisi bo'lmagan edi)."""
    protected = _DECIMAL_RE.sub(_SENTENCE_DOT_PLACEHOLDER, text)
    protected = _ABBREV_RE.sub(lambda m: m.group(0)[:-1] + _SENTENCE_DOT_PLACEHOLDER, protected)
    protected = _INITIALS_RE.sub(lambda m: m.group(0).replace(".", _SENTENCE_DOT_PLACEHOLDER), protected)

    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(protected) if s.strip()]

    result = []
    for sent in raw_sentences:
        sent = sent.replace(_SENTENCE_DOT_PLACEHOLDER, ".")
        added_period = not bool(_SENTENCE_END_RE.search(sent))
        if added_period:
            sent = sent + "."
        result.append((sent, added_period))
    return result


def _generate_nllb(sentences: list[str], src_lang: str, tgt_lang: str, beam_size: int) -> list[str]:
    _nllb_tokenizer.src_lang = src_lang
    inputs = _nllb_tokenizer(sentences, return_tensors="pt", padding=True).to("cuda")
    forced_bos_token_id = _nllb_tokenizer.convert_tokens_to_ids(tgt_lang)
    max_len = min(MT_MAX_LENGTH_CAP, int(inputs["input_ids"].shape[1] * 2 + 10))

    with torch.no_grad():
        generated_tokens = _nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_len,
            num_beams=beam_size,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        )
    return _nllb_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


_LEADING_DASH_RE = re.compile(r"^[\-–—]\s*")


def _strip_leading_dash(s: str) -> str:
    return _LEADING_DASH_RE.sub("", s.strip())


def _dedupe_consecutive(sentences: list[str]) -> list[str]:
    """Ketma-ket, AYNAN bir xil (katta-kichik harfga e'tibor bermasdan)
    gaplarni bittaga qisqartiradi — real logda ko'rilgan "- Я не знаю. -
    Я не знаю." kabi takrorlanish uchun."""
    out = []
    prev_norm = None
    for s in sentences:
        norm = s.strip().lower()
        if norm and norm == prev_norm:
            continue
        out.append(s)
        if norm:
            prev_norm = norm
    return out


def _translate_batch_with_retry_nllb(
    sentences: list[str], src_lang: str, tgt_lang: str
) -> tuple[list[str], int]:
    """Barcha gaplar bitta batch'da + qisqa chiqgan gaplarni qayta tarjima
    qilish. _translate_lock ICHIDA chaqirilishi SHART."""
    translations = _generate_nllb(sentences, src_lang, tgt_lang, beam_size=MT_NUM_BEAMS)

    retry_count = 0
    final = list(translations)
    for i, (src_sent, translated) in enumerate(zip(sentences, translations)):
        if not src_sent:
            continue
        ratio = len(translated) / max(1, len(src_sent))
        if ratio < MT_RETRY_MIN_RATIO:
            retry_translated = _generate_nllb([src_sent], src_lang, tgt_lang, beam_size=MT_RETRY_NUM_BEAMS)[0]
            retry_count += 1
            logger.info(
                f"[MT][RETRY] sentence={src_sent!r} first={translated!r} (ratio={ratio:.2f} "
                f"< {MT_RETRY_MIN_RATIO}) retry={retry_translated!r}"
            )
            if len(retry_translated) > len(translated):
                final[i] = retry_translated
    return final, retry_count


def _translate_nllb(cleaned_text: str, source_lang: str, target_lang: str) -> str:
    _load_nllb_model()

    src_lang = NLLB_LANG_CODES[source_lang]
    tgt_lang = NLLB_LANG_CODES[target_lang]

    protected_text, placeholder_map = _protect_terms_nllb(cleaned_text, source_lang)
    sentence_pairs = _split_sentences(protected_text)
    if not sentence_pairs:
        return cleaned_text

    sentence_texts = [s for s, _ in sentence_pairs]

    with _translate_lock:
        translations, retry_count = _translate_batch_with_retry_nllb(sentence_texts, src_lang, tgt_lang)

    out_sentences = []
    for (_src_sent, added_period), translated in zip(sentence_pairs, translations):
        translated = _strip_leading_dash(translated)
        if added_period and translated.endswith("."):
            translated = translated[:-1]
        out_sentences.append(translated.strip())

    out_sentences = _dedupe_consecutive(out_sentences)
    result = " ".join(s for s in out_sentences if s)
    result = _restore_placeholders(result, placeholder_map)

    if retry_count:
        logger.info(f"[MT] NLLB: {len(sentence_texts)} gap, {retry_count} qayta urinish")
    return result


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def warmup(run_inference: bool = True) -> None:
    if MT_PROVIDER == "deepl":
        _load_deepl_client()
    if MT_PROVIDER == "nllb" or MT_FALLBACK_PRELOAD:
        _load_nllb_model()

    provider_desc = "DeepL (asosiy) + NLLB-600M (zaxira)" if MT_PROVIDER == "deepl" else "NLLB-600M (yagona)"
    logger.info(f"[MT] {provider_desc}")

    if not run_inference:
        return
    try:
        translation, provider = translate_text("Hello.", "ru")
        logger.info(f"[MT] warm-up inference done (provider={provider!r} translation={translation!r})")
    except Exception as e:
        logger.info(f"[MT] warm-up inference skipped: {e!r}")


def translate_text(text: str, target_lang: str, source_lang: str = "en") -> tuple[str, str]:
    """Qaytaradi: (translation, provider) — `provider` "deepl", "nllb" yoki
    (IKKISI HAM muvaffaqiyatsiz bo'lsa) "error" ("" tarjima bilan — chaqiruvchi,
    dialog.py, bu holatda turn'ni SKIP qilmasdan, bo'sh tarjima + xatolik
    bayrog'i bilan frontendga yuboradi)."""
    t0 = time.perf_counter()

    cleaned = _clean_text(_fix_missing_sentence_space(text))

    translation: str | None = None
    provider: str | None = None

    if MT_PROVIDER == "deepl":
        translation, fail_reason = _deepl_attempt(cleaned, source_lang, target_lang)
        if translation is not None:
            provider = "deepl"
        else:
            logger.info(f"[MT][FALLBACK] deepl->nllb reason={fail_reason}")

    if provider is None:
        try:
            translation = _translate_nllb(cleaned, source_lang, target_lang)
            provider = "nllb"
        except Exception as e:
            logger.error(f"[MT] NLLB tarjima HAM muvaffaqiyatsiz bo'ldi: {e!r}")
            translation = ""
            provider = "error"

    logger.info(
        f"[MT] provider={provider} translated {len(text)} -> {len(translation)} chars "
        f"({source_lang} -> {target_lang}) in {time.perf_counter() - t0:.2f}s "
        f"session_chars={_deepl_session_chars}"
    )
    return translation, provider
