import threading
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.config import TRANSLATION_MODEL_NAME, NLLB_LANG_CODES
import logging

logger = logging.getLogger(__name__)


_model = None
_tokenizer = None

# DIALOG_PLAN.md 4-bosqich (D3): `tokenizer.src_lang = ...` global, ULASHILGAN
# tokenizer holatini o'zgartiradi — keyingi `tokenizer(text, ...)` chaqiruvi
# ANIQ shu holatga tayanadi. Agar ikki so'rov (masalan Dialog va Solo rejim,
# yoki ikki parallel Dialog sessiyasi) bir vaqtda turli thread'larda
# translate_text() chaqirsa (run_in_executor/threadpool orqali), ular orasida
# POYGA (race) yuzaga kelishi mumkin edi: A src_lang='eng_Latn' qo'ysa, B
# ulgurib 'rus_Cyrl'ga almashtirsa, A ANIQ shu payt tokenize qilsa — A matni
# NOTO'G'RI manba til bilan tokenlashtiriladi. Shu sababli src_lang
# o'rnatish + tokenize + generate — BUTUNLAY shu lock ostida, bir vaqtning
# o'zida faqat BITTA so'rov bajaradi.
_translate_lock = threading.Lock()


def _load_model():
    global _model, _tokenizer
    if _model is None:
        # Loaded once per process 
        t0 = time.perf_counter()
        _tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL_NAME, torch_dtype=torch.float16
        )
        _model.to("cuda")
        logger.info(f"[MT] translation model loaded in {time.perf_counter() - t0:.2f}s")
    return _model, _tokenizer


def warmup(run_inference: bool = True) -> None:
    _load_model()
    if not run_inference:
        return
    try:
        translate_text("Hello.", "uz")
        logger.info("[MT] warm-up inference done")
    except Exception as e: 
        logger.info(f"[MT] warm-up inference skipped: {e!r}")


def translate_text(text: str, target_lang: str, source_lang: str = "en") -> str:
    model, tokenizer = _load_model()

    t0 = time.perf_counter()

    src_lang = NLLB_LANG_CODES[source_lang]
    tgt_lang = NLLB_LANG_CODES[target_lang]

    with _translate_lock:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
        )
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    logger.info(f"[MT] translated {len(text)} -> {len(translation)} chars "
        f"({source_lang} -> {target_lang}) in {time.perf_counter() - t0:.2f}s")
    return translation
