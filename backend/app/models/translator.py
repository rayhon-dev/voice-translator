import time

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.config import TRANSLATION_MODEL_NAME, NLLB_LANG_CODES

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        # Loaded once per process 
        t0 = time.perf_counter()
        _tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
        _model.to("cuda")
        print(
            f"[MT] translation model loaded in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )
    return _model, _tokenizer


def warmup(run_inference: bool = True) -> None:
    _load_model()
    if not run_inference:
        return
    try:
        translate_text("Hello.", "uz")
        print("[MT] warm-up inference done", flush=True)
    except Exception as e: 
        print(f"[MT] warm-up inference skipped: {e!r}", flush=True)


def translate_text(text: str, target_lang: str) -> str:
    model, tokenizer = _load_model()

    t0 = time.perf_counter()

    src_lang = NLLB_LANG_CODES["en"]
    tgt_lang = NLLB_LANG_CODES[target_lang]

    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=512,
    )
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    print(
        f"[MT] translated {len(text)} -> {len(translation)} chars "
        f"in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    return translation
