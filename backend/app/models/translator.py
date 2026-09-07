import time

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.config import TRANSLATION_MODEL_NAME, NLLB_LANG_CODES

_model = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is None:
        # Must appear EXACTLY ONCE per process. NLLB-200-distilled-600M is
        # ~2.4GB in fp32; first load = disk read + weight init + CUDA transfer
        # and is the prime suspect for the "10+ seconds after stop" delay on
        # the FIRST translation of a session.
        print(f"[MT] Loading translation model {TRANSLATION_MODEL_NAME!r}...", flush=True)
        t0 = time.perf_counter()
        _tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        tok_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        _model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
        _model.to("cuda")
        mdl_s = time.perf_counter() - t1

        try:
            import torch

            dev = next(_model.parameters()).device
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            cuda_ok = torch.cuda.is_available()
        except Exception as e:  # pragma: no cover - diagnostics only
            dev, vram, cuda_ok = f"<{e!r}>", float("nan"), "?"

        print(
            f"[MT] Translation model loaded: tokenizer={tok_s:.2f}s model={mdl_s:.2f}s "
            f"device={dev} torch.cuda.is_available()={cuda_ok} "
            f"vram_allocated={vram:.2f}GB",
            flush=True,
        )
    return _model, _tokenizer


def warmup(run_inference: bool = True) -> None:
    """Load tokenizer + model onto the GPU now, and optionally translate one
    short string so the first real /translate call is fast."""
    _load_model()
    if not run_inference:
        return
    try:
        translate_text("Hello.", "uz")
        print("[MT] warm-up inference done", flush=True)
    except Exception as e:  # pragma: no cover - diagnostics only
        print(f"[MT] warm-up inference skipped: {e!r}", flush=True)


def translate_text(text: str, target_lang: str) -> str:
    model, tokenizer = _load_model()

    total_t0 = time.perf_counter()

    src_lang = NLLB_LANG_CODES["en"]
    tgt_lang = NLLB_LANG_CODES[target_lang]

    tokenizer.src_lang = src_lang

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    n_in = inputs["input_ids"].shape[-1]

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    gen_t0 = time.perf_counter()
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=512,
    )
    gen_s = time.perf_counter() - gen_t0

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    total_s = time.perf_counter() - total_t0

    print(
        f"[MT] translate: in_tokens={n_in} out_tokens={generated_tokens.shape[-1]} "
        f"generate={gen_s:.2f}s total={total_s:.2f}s "
        f"in_chars={len(text)} out_chars={len(translation)}",
        flush=True,
    )
    return translation
