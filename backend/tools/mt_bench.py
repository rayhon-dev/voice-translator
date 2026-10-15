"""Tarjima sinovi — DeepL (asosiy) va NLLB-600M (zaxira) ni real muammoli
misollarda yonma-yon solishtiradi, so'ng DeepL'ga NOTO'G'RI kalit bilan bir
marta sinab, avtomatik NLLB'ga o'tishni tasdiqlaydi.

Ishlatish (backend/ papkasidan, venv faollashtirilgan holda):
    python tools/mt_bench.py
"""
import sys
import time

sys.path.insert(0, ".")

from app.models import translator

# (matn, source_lang, target_lang)
TEST_CASES = [
    ("Здравствуйте", "ru", "en"),
    ("Как дела?", "ru", "en"),
    ("Всем привет! Это Ярослава, и это канал для тех, кто изучает русский язык.", "ru", "en"),
    ("где мы просто болтаем. Я рассказываю, как у меня дела.", "ru", "en"),
    ("Listenaminute.com Social issues. There seems to be so many social issues these days.", "en", "ru"),
    (
        "You know what? I'm really upset with you. You have to do it now, but you didn't do the whole project today.",
        "en",
        "ru",
    ),
    ("I'm just testing again.So show me the text", "en", "ru"),
    ("Uh, it's not, uh, it's not free. [sighs] Every request charges some amount of credit.", "en", "ru"),
    ("I think we should use Spark бэкенд for this task.", "en", "ru"),
]


def _prep(text: str) -> str:
    return translator._clean_text(translator._fix_missing_sentence_space(text))


def translate_with_deepl(text: str, source_lang: str, target_lang: str) -> str:
    return translator._translate_via_deepl(_prep(text), source_lang, target_lang)


def translate_with_nllb(text: str, source_lang: str, target_lang: str) -> str:
    return translator._translate_nllb(_prep(text), source_lang, target_lang)


def run_side_by_side():
    print("=== Tarjima: DeepL vs NLLB-600M (yonma-yon) ===")
    translator._load_deepl_client()
    translator._load_nllb_model()

    rows = []
    for text, src, tgt in TEST_CASES:
        t0 = time.perf_counter()
        try:
            deepl_out = translate_with_deepl(text, src, tgt)
            deepl_time = time.perf_counter() - t0
        except Exception as e:
            deepl_out = f"[XATO: {e!r}]"
            deepl_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        try:
            nllb_out = translate_with_nllb(text, src, tgt)
            nllb_time = time.perf_counter() - t1
        except Exception as e:
            nllb_out = f"[XATO: {e!r}]"
            nllb_time = time.perf_counter() - t1

        print(f"\n[{src}->{tgt}] {text!r}")
        print(f"  DeepL [{deepl_time:.2f}s]: {deepl_out!r}")
        print(f"  NLLB  [{nllb_time:.2f}s]: {nllb_out!r}")
        rows.append((text, src, tgt, deepl_out, deepl_time, nllb_out, nllb_time))

    print("\n\n=== Jadval (markdown) ===\n")
    print("| # | Kirish | Yo'nalish | DeepL (vaqt) | NLLB-600M (vaqt) |")
    print("|---|---|---|---|---|")
    for i, (text, src, tgt, deepl_out, deepl_time, nllb_out, nllb_time) in enumerate(rows, start=1):
        text_md = text.replace("|", "\\|")
        deepl_md = deepl_out.replace("|", "\\|")
        nllb_md = nllb_out.replace("|", "\\|")
        print(f"| {i} | {text_md} | {src}->{tgt} | {deepl_md} ({deepl_time:.2f}s) | {nllb_md} ({nllb_time:.2f}s) |")


def test_bad_key_fallback():
    print("\n\n=== Noto'g'ri DeepL kalit -> avtomatik NLLB'ga o'tish sinovi ===")
    import deepl

    original_client = translator._deepl_client
    original_failures = translator._deepl_consecutive_failures
    original_circuit = translator._deepl_circuit_open_until
    try:
        translator._deepl_client = deepl.Translator("00000000-0000-0000-0000-000000000000:fx")
        translator._deepl_consecutive_failures = 0
        translator._deepl_circuit_open_until = 0.0

        translation, provider = translator.translate_text("Hello.", "ru", "en")
        print(f"natija: provider={provider!r} translation={translation!r}")
        if provider == "nllb":
            print("PASS: noto'g'ri kalitda DeepL muvaffaqiyatsiz bo'lib, avtomatik NLLB'ga o'tdi")
        else:
            print(f"FAIL: kutilgan provider='nllb', lekin {provider!r} qaytdi")
    finally:
        translator._deepl_client = original_client
        translator._deepl_consecutive_failures = original_failures
        translator._deepl_circuit_open_until = original_circuit


if __name__ == "__main__":
    run_side_by_side()
    test_bad_key_fallback()
