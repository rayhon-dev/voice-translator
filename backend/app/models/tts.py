import time

from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
from app.config import TTS_MODEL_NAMES
import uroman as ur


_models = {}
_tokenizers = {}


LATIN_TO_CYRILLIC = {
    "sh": "ш", "Sh": "Ш", "SH": "Ш",
    "ch": "ч", "Ch": "Ч", "CH": "Ч",
    "yo": "ё", "Yo": "Ё", "YO": "Ё",
    "yu": "ю", "Yu": "Ю", "YU": "Ю",
    "ya": "я", "Ya": "Я", "YA": "Я",

    # "o" va "g" dan keyin har xil apostrof variantlari (ʻ, ', ', ')
    "oʻ": "ў", "o'": "ў", "o‘": "ў", "o’": "ў",
    "Oʻ": "Ў", "O'": "Ў", "O‘": "Ў", "O’": "Ў",
    "gʻ": "ғ", "g'": "ғ", "g‘": "ғ", "g’": "ғ",
    "Gʻ": "Ғ", "G'": "Ғ", "G‘": "Ғ", "G’": "Ғ",

    "a": "а", "b": "б", "d": "д", "e": "е", "f": "ф",
    "g": "г", "h": "ҳ", "i": "и", "j": "ж", "k": "к",
    "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "x": "х", "y": "й", "z": "з",
    "A": "А", "B": "Б", "D": "Д", "E": "Е", "F": "Ф",
    "G": "Г", "H": "Ҳ", "I": "И", "J": "Ж", "K": "К",
    "L": "Л", "M": "М", "N": "Н", "O": "О", "P": "П",
    "Q": "Қ", "R": "Р", "S": "С", "T": "Т", "U": "У",
    "V": "В", "X": "Х", "Y": "Й", "Z": "З",
}


MULTI_CHAR_KEYS = sorted(
    [k for k in LATIN_TO_CYRILLIC if len(k) > 1],
    key=len,
    reverse=True,
)


def transliterate_uz_to_cyrillic(text: str) -> str:
    result = ""
    i = 0
    while i < len(text):
        matched = False
        for key in MULTI_CHAR_KEYS:
            if text[i:i + len(key)] == key:
                result += LATIN_TO_CYRILLIC[key]
                i += len(key)
                matched = True
                break
        if not matched:
            char = text[i]
            result += LATIN_TO_CYRILLIC.get(char, char)
            i += 1
    return result


def _load_model(lang: str):
    if lang not in _models:
        model_name = TTS_MODEL_NAMES[lang]
        _tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
        _models[lang] = VitsModel.from_pretrained(model_name).to("cuda")
    return _models[lang], _tokenizers[lang]


_uroman = ur.Uroman()

def romanize_korean(text: str) -> str:
    return _uroman.romanize_string(text)


def text_to_speech(text: str, lang: str) -> tuple[np.ndarray, int]:
    t0 = time.perf_counter()
    text_len = len(text)

    if lang == "uz":
        text = transliterate_uz_to_cyrillic(text)
    elif lang == "ko":
        text = romanize_korean(text)

    model, tokenizer = _load_model(lang)

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model(**inputs).waveform

    audio_array = output.squeeze().cpu().numpy()
    sample_rate = model.config.sampling_rate

    print(
        f"[TTS] lang={lang} {text_len} chars -> "
        f"{len(audio_array) / sample_rate:.1f}s audio in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    return audio_array, sample_rate


def warmup(run_inference: bool = True) -> None:
    samples = {"uz": "Salom", "ko": "안녕", "ru": "Привет"}
    for lang, sample_text in samples.items():
        try:
            if run_inference:
                text_to_speech(sample_text, lang)
            else:
                _load_model(lang)
            print(f"[TTS] warm-up for {lang!r} done", flush=True)
        except Exception as e:  
            print(f"[TTS] warm-up for {lang!r} skipped: {e!r}", flush=True)