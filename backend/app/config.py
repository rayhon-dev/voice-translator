import os

from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

STT_PROVIDER = os.environ.get("STT_PROVIDER", "whisper")

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

SUPPORTED_LANGUAGES = ["en", "ru", "ko"]

TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL_NAME", "facebook/nllb-200-distilled-600M")

MT_PROVIDER = os.getenv("MT_PROVIDER", "deepl")

DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY")

MT_FALLBACK_PRELOAD = os.getenv("MT_FALLBACK_PRELOAD", "true").strip().lower() == "true"

NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "uz": "uzn_Latn",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
}

TTS_MODEL_NAMES = {
    "uz": "facebook/mms-tts-uzb-script_cyrillic",
    "ko": "facebook/mms-tts-kor",
    "ru": "facebook/mms-tts-rus",
    "en": "facebook/mms-tts-eng",
}

DIALOG_PROCESS_INTERVAL = 0.3

MIN_TURN_SECONDS = 1.0

SAVE_DEBUG_AUDIO = True
DEBUG_AUDIO_DIR = "debug_audio"
