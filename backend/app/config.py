WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

SUPPORTED_LANGUAGES = ["en", "ru", "ko"]

TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

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
