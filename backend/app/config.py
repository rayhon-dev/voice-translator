# Whisper (STT) sozlamalari
WHISPER_MODEL_SIZE = "small"      
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"


# NLLB-200 (Tarjima) sozlamalari
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# NLLB kutgan til kodlari
NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "uz": "uzn_Latn",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
}


# MMS-TTS sozlamalari
TTS_MODEL_NAMES = {
    "uz": "facebook/mms-tts-uzb-script_cyrillic",
    "ko": "facebook/mms-tts-kor",
    "ru": "facebook/mms-tts-rus",
}

# Speaker identification (Resemblyzer) sozlamalari
# 1.0 = bir xil ovoz, 0.0 = butunlay farqli. Odatda 0.75 atrofida yaxshi natija beradi.
SPEAKER_SIMILARITY_THRESHOLD = 0.70


# Umumiy sozlamalar
SUPPORTED_TARGET_LANGUAGES = ["uz", "ko", "ru"]
SOURCE_LANGUAGE = "en"  