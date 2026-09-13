# Whisper (STT) sozlamalari
WHISPER_MODEL_SIZE = "medium"      
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
    "en": "facebook/mms-tts-eng",

}

# Speaker identification (Resemblyzer) sozlamalari
# 1.0 = bir xil ovoz, 0.0 = butunlay farqli

# Dialog rejimida: agar eng yaqin topilgan speaker balli shundan past bo'lsa,
# bu — A yoki B emas, notanish (uchinchi) odam deb hisoblanadi.
UNKNOWN_SPEAKER_THRESHOLD = 0.65

DIALOG_PROCESS_INTERVAL = 0.3

# Juda qisqa "gap"larni (masalan tasodifiy shovqin) to'liq navbat deb hisoblamaslik uchun.
MIN_TURN_SECONDS = 1.0