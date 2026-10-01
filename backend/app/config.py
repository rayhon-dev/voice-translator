# Whisper (STT) sozlamalari
WHISPER_MODEL_SIZE = "medium"      
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"


# Butun ilova bo'ylab tanlash uchun ruxsat etilgan tillar (frontend'dagi
# LanguageSelector.jsx bilan qo'lda sinxronlangan — ikkalasi ham shu
# uchtasini ko'rsatadi/aniqlaydi). Whisper'ning til-aniqlashini shu
# ro'yxat bilan cheklash (dialog.py'da candidate_languages sifatida
# ishlatiladi) uning o'xshash-eshitiluvchi, lekin qo'llab-quvvatlanmaydigan
# tillarga (masalan koreyschani turkchaga) adashib ketish ehtimolini
# kamaytiradi.
SUPPORTED_LANGUAGES = ["en", "ru", "ko"]


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


UNKNOWN_SPEAKER_THRESHOLD = 0.65

DIALOG_PROCESS_INTERVAL = 0.3

MIN_TURN_SECONDS = 1.0