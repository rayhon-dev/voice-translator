import os

from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# STT provider tanlovi: "whisper" (lokal, default) yoki "elevenlabs" (Scribe v2
# API orqali) — ikkisini kodni qayta yozmasdan solishtirib sinab ko'rish uchun.
# stt.py'dagi transcribe_audio() shu qiymatga qarab ichki ravishda tanlaydi,
# dialog.py/speaker_diart.py esa buni bilishi/o'zgarishi shart emas.
STT_PROVIDER = os.environ.get("STT_PROVIDER", "whisper")

# .env (yoki muhit o'zgaruvchisi) orqali o'qiladi — repo'ga YOZILMAYDI
# (backend/.env .gitignore'da). STT_PROVIDER="elevenlabs" bo'lganda kerak.
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

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

# Diagnostika uchun: yoqilganda, spiker-aniqlash embedding modeliga AYNAN
# uzatiladigan (trim qilingan) har bir audio kesmasi DEBUG_AUDIO_DIR'ga
# .wav fayl sifatida yoziladi (speaker_diart.py'ga qarang). Faqat lokal
# tekshirish/debug uchun — production'da O'CHIRILGAN bo'lishi kerak,
# shuning uchun default False. Yoqilsa, disk yozish ASOSIY oqimni
# bloklamasligi uchun alohida background thread'da amalga oshiriladi.
SAVE_DEBUG_AUDIO = True
DEBUG_AUDIO_DIR = "debug_audio"
