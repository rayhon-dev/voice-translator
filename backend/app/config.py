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

# Tarjima: DeepL (API, asosiy) + NLLB-600M (lokal, zaxira — internet yo'q,
# DeepL xato bergan/limit tugagan hollarda). 1.3B/CTranslate2 yo'li OLIB
# TASHLANDI (xotira/konvertatsiya muammolari — qarang git tarixi) — NLLB
# ENDI faqat ZAXIRA, shuning uchun eski, kichik 600M modelga qaytarildi.
TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL_NAME", "facebook/nllb-200-distilled-600M")

# "deepl" (default — DeepL API asosiy, NLLB-600M FAQAT zaxira sifatida) yoki
# "nllb" (DeepL UMUMAN ishlatilmaydi, FAQAT lokal NLLB-600M). translator.py
# shu qiymatga qarab tanlaydi — qarang undagi _deepl_attempt()/_translate_nllb().
MT_PROVIDER = os.getenv("MT_PROVIDER", "deepl")

# .env orqali o'qiladi — repo'ga YOZILMAYDI (backend/.env .gitignore'da).
# MT_PROVIDER="deepl" bo'lganda kerak (yo'q bo'lsa — FAQAT NLLB ishlatiladi,
# avtomatik, xatosiz).
DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY")

# true (default) — server ishga tushganda NLLB-600M OLDINDAN yuklanadi
# (DeepL muvaffaqiyatsiz bo'lgan BIRINCHI turnda ham zaxira DARHOL tayyor
# bo'lishi uchun, model yuklanishini kutib turmasdan). false — birinchi
# haqiqiy zaxira-chaqiruvigacha yuklanmaydi (VRAM/vaqtni tejash uchun,
# masalan DeepL hech qachon ishlamay qolmasligi kutilganda).
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

# Diagnostika uchun: yoqilganda, spiker-aniqlash embedding modeliga AYNAN
# uzatiladigan (trim qilingan) har bir audio kesmasi DEBUG_AUDIO_DIR'ga
# .wav fayl sifatida yoziladi (speaker_diart.py'ga qarang). Faqat lokal
# tekshirish/debug uchun — production'da O'CHIRILGAN bo'lishi kerak,
# shuning uchun default False. Yoqilsa, disk yozish ASOSIY oqimni
# bloklamasligi uchun alohida background thread'da amalga oshiriladi.
SAVE_DEBUG_AUDIO = True
DEBUG_AUDIO_DIR = "debug_audio"
