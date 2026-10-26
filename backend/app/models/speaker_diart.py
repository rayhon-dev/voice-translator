import datetime
import glob
import json
import os
import threading
import time

import numpy as np
import soundfile as sf
import torch

from app.config import SAVE_DEBUG_AUDIO, DEBUG_AUDIO_DIR
from app.gpu_lock import GPU_LOCK
import logging

logger = logging.getLogger(__name__)


RMS_SILENCE_THRESHOLD = 0.003  # stt.py bilan bir xil qiymat
_TRIM_FRAME_MS = 50

# _trim_trailing_silence() 50ms'lik LAHZAVIY kadr RMS'ini tekshiradi
MIN_TRIMMED_DURATION_S = 0.5

# ---------------------------------------------------------------------------
# OVOZ EMBEDDING MODELI — ENDI diart ORQALI EMAS, TO'G'RIDAN-TO'G'RI
# pyannote.audio (Model.from_pretrained + Inference(window="whole")) orqali
# yuklanadi. diart paketi hamon requirements.txt'da qoladi (boshqa
# paketlar — xususan pyannote.audio — unga bog'liq bo'lishi mumkin, shuning
# uchun pip uninstall QILINMAYDI), lekin shu fayl endi uni IMPORT ham
# QILMAYDI — kodda diart'ga hech qanday bog'liqlik qolmagan.
#
# Standart model: "pyannote/wespeaker-voxceleb-resnet34-LM" — ilgari
# ishlatilgan "pyannote/embedding" (eski, kichikroq model)dan farqli,
# WeSpeaker ResNet34 — zamonaviy, kattaroq, aniqroq speaker-verification
# modeli (256-o'lchamli embedding). Agar yuklash muvaffaqiyatsiz bo'lsa
# (masalan tarmoq/HF Hub muammosi) — zaxira sifatida speechbrain'ning
# ECAPA-TDNN modeli (speechbrain/spkrec-ecapa-voxceleb, 192-o'lchamli)
# ishlatiladi. Ikkisi ham .env orqali sozlanishi mumkin.
SPK_EMBEDDING_MODEL = os.environ.get("SPK_EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM")
SPK_EMBEDDING_FALLBACK_MODEL = os.environ.get("SPK_EMBEDDING_FALLBACK_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
# Ba'zi HF Hub modellari (gated repolar) uchun token kerak bo'lishi mumkin —
# wespeaker-voxceleb-resnet34-LM ochiq litsenziyali va TOKENSIZ yuklanadi
# (tekshirildi), lekin agar kelajakda boshqa (gated) model tanlansa yoki HF
# Hub so'rov chegarasi tufayli muammo chiqsa, shu orqali beriladi.
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)

# CPU'da bitta embedding 130-320ms olardi — Whisper/NLLB allaqachon GPU
# ishlatayotgan bo'lsa ham, bitta jarayonda ikkisi BIR XIL GPU'ni navbat
# bilan ishlatadi (umumiy VRAM yetarli bo'lsa muammo emas), shuning uchun
# CUDA mavjud bo'lsa ENDI SHU model ham GPU'ga o'tkaziladi. SPK_DEVICE
# orqali qo'lda CPU'ga majburlash ham mumkin (masalan VRAM tanqis bo'lsa).
SPK_DEVICE = os.environ.get("SPK_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

_embedding_backend: str | None = None  # "pyannote" | "speechbrain"
_embedding_model = None  # pyannote Inference OBYEKTI yoki speechbrain EncoderClassifier
_embedding_model_name: str | None = None
_embedding_load_lock = threading.Lock()

# ---------------------------------------------------------------------------
# ONLAYN SPIKER KLASTERLASH — cheklanmagan sonli spiker, FAQAT ovoz asosida
# (til butunlay chiqarib tashlandi — endi faqat tarjima yo'nalishi uchun,
# dialog.py'da ishlatiladi). Har bir spiker uchun "prototiplar" ro'yxati
# saqlanadi (bitta o'rtacha markaz O'RNIGA) — o'xshashlik shu spikerning
# BARCHA prototiplari bilan solishtirib, ENG KATTASI sifatida hisoblanadi
# (nearest-neighbor, emotsiya/intonatsiya bo'yicha ovoz o'zgarishiga
# chidamliroq). Barcha qiymatlar .env orqali sozlanadi — real suhbat
# loglaridan ([SPK] qatoriga qarang) moslashtirish uchun.
#
# REAL TESTDA TOPILGAN 2 XATO (ikkisi ham shu versiyada TUZATILDI):
#   1. Eski "pending/tasdiqlash" nomzod mexanizmi (yangi spikerni DARHOL
#      ochish o'rniga, keyingi turn tasdiqlashini kutish) `next_id`ni FAQAT
#      tasdiqlangandan keyin oshirardi — agar TASDIQLANMAGAN nomzoddan
#      keyin BOSHQA (mos kelmaydigan) odam gapirsa, ikkinchisi xuddi SHU
#      (hali band qilinmagan) raqamli YANGI nomzod sifatida ochilib,
#      birinchisining o'rnini bosib ketardi — ikki xil odam bitta raqamga
#      "sig'ib" qolardi. Mexanizm BUTUNLAY olib tashlandi: endi
#      `best < SPK_NEW` va nutq yetarli uzun bo'lsa, YANGI spiker DARHOL
#      (tasdiqlashsiz) ochiladi.
#   2. Profilga YANGI prototip qo'shish uchun chegara (SPK_ENROLL_MIN_SIM)
#      spiker hali EMAS/kam sonli prototiplarga ega bo'lgandayam baland
#      edi — bitta "qiyin" founding namunadan keyin profil uzoq vaqt
#      o'smay qolishi mumkin edi. SPK_BOOTSTRAP_PROTOTYPES soni (default 3)
#      prototiplar yetib kelmaguncha, SPK_ACCEPT'dan yuqori HAR QANDAY
#      turn profilni to'ldirishga yordam beradi — shundan keyin ANIQ
#      SPK_ENROLL_MIN_SIM chegarasi amal qiladi.
# ---------------------------------------------------------------------------

# Shu va undan yuqori o'xshashlikda turn SHU spikerga ANIQ tegishli deb
# qabul qilinadi.
SPK_ACCEPT = float(os.environ.get("SPK_ACCEPT", "0.55"))

# REAL TESTDA TOPILGAN MUAMMO (teng/yaqin ikki nomzod): `best_sim >=
# SPK_ACCEPT` o'zi YETARLI EMAS — agar IKKINCHI eng yaqin spiker ham
# deyarli XUDDI SHUNDAY o'xshashlikka ega bo'lsa (masalan sims={1:0.58,
# 3:0.58} — ikkisi TENG), "eng kattasi" shunchaki tasodifiy tanlov bo'lib
# qoladi, aslida qaysi odam ekani ANIQ EMAS. Shu sababli endi `accept`
# qarori QO'SHIMCHA shartni talab qiladi: eng yaqin ikki nomzod orasidagi
# FARQ (margin) kamida shuncha bo'lishi SHART — aks holda spiker
# "tie_uncertain" (eng yaqinga TAYINLANADI, lekin uncertain=True, prototip
# QO'SHILMAYDI). Bitta spiker mavjud bo'lganda (ikkinchisi yo'q) margin
# cheksiz (-1.0'dan farq) hisoblanadi — bitta nomzod bilan "teng taqqoslash"
# ma'nosiz, chegara UMUMAN qo'llanmaydi.
SPK_MIN_MARGIN = float(os.environ.get("SPK_MIN_MARGIN", "0.08"))
# Shundan past o'xshashlikda turn HECH QANDAY mavjud spikerga tegishli
# EMAS deb hisoblanadi (yangi ovoz nomzodi). Real logdan: bir xil odamning
# ENG PAST o'xshashligi 0.66, TURLI odamlarniki ENG KO'PI 0.33 edi — 0.35
# bu ikkisi orasida juda chekka (deyarli turli-odam tomonga yopishgan)
# turardi, 0.40'ga ko'tarildi.
SPK_NEW = float(os.environ.get("SPK_NEW", "0.40"))
# Yangi spiker (yoki yangi prototip) uchun "nutq" (trim qilingandan keyingi,
# haqiqiy ovoz bo'lgan) davomiyligi kamida shuncha bo'lishi SHART — qisqa
# namunalar ovoz profilini "ifloslantirishi" (noaniq/past-sifatli embedding)
# xavfi yuqori.
SPK_MIN_ENROLL_SPEECH_S = float(os.environ.get("SPK_MIN_ENROLL_SPEECH_S", "2.0"))
# Bootstrap can use a confident short answer without relaxing later enrollment.
SPK_BOOTSTRAP_MIN_SPEECH_S = float(os.environ.get("SPK_BOOTSTRAP_MIN_SPEECH_S", "1.5"))
SPK_NEW_IMMEDIATE_PAIR_MASS = float(os.environ.get("SPK_NEW_IMMEDIATE_PAIR_MASS", "0.95"))
SPK_NEW_IMMEDIATE_NEAR_RATIO = float(os.environ.get("SPK_NEW_IMMEDIATE_NEAR_RATIO", "0.75"))
# Har bir spiker uchun saqlanadigan ENG KO'P prototiplar soni — birinchi
# ("founding") prototip HAR DOIM saqlanadi, undan oshganda eng ESKI
# (lekin birinchi EMAS) prototip olib tashlanadi (FIFO).
SPK_MAX_PROTOTYPES = int(os.environ.get("SPK_MAX_PROTOTYPES", "10"))
# Shundan QISQAROQ (xom, trim qilinmagan) turn uchun — agar hech kimga
# ANIQ mos kelmasa ("best < SPK_ACCEPT") — o'xshashlikdan ko'ra ENG SO'NGGI
# gapirgan spikerga tegishli deb hisoblash ko'proq ishonchli (juda qisqa
# "ha"/"mhm" kabi javoblarda ovoz-o'xshashlik o'zi ishonchsiz bo'ladi).
SPK_SHORT_TURN_S = float(os.environ.get("SPK_SHORT_TURN_S", "1.5"))

# Prototipga aylanish uchun embedding YANGI prototip sifatida qo'shiladigan
# eng YUQORI o'xshashlik — bundan baland bo'lsa, bu amalda MAVJUD
# prototiplardan biriga "deyarli bir xil" (takroriy namuna), qo'shishning
# hojati yo'q.
SPK_DUPLICATE_PROTOTYPE_SIM = 0.95

# REAL TESTDA TOPILGAN MUAMMO: ilgari SPK_ACCEPT (0.55)dan yuqori HAR
# QANDAY turn (hatto chegaraga yaqin, masalan 0.56) ham yangi prototip
# sifatida qo'shilardi — agar bu turn aslida XATO qabul qilingan bo'lsa
# (masalan boshqa odam SPK_ACCEPT'ni arang o'tib ketsa), noto'g'ri
# prototip profilni "ifloslantirib", keyingi turnlar uchun aniqlikni
# yomonlashtirardi. Endi prototip qo'shish uchun ANCHA qattiqroq, alohida
# chegara — SPK_ACCEPT'dan yuqori bo'lsa ham, SPK_ENROLL_MIN_SIM'dan past
# o'xshashlik FAQAT spikerga TAYINLASH uchun yetarli, lekin ovoz profilini
# YANGILASH uchun YETARLI EMAS deb hisoblanadi. BUNING YAGONA ISTISNOSI —
# pastdagi SPK_BOOTSTRAP_PROTOTYPES (profil hali YETARLICHA prototipga ega
# bo'lmaganda, bu chegara vaqtincha qo'llanmaydi).
SPK_ENROLL_MIN_SIM = float(os.environ.get("SPK_ENROLL_MIN_SIM", "0.65"))

# REAL TESTDA TOPILGAN MUAMMO: spiker hali FAQAT 1 ta ("founding")
# prototipga ega bo'lganda, agar o'sha YAGONA namuna tasodifan "qiyin"
# (odamning o'z ovoz diapazonidagi chekka holatini aks ettiruvchi) bo'lsa,
# SPK_ENROLL_MIN_SIM (0.65) talabi KEYINGI haqiqiy namunalar tomonidan
# ham uzoq vaqt qondirilmay qolishi mumkin edi — profil "qotib" qolardi
# (1 ta prototipdan o'smasdi). Shu sababli: profilda SPK_BOOTSTRAP_
# PROTOTYPES'dan KAM prototip bo'lsa, SPK_ACCEPT'dan yuqori HAR QANDAY
# turn (SPK_ENROLL_MIN_SIM'ni tekshirmasdan) profilni to'ldirishga
# yordam beradi — shu dastlabki bosqichdan keyin ANIQ SPK_ENROLL_MIN_SIM
# chegarasi amal qiladi.
SPK_BOOTSTRAP_PROTOTYPES = int(os.environ.get("SPK_BOOTSTRAP_PROTOTYPES", "3"))
# REAL TESTDA TOPILGAN MUAMMO (bitta odam ikki spikerga bo'lindi): 1-spiker
# 1.57s'lik BITTA "Hello, how are you doing?" namunasidan ochilgan, shu
# odamning keyingi uzun gaplari esa 0.53/0.49 o'xshashlik bilan "uncertain"
# bo'lib qolib, profilga HECH QACHON qo'shilmagan — profil 1 prototipda
# qotib qoldi va keyinroq o'sha odam boshqa (3 prototipli) spikerga
# tayinlandi. Profil hali bootstrap bosqichida bo'lsa (< SPK_BOOTSTRAP_
# PROTOTYPES), shu chegaradan yuqori, aniq margin'li, uzun va toza turn
# qabul qilinadi va prototip sifatida qo'shiladi.
SPK_BOOTSTRAP_ACCEPT = float(os.environ.get("SPK_BOOTSTRAP_ACCEPT", "0.45"))
# REAL TESTDA: spikerning QISQA (< SPK_MIN_ENROLL_SPEECH_S) davom gapi
# (masalan 1.8s "I wish you were different.") embedding'i ishonchsiz
# bo'lgani uchun o'z spikeriga ham 0.35 o'xshashlik bergan va "Aniqlanmoqda…"
# bo'lib qolgan, garchi boshqa hamma spikerlardan aniq yaqinroq bo'lsa ham.
# Eng yaqin spiker OXIRGI gapirgan spiker bo'lsa, margin yetarli bo'lsa va
# o'xshashlik shundan yuqori bo'lsa — shu spikerga (uncertain, prototip
# qo'shilmaydi) tayinlanadi.
SPK_SHORT_CONTINUE_SIM = float(os.environ.get("SPK_SHORT_CONTINUE_SIM", "0.30"))
# 2.3s'lik "Sheila." ham (oxirgi spikerga 0.39, boshqalarga ≤0.16) shu
# sababli "Aniqlanmoqda…" bo'lib qolgan — "qisqa" chegarasi 3s.
SPK_SHORT_CONTINUE_MAX_SPEECH_S = float(os.environ.get("SPK_SHORT_CONTINUE_MAX_SPEECH_S", "3.0"))

# REAL TESTDA TOPILGAN MUAMMO: yuqoridagi tuzatishlardan OLDIN, bitta
# tuzatish (correct_speaker) xato spikerga qilingandan keyin ("Bir xil
# spikerga tuzatish profilni ifloslantirgan") ikki TURLI raqamli spiker
# (masalan 2 va 3) AMALDA bir xil odamning ovozi bo'lib qolgan edi
# (sims={2:0.83, 3:0.83}). Bunday holatlarni AVTOMATIK tuzatish uchun:
# har bir enrollment/tuzatishdan KEYIN, ixtiyoriy ikki spiker prototiplari
# orasidagi ENG KATTA o'xshashlik shu chegaradan yuqori bo'lsa, ikkisi
# BITTA spikerga (kichik raqamliga) birlashtiriladi (qarang
# _check_and_merge). Bu — profilni "tozalab turadigan" xavfsizlik to'ri,
# yuqoridagi tuzatishlar asosiy sabablarni yo'qotgandan keyin ham foydali
# (masalan foydalanuvchi qo'lda noto'g'ri tuzatish qilsa).
SPK_MERGE_SIM = float(os.environ.get("SPK_MERGE_SIM", "0.65"))

# Mentor demo'da topilgan 7-muammo (B-guruh, qism-segmentatsiya): agar
# shu turn/qism segmenter.segment_turn() tomonidan "iflos" deb baholangan
# bo'lsa (purity — qismning o'z ICHIDAGI oynalar orasidagi eng past
# o'xshashlik, ehtimol IKKI ovoz aralashgani uchun past) — bu qism HALI
# HAM spikerga TAYINLANADI (ovoz-o'xshashlik bo'yicha), lekin ovoz
# PROFILINI (YANGI prototip SIFATIDA) ifloslantirmasligi uchun enrollment
# o'tkazib yuboriladi. Segmentatsiya ishlatilmagan (yoki bitta qismli)
# holatlarda purity=1.0 (qarang segmenter._single_segment) — bu chegara
# HECH QACHON ESKI xatti-harakatga ta'sir qilmaydi.
SPK_ENROLL_MIN_PURITY = float(os.environ.get("SPK_ENROLL_MIN_PURITY", "0.6"))

# ---------------------------------------------------------------------------
# Mentor demo'da topilgan 4-muammo (shovqinli xona): fondagi begona
# nutqning BA'ZILARI (pair_mass MIN_PAIR_MASS'dan tasodifan o'tgan, YOKI
# near-field filtri hali yo'q/zaif bo'lgan paytda) "yangi spiker" sifatida
# OCHILIB qolardi — bu SOXTA spiker. Yangi spiker (yoki "unknown"
# buferidan ko'tarish orqali yangi spiker) ochish endi QO'SHIMCHA ikki
# dalilni talab qiladi (mavjud purity/davomiylik talablariga QO'SHIMCHA):
#   - near_field_ratio (dialog.py hisoblaydi, FAR_FIELD_TURN_RATIO bilan
#     BIR XIL "nutq RMS / near_ref" nisbati) >= SPK_NEW_MIN_NEAR_FIELD_RATIO
#     — fondagi (uzoqroq) ovoz YANGI identity sifatida "tasdiqlanmaydi".
#   - pair_mass (stt.py) >= SPK_NEW_MIN_PAIR_MASS — oddiy MIN_PAIR_MASS
#     (0.65)dan YUQORI, chunki "yangi spiker ochish" (profilga umrbod
#     ta'sir qiladigan) qaror MIN_PAIR_MASS'dan ko'ra ko'proq ishonch
#     talab qiladi.
# Ikkisi ham `None` bo'lsa (masalan near_ref hali kalibrlanmagan — sovuq
# start, YOKI dialog.py eski chaqiruv uslubida pair_mass uzatmasa) — mos
# tekshiruv O'TKAZIB YUBORILADI (xavfsiz standart — eski xatti-harakat
# o'zgarmaydi, "Umumiy" talabiga mos).
SPK_NEW_MIN_NEAR_FIELD_RATIO = float(os.environ.get("SPK_NEW_MIN_NEAR_FIELD_RATIO", "0.5"))
SPK_NEW_CONFIRM_TURNS = max(2, int(os.environ.get("SPK_NEW_CONFIRM_TURNS", "2")))
SPK_NEW_MIN_PAIR_MASS = float(os.environ.get("SPK_NEW_MIN_PAIR_MASS", "0.90"))


def _new_speaker_evidence(purity: float, near_field_ratio: float | None, pair_mass: float | None):
    """Yangi spiker (yoki bufer-promote) ochish uchun dalil YETARLI
    ekanini tekshiradi — qarang SPK_NEW_MIN_NEAR_FIELD_RATIO izohi
    yuqorida. Qaytaradi: (ok: bool, reasons: list[str]) — `reasons`
    FAQAT ok=False bo'lganda to'ldiriladi, [SPK] logida ko'rsatish uchun."""
    reasons = []
    if purity < SPK_ENROLL_MIN_PURITY:
        reasons.append(f"purity={purity:.3f}<{SPK_ENROLL_MIN_PURITY}")
    if near_field_ratio is not None and near_field_ratio < SPK_NEW_MIN_NEAR_FIELD_RATIO:
        reasons.append(f"near_field_ratio={near_field_ratio:.3f}<{SPK_NEW_MIN_NEAR_FIELD_RATIO}")
    if pair_mass is not None and pair_mass < SPK_NEW_MIN_PAIR_MASS:
        reasons.append(f"pair_mass={pair_mass:.3f}<{SPK_NEW_MIN_PAIR_MASS}")
    return (not reasons), reasons

# ---------------------------------------------------------------------------
# Mentor demo'da topilgan 1/2-muammolar: "NOMA'LUM" bo'lak bufer mexanizmi.
#
# 1-muammo: ilgari qisqa (< SPK_SHORT_TURN_S) bo'lak uchun, agar best_sim
# SPK_ACCEPT'dan past bo'lsa — SOHA NAZARIDAN farqi (masalan best_sim=0.00
# — ANIQ boshqa odam) HISOBGA OLINMASDAN, SO'ZSIZ `session["last_speaker"]`ga
# (uncertain=True bilan) berib yuborilardi. Natijada sessiyada 3-chi odam
# gapirsa ham, u 2-spiker bo'lib "yopishib" qolardi.
#
# 2-muammo: shu bilan BOG'LIQ — agar bo'lak QISQA bo'lsa VA hali birorta
# ham spiker yo'q bo'lsa (sessiya boshi) YOKI best_sim juda past bo'lsa,
# eski kod PROTOTIPSIZ ("new_short") spiker ochardi — bu spiker keyin
# HECH QACHON hech kimga mos kelmasdi (sims=-1.00 abadiy), shuning uchun
# har bir keyingi "last_speaker" fallback shu bo'sh spikerga yopishib
# qolishi mumkin edi.
#
# YECHIM: best_sim < SPK_NEW (ya'ni ANIQ, hech kimga mos kelmaydi) VA
# nutq SPK_MIN_ENROLL_SPEECH_S'dan qisqa bo'lgan bo'laklar ENDI HECH
# QANDAY spikerga (yangi HAM, mavjud HAM) DARHOL tayinlanmaydi — ular
# "unknown" (speaker=None, uncertain=True) deb qaytariladi va shu
# embedding SPK_UNKNOWN_BUFFER_WINDOW_S oyna ichida saqlanadigan
# `session["unknown_buffer"]`ga qo'shiladi. Har safar bufer yangilanganda:
#   a) Bufer ichidagi har bir yozuv MAVJUD spikerlar bilan qayta
#      solishtiriladi — agar endi (masalan shu orada profil kattalashgani
#      uchun) SPK_ACCEPT'dan yuqori mos kelsa, o'sha spikerga ko'chiriladi.
#   b) Qolgan yozuvlar o'zaro (juft-jufti bilan) solishtiriladi — agar
#      bir nechtasi bir-biriga SPK_ACCEPT'dan yuqori o'xshasa VA ularning
#      JAMI nutqi SPK_MIN_ENROLL_SPEECH_S'dan oshsa — bular BIRGALIKDA
#      YANGI spikerga aylanadi (hech qachon PROTOTIPSIZ — kamida ikkita
#      prototip bilan).
# Ikkisi ham `identify_speaker()`ga `reassigned: [(seq, new_speaker), ...]`
# ro'yxatini qaytaradi — dialog.py buni ko'rib, allaqachon frontendga
# "unknown" sifatida yuborilgan bubble(lar)ni YANGI spiker bilan qayta
# yuboradi (upsert).
# ---------------------------------------------------------------------------
SPK_UNKNOWN_BUFFER_WINDOW_S = float(os.environ.get("SPK_UNKNOWN_BUFFER_WINDOW_S", "30"))


def _trim_trailing_silence(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    #Audio oxiridagi jim (RMS past) qismni kesib tashlaydi.
    frame_len = max(1, int(sample_rate * _TRIM_FRAME_MS / 1000))
    n = len(wav)
    end = n
    while end > 0:
        start = max(0, end - frame_len)
        frame = wav[start:end]
        rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
        if rms >= RMS_SILENCE_THRESHOLD:
            break
        end = start
    return wav[:end]


def _load_pyannote_model(model_name: str):
    from pyannote.audio import Model, Inference

    model = Model.from_pretrained(model_name, token=HF_TOKEN)
    model.eval()
    inference = Inference(model, window="whole")
    # Inference.infer() ICHKI ravishda kiruvchi audio'ni `self.device`ga
    # ko'chiradi va natijani CPU'ga qaytaradi (pyannote.audio manbasida
    # tekshirildi) — shuning uchun bu yerda FAQAT `.to()` chaqirish
    # YETARLI, _embed()da qo'lda device ko'chirish shart emas.
    inference.to(torch.device(SPK_DEVICE))
    return inference


def _load_speechbrain_model(model_name: str):
    from speechbrain.inference.speaker import EncoderClassifier

    savedir = os.path.join("pretrained_models", model_name.replace("/", "_"))
    return EncoderClassifier.from_hparams(
        source=model_name,
        run_opts={"device": SPK_DEVICE},
        savedir=savedir,
    )


def _load_model():
    """Jarayonga BIR MARTA — avval SPK_EMBEDDING_MODEL (standart:
    pyannote/wespeaker-voxceleb-resnet34-LM) pyannote.audio orqali
    to'g'ridan-to'g'ri (diart'siz) yuklashga harakat qiladi; muvaffaqiyatsiz
    bo'lsa (tarmoq/HF Hub xatosi, gated-repo/token muammosi va h.k.) —
    SPK_EMBEDDING_FALLBACK_MODEL (standart: speechbrain/spkrec-ecapa-
    voxceleb) orqali zaxira sifatida yuklanadi. Ikkisi ham butunlay
    boshqa Python API'ga ega bo'lgani uchun `_embedding_backend` orqali
    qaysi biri yuklanganini `_embed()` keyinroq biladi."""
    global _embedding_backend, _embedding_model, _embedding_model_name
    if _embedding_model is not None:
        return _embedding_model

    with _embedding_load_lock:
        if _embedding_model is not None:
            return _embedding_model

        try:
            _embedding_model = _load_pyannote_model(SPK_EMBEDDING_MODEL)
            _embedding_backend = "pyannote"
            _embedding_model_name = SPK_EMBEDDING_MODEL
        except Exception as e:
            logger.error(
                f"[SPK] {SPK_EMBEDDING_MODEL!r} modelini pyannote.audio orqali yuklab bo'lmadi: "
                f"{e!r} — zaxira modelga ({SPK_EMBEDDING_FALLBACK_MODEL!r}, speechbrain) o'tilyapti. "
                f"Agar bu gated/xususiy HF Hub repo bo'lsa, HF_TOKEN muhit o'zgaruvchisini "
                f"sozlash kerak bo'lishi mumkin."
            )
            _embedding_model = _load_speechbrain_model(SPK_EMBEDDING_FALLBACK_MODEL)
            _embedding_backend = "speechbrain"
            _embedding_model_name = SPK_EMBEDDING_FALLBACK_MODEL

        return _embedding_model


def _clear_debug_audio_dir() -> None:
    """Server ishga tushganda (faqat SAVE_DEBUG_AUDIO yoqilgan bo'lsa)
    debug_audio/ papkasidagi OLDINGI ishga tushishdan qolgan .wav
    fayllarni tozalaydi — aks holda bu fayllar vaqt o'tishi bilan
    cheksiz ko'payib, diskni to'ldirib yuboradi."""
    if not SAVE_DEBUG_AUDIO:
        return
    os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(DEBUG_AUDIO_DIR, "*.wav"))
    for path in existing:
        try:
            os.remove(path)
        except OSError as e:
            logger.info(f"[SPK][DEBUG_AUDIO] could not remove old file {path!r}: {e!r}")
    logger.info(f"[SPK][DEBUG_AUDIO] SAVE_DEBUG_AUDIO=True — {len(existing)} eski fayl(lar) "
        f"{DEBUG_AUDIO_DIR!r}'dan tozalandi, YANGI debug audio shu papkaga yoziladi. "
        f"ESLATMA: bu faqat diagnostika rejimi — production'da SAVE_DEBUG_AUDIO=False "
        f"bo'lishi kerak.")


def _save_debug_audio_async(
    wav: np.ndarray, sample_rate: int, duration_s: float, rms: float, outcome: str
) -> str | None:
    """Embedding modelga AYNAN uzatiladigan (trim qilingan) audio'ni
    diagnostika uchun diskka yozadi — faqat SAVE_DEBUG_AUDIO=True bo'lsa.
    Asosiy oqimni bloklamasligi uchun alohida (daemon) thread'da ishlaydi —
    bu funksiyaning o'zi disk I/O'ni kutmasdan darhol qaytadi."""
    if not SAVE_DEBUG_AUDIO:
        return None

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{ts}_dur{duration_s:.2f}s_rms{rms:.4f}_{outcome}.wav"

    def _write():
        try:
            os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
            sf.write(os.path.join(DEBUG_AUDIO_DIR, filename), wav, sample_rate)
        except Exception as e:
            logger.error(f"[SPK][DEBUG_AUDIO] failed to save debug audio: {e!r}")

    threading.Thread(target=_write, daemon=True).start()
    return filename


def save_turn_label(wav_filename: str | None, speaker, lang: str | None, text: str | None) -> None:
    """dialog.py tomonidan, bitta turn uchun YAKUNIY natija (spiker ID, til,
    transkripsiya) ma'lum bo'lgandan keyin chaqiriladi. `wav_filename` —
    identify_speaker() debug_label_out orqali qaytargan, AYNAN shu turnga
    tegishli audio fayl nomi (_save_debug_audio_async natijasi). Shu nom bilan
    bir xil, lekin .json kengaytmali faylga {"speaker", "lang", "text"} yoziladi
    — eval_dialog.py labels.json shablonini shundan avtomatik to'ldiradi
    (foydalanuvchi faqat xatolarni tuzatadi).

    `wav_filename=None` bo'lsa (SAVE_DEBUG_AUDIO=False edi, YOKI audio juda
    qisqa/jim bo'lgani uchun identify_speaker hech qanday wav saqlamagan
    edi) — yozadigan narsa yo'q, hech narsa qilinmaydi."""
    if not SAVE_DEBUG_AUDIO or not wav_filename:
        return

    def _write():
        try:
            os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
            label_filename = os.path.splitext(wav_filename)[0] + ".json"
            with open(os.path.join(DEBUG_AUDIO_DIR, label_filename), "w", encoding="utf-8") as f:
                json.dump({"speaker": speaker, "lang": lang, "text": text}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[SPK][DEBUG_AUDIO] failed to save turn label: {e!r}")

    threading.Thread(target=_write, daemon=True).start()


def warmup(run_inference: bool = True) -> None:
    _clear_debug_audio_dir()
    t0 = time.perf_counter()
    _load_model()
    load_s = time.perf_counter() - t0
    logger.info(
        f"[SPK] ovoz embedding modeli yuklandi: backend={_embedding_backend!r} "
        f"model={_embedding_model_name!r} device={SPK_DEVICE!r} ({load_s:.2f}s)"
    )
    if not run_inference:
        return
    try:
        dummy_wav = (np.random.randn(16000 * 2).astype(np.float32) * 0.05)
        t1 = time.perf_counter()
        _embed(dummy_wav, 16000)
        embed_ms = (time.perf_counter() - t1) * 1000
        logger.info(f"[SPK] bitta embedding (2.0s audio, device={SPK_DEVICE!r}): {embed_ms:.1f}ms")

        session = create_session()
        identify_speaker(session, dummy_wav, 16000, seq=0)
        logger.info("[SPK] warm-up inference done")
    except Exception as e:
        logger.info(f"[SPK] warm-up inference skipped: {e!r}")


def create_session():
    return {
        # speaker_id(int) -> {"prototypes": [{"seq": int, "embedding": np.ndarray}, ...]}
        # — har bir prototip AYNAN qaysi turn (seq) orqali qo'shilganini
        # eslab turadi (correct_speaker/delete_turn shu orqali, aynan
        # O'ZI qo'shgan namunani topib olib tashlashi uchun). Birinchi
        # (index 0) prototip "founding" namuna — SPK_MAX_PROTOTYPES'dan
        # oshganda avtomatik FIFO orqali olib tashlanmaydi (qarang
        # _add_prototype), lekin aniq seq(lar) bo'yicha (_pop_prototypes_by_seqs)
        # BOSHQA har qanday prototip kabi olib tashlanishi mumkin.
        "speakers": {},
        "next_id": 1,
        # SPK_SHORT_TURN_S'dan qisqa, noaniq turnlar uchun fallback —
        # eng SO'NGGI qaytarilgan (uncertain ham) spiker.
        "last_speaker": None,
        # Mentor demo'da topilgan 1/2-muammolar: "NOMA'LUM" (hech kimga mos
        # kelmaydigan, lekin yangi spiker ochish uchun ham qisqa) bo'laklar
        # — {"seq", "embedding", "speech_s", "received_at"} — qarang
        # SPK_UNKNOWN_BUFFER_WINDOW_S izohi yuqorida.
        "unknown_buffer": [],
        "lock": threading.Lock(),
    }


def _embed(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    model = _load_model()
    waveform = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).reshape(1, -1)
    # Latency spike tuzatishi (qarang app/gpu_lock.py izohi): bir nechta
    # WebSocket sessiya BIR VAQTDA GPU'ga murojaat qilsa, lock ularni
    # ketma-ketlashtiradi (thrashing o'rniga oddiy kutish).
    with GPU_LOCK:
        if _embedding_backend == "pyannote":
            emb = model({"waveform": waveform, "sample_rate": sample_rate})
            emb = np.asarray(emb, dtype=np.float64).reshape(-1)
        else:  # "speechbrain"
            with torch.no_grad():
                emb = model.encode_batch(waveform)
            emb = emb.squeeze().cpu().numpy().astype(np.float64).reshape(-1)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Ikkisi ham _embed() orqali allaqachon birlik-normallashtirilgan —
    # shuning uchun oddiy dot product cosine o'xshashlikka teng.
    return float(np.dot(a, b))


def _speaker_best_sim(profile: dict, embedding: np.ndarray) -> float:
    """Spikerning BARCHA prototiplari bilan o'xshashliklarning ENG
    KATTASI — "eng yaqin prototip" (nearest-neighbor). Prototip hali
    yo'q bo'lsa (nazariy jihatdan mumkin — masalan embedding'siz
    yaratilgan "zaxira" spiker) — hech qachon moslashmaydigan -1.0."""
    if not profile["prototypes"]:
        return -1.0
    return max(_cosine_sim(embedding, p["embedding"]) for p in profile["prototypes"])


def _add_prototype(profile: dict, embedding: np.ndarray, seq: int) -> None:
    profile["prototypes"].append({"seq": seq, "embedding": embedding})
    if len(profile["prototypes"]) > SPK_MAX_PROTOTYPES:
        # index 0 — "founding" prototip, FIFO orqali HECH QACHON olib
        # tashlanmaydi; eng ESKI (lekin founding EMAS) prototip index
        # 1'da turadi.
        del profile["prototypes"][1]


def _pop_prototypes_by_seqs(profile: dict, seqs) -> list[dict]:
    """`seqs` (bitta bubble'ga tegishli BARCHA seq'lar — qarang dialog.py'dagi
    `merged_seqs`, RETROAKTIV BIRLASHTIRISH orqali bitta bubble bir nechta
    audio bo'lagidan — har biri O'ZINING prototipini qo'shgan bo'lishi
    mumkin — tashkil topgan bo'lishi mumkin) bilan TEGLANGAN prototiplarni
    ro'yxatdan olib tashlaydi va ularni ({"seq", "embedding"} lug'atlari
    ro'yxati sifatida, ASL seq tegi bilan) qaytaradi — correct_speaker
    shu tegni SAQLAB qolib, target spikerga qo'shadi.

    REAL TESTDA TOPILGAN XATO: ilgari FAQAT bitta (asosiy) seq bo'yicha
    qidirilardi — agar davom bo'lagi (masalan seq=10) ALLAQACHON o'z
    prototipini (seq=10 tegi bilan) qo'shgan bo'lsa, u ESKI spikerda
    saqlanib qolardi (tuzatish FAQAT seq=9 tegli prototipni ko'chirardi).

    Bu yerda "founding" himoyasi YO'Q (FIFO'dan farqli) — foydalanuvchi
    ANIQ shu bubble'ni o'chirish/tuzatishni so'ragan, demak bu DELIBERATE
    harakat."""
    seqs_set = set(seqs)
    popped = [proto for proto in profile["prototypes"] if proto["seq"] in seqs_set]
    profile["prototypes"] = [proto for proto in profile["prototypes"] if proto["seq"] not in seqs_set]
    return popped


def _max_cross_sim(profile_a: dict, profile_b: dict) -> float:
    best = -1.0
    for pa in profile_a["prototypes"]:
        for pb in profile_b["prototypes"]:
            sim = _cosine_sim(pa["embedding"], pb["embedding"])
            if sim > best:
                best = sim
    return best


def _merge_profile_into(target_profile: dict, source_profile: dict) -> None:
    target_profile["prototypes"].extend(source_profile["prototypes"])
    while len(target_profile["prototypes"]) > SPK_MAX_PROTOTYPES:
        del target_profile["prototypes"][1]


def _check_and_merge(session) -> list[tuple[int, int, float]]:
    """`session["lock"]` ALLAQACHON ushlab turilgan paytda chaqirilishi
    SHART (identify_speaker/apply_speaker_correction/remove_turn_prototype
    ICHIDAN). Ixtiyoriy ikki spikerning prototiplari orasidagi ENG KATTA
    o'xshashlik SPK_MERGE_SIM'dan yuqori bo'lsa, ikkisini KICHIK raqamli
    spikerga birlashtiradi (prototiplar qo'shiladi, SPK_MAX_PROTOTYPES
    qoidasi bilan) — bir nechta birlashish ZANJIRI (masalan 3->2, keyin
    2->1) bo'lsa, HAMMASI bitta chaqiruvda hal qilinadi (`while changed`).

    Qaytaradi: [(from_id, into_id, sim), ...] — AMALGA OSHGAN birlashishlar
    ro'yxati, SODIR BO'LGAN tartibda (chaqiruvchi shu tartibda ketma-ket
    qo'llab, zanjirlarni to'g'ri "kuzatib borishi" mumkin — qarang
    dialog.py'dagi _apply_merges_and_notify)."""
    events: list[tuple[int, int, float]] = []
    changed = True
    while changed:
        changed = False
        ids = sorted(session["speakers"].keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                sim = _max_cross_sim(session["speakers"][a], session["speakers"][b])
                if sim >= SPK_MERGE_SIM:
                    _merge_profile_into(session["speakers"][a], session["speakers"][b])
                    del session["speakers"][b]
                    events.append((b, a, sim))
                    logger.info(f"[SPK][MERGE] from={b} into={a} sim={sim:.3f}")
                    changed = True
                    break
            if changed:
                break
    return events


def _apply_remap(speaker_id, merges: list[tuple[int, int, float]]):
    """Berilgan spiker ID'ni, `merges` ro'yxatidagi (ketma-ket, zanjir
    bo'lishi mumkin) birlashishlardan KEYINGI YAKUNIY ID'ga aylantiradi."""
    for frm, into, _ in merges:
        if speaker_id == frm:
            speaker_id = into
    return speaker_id


def _trim_unknown_buffer(session) -> None:
    now = time.monotonic()
    session["unknown_buffer"] = [
        e for e in session["unknown_buffer"] if now - e["received_at"] <= SPK_UNKNOWN_BUFFER_WINDOW_S
    ]


def _best_existing_match(session, embedding: np.ndarray):
    best_id, best_sim = None, -1.0
    for sid, profile in session["speakers"].items():
        sim = _speaker_best_sim(profile, embedding)
        if sim > best_sim:
            best_id, best_sim = sid, sim
    return best_id, best_sim


def _best_match_with_margin(session, embedding: np.ndarray):
    """`_best_existing_match`'ning margin-ogoh (qarang SPK_MIN_MARGIN izohi
    yuqorida) versiyasi — eng yaqin IKKI nomzod orasidagi farqni ham
    qaytaradi. Qaytaradi: (best_id, best_sim, margin) — bitta (yoki hech
    qanday) spiker bo'lsa `margin=None` (chegara ma'nosiz, hech qachon
    "tie" deb hisoblanmaydi)."""
    sims: dict = {}
    for sid, profile in session["speakers"].items():
        sims[sid] = _speaker_best_sim(profile, embedding)
    if not sims:
        return None, -1.0, None
    best_id = max(sims, key=sims.get)
    best_sim = sims[best_id]
    others = [sim for sid, sim in sims.items() if sid != best_id]
    if not others:
        return best_id, best_sim, None
    return best_id, best_sim, best_sim - max(others)


def embed_for_match(wav: np.ndarray, sample_rate: int) -> tuple:
    """Trim + embed ONCE — shared by identify_speaker() (normal path, via
    `precomputed=`) and the dialog.py language-gating pre-check
    (quick_speaker_match) below, so one subsegment's audio is embedded at
    most once even when its voice identity is needed before BOTH the
    language decision (Plan B: speaker-conditioned threshold) and the
    actual enrollment/clustering decision.

    Qaytaradi: (embedding|None, trimmed, speech_s) — xuddi identify_speaker()
    ICHIDAGI eski (endi shu yerga ko'chirilgan) trim/embed bosqichi kabi."""
    trimmed = _trim_trailing_silence(wav, sample_rate)
    turn_s = len(wav) / sample_rate
    speech_s = len(trimmed) / sample_rate
    if trimmed.size > 0 and speech_s < MIN_TRIMMED_DURATION_S and speech_s < turn_s:
        logger.info(f"[SPK][TRIM] trim too aggressive ({turn_s:.2f}s -> {speech_s:.2f}s), using original audio")
        trimmed = wav
        speech_s = turn_s
    embedding = None
    if trimmed.size > 0 and speech_s >= MIN_TRIMMED_DURATION_S:
        embedding = _embed(trimmed, sample_rate)
    return embedding, trimmed, speech_s


def quick_speaker_match(session, embedding: np.ndarray | None):
    """Read-only best-match lookup against ALREADY ENROLLED speakers —
    does NOT mutate session state (no enrollment, no new-speaker/unknown-
    buffer side effects). Used BEFORE the language decision (Plan B, qarang
    dialog.py) so a short/mixed subsegment's language threshold can be
    relaxed when the voice clearly already belongs to a known speaker.
    `embedding=None` (audio too short/silent) — hech kimga mos kelmaydi.

    Qaytaradi: (best_id, best_sim, margin) — qarang SPK_MIN_MARGIN izohi:
    chaqiruvchi (dialog.py) `margin`ni HAM tekshirishi SHART (bitta
    yaqinroq spiker tanlashning O'ZI yetarli EMAS — ikkinchi nomzod bilan
    TENG bo'lsa, bu "aniq tanilgan" hisoblanmasligi kerak)."""
    if embedding is None:
        return None, -1.0, None
    with session["lock"]:
        return _best_match_with_margin(session, embedding)


def _process_unknown_buffer(session) -> list[tuple[int, int]]:
    """`session["lock"]` ALLAQACHON ushlab turilgan paytda chaqirilishi
    SHART. Qarang SPK_UNKNOWN_BUFFER_WINDOW_S izohi (1/2-muammolar) —
    buferdagi "noma'lum" bo'laklarni (a) mavjud spikerlarga mos kelsa
    ko'chiradi, (b) o'zaro o'xshash klasterlarni YANGI spikerga aylantiradi
    (kamida 2 a'zo, jami nutq >= SPK_MIN_ENROLL_SPEECH_S).

    Qaytaradi: [(seq, new_speaker_id), ...] — shu chaqiruv davomida
    QAYTA TAYINLANGAN (frontend'da allaqachon "unknown" sifatida
    ko'rsatilgan bo'lishi mumkin bo'lgan) bo'laklar ro'yxati."""
    _trim_unknown_buffer(session)
    buf = session["unknown_buffer"]
    reassigned: list[tuple[int, int]] = []

    # (a) Mavjud spikerlarga mos kelganlarni ko'chirish — masalan shu
    # orada (boshqa, bufer bilan bog'liq bo'lmagan turn orqali) profil
    # kattalashgan bo'lishi mumkin. 7-band: "iflos" (purity past) yozuv
    # ham spikerga TAYINLANADI, lekin profilga YANGI prototip sifatida
    # QO'SHILMAYDI (ifloslantirish xavfi).
    still_unknown = []
    for entry in buf:
        sid, sim = _best_existing_match(session, entry["embedding"])
        if sid is not None and sim >= SPK_ACCEPT:
            purity = entry.get("purity", 1.0)
            if purity >= SPK_ENROLL_MIN_PURITY:
                _add_prototype(session["speakers"][sid], entry["embedding"], entry["seq"])
            else:
                logger.info(
                    f"[SPK][ENROLL] skipped reason=impure purity={purity:.3f} "
                    f"seq={entry['seq']} (buffer_assign -> speaker={sid})"
                )
            reassigned.append((entry["seq"], sid))
            logger.info(
                f"[SPK] decision=buffer_assign seq={entry['seq']} -> speaker={sid} sim={sim:.3f}"
            )
        else:
            still_unknown.append(entry)
    buf[:] = still_unknown

    # (b) O'zaro o'xshash klaster(lar)ni yangi spikerga aylantirish —
    # zanjir bo'lishi mumkin (bitta promote keyingisini ochib berishi
    # mumkin emas, lekin xavfsizlik uchun `while changed` saqlanadi).
    changed = True
    while changed:
        changed = False
        n = len(buf)
        for i in range(n):
            cluster = [buf[i]]
            for j in range(n):
                if j == i:
                    continue
                sim = _cosine_sim(buf[i]["embedding"], buf[j]["embedding"])
                if sim >= SPK_ACCEPT:
                    cluster.append(buf[j])
            total_speech = sum(e["speech_s"] for e in cluster)
            if len(cluster) >= 2 and total_speech >= SPK_MIN_ENROLL_SPEECH_S:
                # 7-band: FAQAT "toza" (purity >= SPK_ENROLL_MIN_PURITY)
                # a'zolar prototip sifatida qo'shiladi — item2: agar
                # HECH QAYSI a'zo toza bo'lmasa, prototipsiz spiker
                # OCHILMAYDI (bu urinish bekor qilinadi, a'zolar bufer'da
                # QOLADI — keyinroq ko'proq/toza namuna to'planishi mumkin).
                # 4-muammo: endi FAQAT purity emas, near_field_ratio/
                # pair_mass ham tekshiriladi — qarang _new_speaker_evidence.
                strong = []
                for entry in cluster:
                    ok, reasons = _new_speaker_evidence(
                        entry.get("purity", 1.0), entry.get("near_field_ratio"), entry.get("pair_mass")
                    )
                    if ok:
                        strong.append(entry)
                    else:
                        logger.info(f"[SPK] decision=unknown reason=weak_evidence seq={entry['seq']} ({'; '.join(reasons)})")
                # Subsegments of one browser chunk are not independent confirmations.
                independent_turns = {int(e["seq"]) for e in strong}
                if len(independent_turns) < SPK_NEW_CONFIRM_TURNS or sum(e["speech_s"] for e in strong) < SPK_MIN_ENROLL_SPEECH_S:
                    continue
                cluster = strong
                strong_seqs = {e["seq"] for e in strong}
                total_speech = sum(e["speech_s"] for e in strong)
                new_id = session["next_id"]
                session["next_id"] += 1
                session["speakers"][new_id] = {"prototypes": []}
                cluster_seqs = []
                for e in cluster:
                    if e["seq"] in strong_seqs:
                        _add_prototype(session["speakers"][new_id], e["embedding"], e["seq"])
                    reassigned.append((e["seq"], new_id))
                    cluster_seqs.append(e["seq"])
                seqs_set = set(cluster_seqs)
                buf[:] = [e for e in buf if e["seq"] not in seqs_set]
                logger.info(
                    f"[SPK] decision=buffer_promote seqs={sorted(cluster_seqs)} -> "
                    f"speaker={new_id} total_speech={total_speech:.2f}s "
                    f"strong_members={len(strong_seqs)}/{len(cluster)}"
                )
                changed = True
                break
    return reassigned


def identify_speaker(
    session,
    wav: np.ndarray,
    sample_rate: int,
    seq: int,
    detected_lang: str | None = None,
    debug_label_out: dict | None = None,
    purity: float = 1.0,
    near_field_ratio: float | None = None,
    pair_mass: float | None = None,
    precomputed: tuple | None = None,
) -> dict:
    """Cheklanmagan sonli spiker uchun ONLAYN klasterlash — FAQAT ovoz
    asosida (til BUTUNLAY chiqarib tashlandi, `detected_lang` bu yerda
    FAQAT [SPK] logiga chiqarish uchun — qarorga HECH QANDAY ta'sir
    qilmaydi, tarjima yo'nalishi dialog.py'da mustaqil hal qilinadi).

    Qaytaradi: {"speaker": int|None, "uncertain": bool, "decision": str,
    "merges": [(from_id, into_id, sim), ...], "reassigned": [(seq, speaker_id), ...]}.

        "merges" — enrollment natijasida ixtiyoriy ikki spiker BIR ODAM
        ekani aniqlanib AVTOMATIK birlashtirilgan bo'lsa (qarang
        _check_and_merge), shu voqealar ro'yxati — dialog.py buni frontendga
        "speakers_merged" sifatida uzatishi va o'zining turn_history/
        last_turn_by_speaker holatini mos ravishda yangilashi SHART.

        "reassigned" — Mentor demo'da topilgan 1-muammo: "unknown" buferi
        (pastga, 3d'ga qarang) orqali, JORIY `seq`dan TASHQARI, BOSHQA
        (oldin "unknown" sifatida allaqachon frontendga yuborilgan)
        bo'laklar ham shu chaqiruv ichida qayta tayinlangan bo'lishi
        mumkin — dialog.py shularni ko'rib, mos bubble'larni yangi
        spiker bilan qayta yuborishi SHART (xuddi "merges" kabi).

    Qaror daraxti (batafsil — [SPK] logiga qarang, chegaralarni real
    suhbat loglaridan sozlash uchun):
      1. sims = {speaker_id: eng yaqin prototip bilan o'xshashlik}.
      2. best = eng yuqori sims.
      3a. best >= SPK_ACCEPT → shu spiker. Agar NUTQ (trim qilingan)
          >= SPK_MIN_ENROLL_SPEECH_S VA best < SPK_DUPLICATE_PROTOTYPE_SIM
          (ya'ni bu aynan takroriy namuna EMAS) VA (profilda hali
          SPK_BOOTSTRAP_PROTOTYPES'dan KAM prototip BOR YOKI best >=
          SPK_ENROLL_MIN_SIM) — YANGI prototip qo'shiladi.
      3b. SPK_NEW <= best < SPK_ACCEPT (ANIQ boshqa odam EMAS, lekin
          SPK_ACCEPT'ga ham yetmaydi) → eng yaqin spikerga, uncertain=True,
          prototip QO'SHILMAYDI. TURN (xom) SPK_SHORT_TURN_S'dan qisqa VA
          avval kimdir gapirgan bo'lsa — ovoz-o'xshashlik o'rniga ENG
          SO'NGGI gapirgan spiker (last_speaker) ishlatiladi ("suhbat
          davomiyligi" qisqa javoblar uchun ko'proq ishonchli signal).
      3c. best < SPK_NEW: birinchi profil uchun ishonchli nutq kamida
          SPK_BOOTSTRAP_MIN_SPEECH_S; keyingi yangi profil uchun nutq
          SPK_MIN_ENROLL_SPEECH_S va kuchli til/near-field dalili kerak.
          Qolgan nomzodlar unknown buferida mustaqil tasdiq kutadi.
      3d. best < SPK_NEW VA nutq QISQA — Mentor demo'da topilgan 1/2-
          muammolar: bu holat ENDI mavjud spikerga HAM (eski "short"/
          "uncertain" fallback), prototipsiz yangi spikerga HAM (eski
          "new_short") tayinlanmaydi — speaker=None, decision="unknown"
          ("Aniqlanmoqda…") qaytariladi va embedding "unknown" buferiga
          qo'shiladi (qarang _process_unknown_buffer). Agar shu CHAQIRUV
          ichida bufer orqali (mavjud spikerga mos kelib YOKI boshqa
          bufer a'zolari bilan yangi spikerga aylanib) darhol hal
          bo'lsa — decision="buffer_assigned", speaker=o'sha ID,
          uncertain=False.

    `precomputed` — Plan B (speaker-conditioned language threshold):
    dialog.py ba'zan shu AYNAN audio uchun embeddingni ALLAQACHON
    (til qarori OLDIDAN, quick_speaker_match() orqali) hisoblagan bo'ladi
    — shu holda `(embedding, trimmed, speech_s)` shu yerga uzatiladi va
    pastdagi trim/embed bosqichi QAYTA hisoblashdan saqlanish uchun
    o'tkazib yuboriladi (qarang embed_for_match())."""
    t0 = time.perf_counter()
    if precomputed is not None:
        embedding, trimmed, speech_s = precomputed
    else:
        embedding, trimmed, speech_s = embed_for_match(wav, sample_rate)
    trim_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[SPK][TRIM] wav_len={len(wav)} trimmed_len={len(trimmed)} took={trim_ms:.2f}ms")

    turn_s = len(wav) / sample_rate
    trimmed_rms = float(np.sqrt(np.mean(trimmed.astype(np.float64) ** 2))) if trimmed.size else 0.0

    try:
        with session["lock"]:
            speakers = session["speakers"]

            sims: dict = {}
            if embedding is not None:
                for sid, profile in speakers.items():
                    sims[sid] = _speaker_best_sim(profile, embedding)

            if sims:
                best_id = max(sims, key=sims.get)
                best_sim = sims[best_id]
                _others = [s for sid, s in sims.items() if sid != best_id]
                margin = (best_sim - max(_others)) if _others else None
            else:
                best_id = None
                best_sim = -1.0
                margin = None

            decision: str
            result_speaker = None
            uncertain = False
            # Qarang _process_unknown_buffer() — FAQAT "unknown" tarmog'ida
            # to'ldiriladi, qolgan HAMMA holatda bo'sh (quyida, qaytarish
            # uchun, joriy `seq`dan TASHQARI barcha yozuvlar ishlatiladi).
            buffer_reassigned: list[tuple[int, int]] = []

            if embedding is None:
                # Audio juda qisqa/jim — hech qanday ovoz dalili yo'q.
                # Amalda deyarli bo'lmaydi (dialog.py'dagi MIN_TURN_SECONDS/
                # STT MIN_SPEECH_S allaqachon bundan oldin filtrlaydi), lekin
                # oxirgi spikerga (bo'lsa) noaniq tayinlaymiz; aks holda
                # ovoz dalilisiz profil ochmasdan unknown qaytaramiz.
                decision = "no_embedding"
                uncertain = True
                if session["last_speaker"] is not None:
                    result_speaker = session["last_speaker"]
                else:
                    decision = "unknown"
                    logger.info(f"[SPK] decision=unknown reason=weak_evidence seq={seq} (no_embedding)")

            elif best_sim >= SPK_ACCEPT and (margin is None or margin >= SPK_MIN_MARGIN):
                decision = "accept"
                result_speaker = best_id
                profile = speakers[best_id]
                if speech_s >= SPK_MIN_ENROLL_SPEECH_S and best_sim < SPK_DUPLICATE_PROTOTYPE_SIM:
                    is_bootstrap = len(profile["prototypes"]) < SPK_BOOTSTRAP_PROTOTYPES
                    if purity < SPK_ENROLL_MIN_PURITY:
                        # 7-band: qism "iflos" (segmenter.py — ehtimol ikki
                        # ovoz aralashgan) — spikerga TAYINLASH davom etadi,
                        # lekin profil YANGI prototip bilan ifloslantirilmaydi.
                        logger.info(
                            f"[SPK][ENROLL] skipped reason=impure purity={purity:.3f} < "
                            f"{SPK_ENROLL_MIN_PURITY} speaker={best_id}"
                        )
                    elif is_bootstrap or best_sim >= SPK_ENROLL_MIN_SIM:
                        _add_prototype(profile, embedding, seq)
                        logger.info(
                            f"[SPK][ENROLL] speaker={best_id} yangi prototip qo'shildi "
                            f"(sim={best_sim:.3f}{' bootstrap' if is_bootstrap else ''}) — "
                            f"jami prototiplar={len(profile['prototypes'])}"
                        )
                    else:
                        # Turn SPK_ACCEPT'ni o'tdi (spikerga TAYINLANDI), lekin
                        # SPK_ENROLL_MIN_SIM'dan past — profilga XATO (noto'g'ri
                        # qabul qilingan bo'lishi mumkin) namuna qo'shib,
                        # ifloslantirish xavfidan saqlanish uchun prototip
                        # QO'SHILMAYDI.
                        logger.info(
                            f"[SPK][ENROLL] skipped sim={best_sim:.3f} < {SPK_ENROLL_MIN_SIM} "
                            f"speaker={best_id}"
                        )

            elif best_sim >= SPK_ACCEPT:
                # REAL TESTDA TOPILGAN MUAMMO: best_sim SPK_ACCEPT'dan
                # yuqori, LEKIN ikkinchi eng yaqin spiker bilan farqi
                # (margin) SPK_MIN_MARGIN'dan kam — qarang izoh yuqorida.
                # Eng yaqinga TAYINLANADI (bubble ko'rsatish uchun BIROR
                # ID kerak), lekin uncertain=True va PROFIL
                # YANGILANMAYDI (noto'g'ri spikerga qo'shilib ifloslanish
                # xavfi) — bu natija dialog.py'dagi Plan B (accept_known_
                # speaker) relaxation uchun HAM ishlatilmaydi (chaqiruvchi
                # quick_speaker_match orqali, MUSTAQIL, o'zining margin
                # tekshiruvini bajaradi).
                decision = "tie_uncertain"
                result_speaker = best_id
                uncertain = True
                logger.info(
                    f"[SPK] decision=tie_uncertain seq={seq} best={best_id}({best_sim:.3f}) "
                    f"margin={margin:.3f} < {SPK_MIN_MARGIN}"
                )

            elif (
                best_id is not None
                and best_sim >= SPK_BOOTSTRAP_ACCEPT
                and len(speakers[best_id]["prototypes"]) < SPK_BOOTSTRAP_PROTOTYPES
                and (margin is None or margin >= SPK_MIN_MARGIN)
                and speech_s >= SPK_MIN_ENROLL_SPEECH_S
                and purity >= SPK_ENROLL_MIN_PURITY
            ):
                # Qarang SPK_BOOTSTRAP_ACCEPT izohi.
                decision = "accept"
                result_speaker = best_id
                profile = speakers[best_id]
                _add_prototype(profile, embedding, seq)
                logger.info(
                    f"[SPK][ENROLL] speaker={best_id} yangi prototip qo'shildi "
                    f"(sim={best_sim:.3f} bootstrap_low) — jami prototiplar={len(profile['prototypes'])}"
                )

            elif best_id is not None and SPK_NEW <= best_sim < SPK_ACCEPT:
                # Mentor demo'da topilgan 1-muammo: eski "qisqa turn ->
                # SO'ZSIZ last_speaker" qoidasi ENDI shu (haqiqatan "biroz
                # o'xshash") band ICHIDA, QO'SHIMCHA shart sifatida —
                # best_sim BU YERDA ALLAQACHON SPK_NEW'dan yuqori, ya'ni
                # bu ANIQ boshqa odam EMAS, shuning uchun qisqa javob
                # uchun "suhbat davomiyligi" (last_speaker) ovoz-
                # o'xshashlikdan (best_id) ustun qo'yish xavfsiz.
                if turn_s < SPK_SHORT_TURN_S and session["last_speaker"] is not None:
                    decision = "short"
                    result_speaker = session["last_speaker"]
                else:
                    decision = "uncertain"
                    result_speaker = best_id
                uncertain = True

            elif (
                best_id is not None
                and best_id == session["last_speaker"]
                and best_sim >= SPK_SHORT_CONTINUE_SIM
                and (margin is None or margin >= SPK_MIN_MARGIN)
                and speech_s < SPK_SHORT_CONTINUE_MAX_SPEECH_S
            ):
                # Qarang SPK_SHORT_CONTINUE_SIM izohi.
                decision = "short_continue"
                result_speaker = best_id
                uncertain = True

            elif _new_speaker_evidence(purity, near_field_ratio, pair_mass)[0] and (
                (not speakers and speech_s >= SPK_BOOTSTRAP_MIN_SPEECH_S)
                or (
                    bool(speakers) and speech_s >= SPK_MIN_ENROLL_SPEECH_S
                    and pair_mass is not None and pair_mass >= SPK_NEW_IMMEDIATE_PAIR_MASS
                    and (near_field_ratio is None or near_field_ratio >= SPK_NEW_IMMEDIATE_NEAR_RATIO)
                )
            ):
                # Strong single-turn evidence gets an ID immediately. Weaker
                # candidates still need independent confirmation in the buffer.
                new_id = session["next_id"]
                session["next_id"] += 1
                speakers[new_id] = {"prototypes": [{"seq": seq, "embedding": embedding}]}
                decision = "new"
                result_speaker = new_id

            else:
                # Mentor demo'da topilgan 1/2/4/7-muammolar: best < SPK_NEW
                # (ANIQ boshqa odam, yoki hali spiker yo'q) VA (nutq QISQA
                # YOKI purity/near-field/pair_mass dalili yetarli emas —
                # qarang _new_speaker_evidence). ENDI bu holat na mavjud
                # spikerga (eski "uncertain"/"short"), na prototipsiz/
                # kuchsiz-dalilli yangi spikerga ("new_short") tayinlanmaydi
                # — "unknown" buferiga qo'yiladi (qarang _process_unknown_
                # buffer va SPK_UNKNOWN_BUFFER_WINDOW_S izohi yuqorida).
                # Frontend buni speaker=None orqali ko'rib, kulrang
                # "Aniqlanmoqda…" yorlig'ini ko'rsatadi (DialogBubble.jsx).
                decision = "unknown"
                result_speaker = None
                uncertain = True
                if speech_s >= SPK_MIN_ENROLL_SPEECH_S:
                    # Shu branch'ga FAQAT "new_evidence_ok=False" orqali
                    # tushgan bo'lsa (ya'ni nutq yetarli uzun edi, lekin
                    # 4-band dalili yetarli emas) — aniq sababini log qilamiz.
                    _, weak_reasons = _new_speaker_evidence(purity, near_field_ratio, pair_mass)
                    if weak_reasons:
                        logger.info(
                            f"[SPK] decision=unknown reason=weak_evidence seq={seq} ({'; '.join(weak_reasons)})"
                        )
                    else:
                        logger.info(f"[SPK] decision=unknown reason=pending_confirmation seq={seq} required_turns={SPK_NEW_CONFIRM_TURNS}")
                session["unknown_buffer"].append(
                    {
                        "seq": seq,
                        "embedding": embedding,
                        "speech_s": speech_s,
                        "purity": purity,
                        "near_field_ratio": near_field_ratio,
                        "pair_mass": pair_mass,
                        "received_at": time.monotonic(),
                    }
                )
                buffer_reassigned = _process_unknown_buffer(session)
                for r_seq, r_speaker in buffer_reassigned:
                    if r_seq == seq:
                        # Aynan shu (hozirgi) bo'lak SHU chaqiruv ichida
                        # darhol qayta tayinlandi (mavjud spikerga mos
                        # keldi YOKI boshqa bufer a'zolari bilan birga
                        # yangi spikerga aylandi) — "unknown" o'rniga
                        # TO'G'RIDAN-TO'G'RI o'sha natijani qaytaramiz.
                        decision = "buffer_assigned"
                        result_speaker = r_speaker
                        uncertain = False

            # A new or improved profile can resolve earlier unknown turns,
            # even when the current turn did not enter the unknown branch.
            if decision != "unknown" and session["unknown_buffer"]:
                buffer_reassigned.extend(_process_unknown_buffer(session))

            # Har bir enrollment/yangi-spikerdan KEYIN — ixtiyoriy ikki
            # spiker aslida bitta odam bo'lib qolmaganini tekshiramiz
            # (qarang SPK_MERGE_SIM izohi yuqorida).
            merges = _check_and_merge(session)
            result_speaker = _apply_remap(result_speaker, merges)
            # `buffer_reassigned` — joriy `seq` BUNDAN TASHQARI (o'zi
            # `result_speaker` orqali alohida qaytariladi) bufer orqali
            # qayta tayinlangan BOSHQA bo'laklar — dialog.py shularni
            # ko'rib, allaqachon frontendga yuborilgan bubble(lar)ni YANGI
            # spiker bilan qayta yuboradi (qarang identify_speaker
            # docstring'idagi "reassigned").
            reassigned = [
                (r_seq, _apply_remap(r_speaker, merges))
                for r_seq, r_speaker in buffer_reassigned
                if r_seq != seq
            ]

            if result_speaker is not None:
                session["last_speaker"] = result_speaker

            sims_str = ", ".join(f"{sid}:{sim:.2f}" for sid, sim in sims.items())
            best_str = f"{best_id}({best_sim:.2f})" if best_id is not None else "none"
            proto_str = ", ".join(f"{sid}:{len(p['prototypes'])}" for sid, p in speakers.items())
            logger.info(
                f"[SPK] seq={seq} speech={speech_s:.2f}s turn={turn_s:.2f}s lang={detected_lang!r} "
                f"sims={{{sims_str}}} best={best_str} decision={decision} speaker={result_speaker} "
                f"uncertain={uncertain} prototypes={{{proto_str}}} "
                f"thresholds=(accept={SPK_ACCEPT},new={SPK_NEW})"
            )

            wav_filename = _save_debug_audio_async(
                trimmed, sample_rate, speech_s, trimmed_rms, f"speaker_{result_speaker}_{decision}"
            )
            if debug_label_out is not None:
                debug_label_out["wav_filename"] = wav_filename

            return {
                "speaker": result_speaker,
                "uncertain": uncertain,
                "decision": decision,
                "merges": merges,
                # Mentor demo'da topilgan 1-muammo: "unknown" buferi orqali
                # BOSHQA (joriy `seq`dan tashqari) bo'laklar ham qayta
                # tayinlangan bo'lishi mumkin — dialog.py shularni ko'rib,
                # allaqachon frontendga yuborilgan bubble(lar)ni yangi
                # spiker bilan qayta yuboradi.
                "reassigned": reassigned,
            }
    except Exception as e:
        logger.error(f"[SPK][ERROR] identify_speaker raised {e!r}")
        raise


def has_known_speakers(session) -> bool:
    """dialog.py'dagi "fast path" uchun — kamida BITTA spiker uchun ovoz
    profili allaqachon mavjud bo'lsa, juda qisqa/juda past-energiyali
    (frontend VAD'ning past chegarasi tufayli shovqindan ham "nutq" deb
    yuborilgan bo'lishi mumkin bo'lgan) audio bo'laklarini STT/embedding'ga
    yubormasdan oldin saralab tashlash mumkin."""
    with session["lock"]:
        return len(session["speakers"]) > 0


def apply_speaker_correction(
    session, wav: np.ndarray, sample_rate: int, seqs: list[int], old_speaker, target_speaker
) -> dict:
    """Har bir bubble'dagi "spikerni almashtirish" menyusi uchun —
    foydalanuvchi mavjud spikerlardan birini YOKI "Yangi spiker"ni
    tanlaganda chaqiriladi (dialog.py, `correct_speaker` action).
    `target_speaker` — mavjud spiker ID (int) yoki literal `"new"` satri.
    (`target_speaker == old_speaker` holati dialog.py tarafida ALLAQACHON
    no-op sifatida ushlanadi — bu funksiyaga umuman yetib kelmaydi.)

    `seqs` — shu BUBBLE'ga tegishli BARCHA seq'lar (dialog.py'dagi
    `merged_seqs`: RETROAKTIV BIRLASHTIRISH orqali bir nechta audio
    bo'lagi, HAR BIRI o'zining prototipini qo'shgan bo'lishi mumkin, bitta
    bubble'ga birlashtirilgan bo'lishi mumkin). ESKI spikerga (`old_speaker`)
    shu seq'larning BIRI orqali qo'shilgan BARCHA prototiplar topib olib
    tashlanadi (REAL TESTDA TOPILGAN XATO: ilgari FAQAT asosiy seq
    tekshirilardi — davom bo'lagi tegli prototip eski spikerda saqlanib
    qolib, "Speaker 3 o'chmadi va o'sha odamning ovozini saqlab qoldi"
    degan holatga olib kelardi). Keyin O'SHA (ASL seq teglari bilan, yoki,
    agar hech qaysi seq prototip qo'shmagan bo'lsa, yangidan hisoblangan va
    birinchi seq bilan teglangan) embedding(lar) TARGET spikerga qo'shiladi.

    Qaytaradi: {"speaker": int, "removed_speaker": int|None,
    "merges": [...]}. `removed_speaker` — ESKI spikerda ENDI birorta ham
    prototip qolmasa (shu bubble uning YAGONA namunasi bo'lgan ekan),
    shu spikerning o'zi butunlay o'chiriladi va ID'si shu yerda qaytadi —
    dialog.py buni frontendga "speaker_removed" sifatida uzatishi kerak."""
    trimmed = _trim_trailing_silence(wav, sample_rate)
    duration_s = len(trimmed) / sample_rate
    original_duration_s = len(wav) / sample_rate
    if trimmed.size > 0 and duration_s < MIN_TRIMMED_DURATION_S and duration_s < original_duration_s:
        trimmed = wav
        duration_s = original_duration_s

    primary_seq = seqs[0]

    with session["lock"]:
        removed_speaker = None

        # Mentor demo'da topilgan 1-muammo: agar shu bo'lak(lar) hali
        # "unknown" buferida tursa (foydalanuvchi ularni avtomatik
        # klasterlash ulgurmasdan QO'LDA tuzatgan bo'lsa) — endi quyida
        # TO'G'RIDAN-TO'G'RI tayinlanadi, shuning uchun bufer yozuvi
        # endi ORTIQCHA (keyinroq avtomatik qayta ko'rib chiqilsa,
        # TAKRORIY prototip qo'shib yuborishi mumkin edi).
        seqs_set = set(seqs)
        session["unknown_buffer"] = [
            e for e in session["unknown_buffer"] if e["seq"] not in seqs_set
        ]

        old_profile = session["speakers"].get(old_speaker)
        moved_protos: list[dict] = []
        if old_profile is not None:
            moved_protos = _pop_prototypes_by_seqs(old_profile, seqs)
            if not old_profile["prototypes"]:
                del session["speakers"][old_speaker]
                removed_speaker = old_speaker
                if session["last_speaker"] == old_speaker:
                    session["last_speaker"] = None

        if target_speaker == "new":
            target_speaker = session["next_id"]
            session["next_id"] += 1

        # Qo'shiladigan embedding(lar): agar bu bubble ALLAQACHON bir yoki
        # bir nechta prototip ("moved_protos") qo'shgan bo'lsa, AYNAN
        # O'SHALARNI (ASL seq teglari bilan) qayta ishlatamiz (qayta
        # hisoblash shart emas) — aks holda, agar audio yetarlicha uzun
        # bo'lsa, yangidan hisoblab, `primary_seq` bilan teglaymiz.
        # Foydalanuvchi TASDIQLAGANI uchun hech qanday o'xshashlik chegarasi
        # (SPK_ACCEPT/SPK_ENROLL_MIN_SIM) bu yerda tekshirilmaydi.
        if not moved_protos and duration_s >= MIN_TRIMMED_DURATION_S:
            moved_protos = [{"seq": primary_seq, "embedding": _embed(trimmed, sample_rate)}]

        if moved_protos:
            profile = session["speakers"].get(target_speaker)
            if profile is None:
                session["speakers"][target_speaker] = {"prototypes": []}
                profile = session["speakers"][target_speaker]
            for proto in moved_protos:
                _add_prototype(profile, proto["embedding"], proto["seq"])
        elif target_speaker not in session["speakers"]:
            session["speakers"][target_speaker] = {"prototypes": []}

        session["last_speaker"] = target_speaker

        merges = _check_and_merge(session)
        target_speaker = _apply_remap(target_speaker, merges)

    logger.info(
        f"[SPK][CORRECT] seqs={seqs} {old_speaker} -> {target_speaker}"
        + (f" (speaker={removed_speaker} prototip qolmagani uchun o'chirildi)" if removed_speaker is not None else "")
    )
    return {"speaker": target_speaker, "removed_speaker": removed_speaker, "merges": merges}


def remove_turn_prototype(session, speaker_id, seqs: list[int]) -> dict:
    """delete_turn uchun — shu bubble'ga tegishli BARCHA seq'lar (`seqs`,
    qarang dialog.py'dagi `merged_seqs`) `speaker_id`ga QO'SHGAN BARCHA
    prototiplar (agar bo'lsa) olib tashlanadi. Agar shu spikerda ENDI
    birorta ham prototip qolmasa, spiker profilining o'zi butunlay
    o'chiriladi.

    Qaytaradi: {"removed_speaker": int|None, "merges": [...]} — dialog.py
    buni correct_speaker bilan BIR XIL tarzda frontendga uzatadi."""
    with session["lock"]:
        removed_speaker = None
        profile = session["speakers"].get(speaker_id)
        if profile is not None:
            popped = _pop_prototypes_by_seqs(profile, seqs)
            if popped and not profile["prototypes"]:
                del session["speakers"][speaker_id]
                removed_speaker = speaker_id
                if session["last_speaker"] == speaker_id:
                    session["last_speaker"] = None

        merges = _check_and_merge(session)

    if removed_speaker is not None:
        logger.info(
            f"[SPK][DELETE] speaker={speaker_id} profili o'chirildi "
            f"(turn seqs={seqs} o'chirilgandan keyin prototip qolmadi)"
        )
    return {"removed_speaker": removed_speaker, "merges": merges}


def merge_speakers_manual(session, from_speaker, into_speaker) -> dict:
    """Foydalanuvchi ikkita spikerga BIR XIL ism bergandan keyin — frontend
    (DialogSessionView.jsx) "bitta odam sifatida birlashtiraymi?" so'rovini
    tasdiqlagach — chaqiriladi (dialog.py, `merge_speakers` action).
    `from_speaker`ning BARCHA prototiplari `into_speaker`ga ko'chiriladi
    (SPK_MAX_PROTOTYPES qoidasi — birinchi/"founding" prototiplar
    saqlanib qolishi — bilan, qarang _merge_profile_into), so'ng
    `from_speaker` profili butunlay o'chiriladi.

    REAL TESTDA TOPILGAN MUAMMO: avtomatik birlashtirish (_check_and_merge,
    SPK_MERGE_SIM=0.65) bu holatni ushlay olmagan edi (sims={3:0.72,
    4:0.61} — 0.61 < 0.65), chunki ikki spiker profili HALI yetarlicha
    o'xshash emas edi, garchi ism bo'yicha BITTA odam ekani aniq bo'lsa
    ham — shu sababli foydalanuvchi QO'LDA tasdiqlagan birlashtirish kerak.

    Qaytaradi: {"merged": bool, "merges": [(from, into, 1.0), ...]} —
    birinchi element shu AYNAN so'ralgan birlashtirish, qolganlari (agar
    bo'lsa) shu amaldan KEYIN avtomatik aniqlangan ZANJIRDAGI qo'shimcha
    birlashtirishlar (_check_and_merge)."""
    with session["lock"]:
        if from_speaker == into_speaker:
            return {"merged": False, "merges": []}
        from_profile = session["speakers"].get(from_speaker)
        into_profile = session["speakers"].get(into_speaker)
        if from_profile is None or into_profile is None:
            return {"merged": False, "merges": []}

        _merge_profile_into(into_profile, from_profile)
        del session["speakers"][from_speaker]
        if session["last_speaker"] == from_speaker:
            session["last_speaker"] = into_speaker

        # `chain_merges` — shu AMALDAN KEYIN (masalan natijada endi 2 va 5
        # ham o'xshash chiqib qolsa) ANIQLANGAN qo'shimcha birlashtirishlar.
        # Qaytariladigan ro'yxatning BIRINCHI elementi har doim ASL
        # (remap qilinmagan) `into_speaker`ga ishora qilishi SHART — dialog.py
        # _apply_merges_and_notify() ularni shu TARTIBDA ketma-ket qo'llaydi
        # (avval from->into, keyin zanjirni "ta'qib qiladi").
        chain_merges = _check_and_merge(session)

    final_into = _apply_remap(into_speaker, chain_merges)
    logger.info(f"[SPK][MERGE] reason=user from={from_speaker} into={final_into}")
    return {"merged": True, "merges": [(from_speaker, into_speaker, 1.0)] + chain_merges}
