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
        "lock": threading.Lock(),
    }


def _embed(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    model = _load_model()
    waveform = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).reshape(1, -1)
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


def identify_speaker(
    session,
    wav: np.ndarray,
    sample_rate: int,
    seq: int,
    detected_lang: str | None = None,
    debug_label_out: dict | None = None,
) -> dict:
    """Cheklanmagan sonli spiker uchun ONLAYN klasterlash — FAQAT ovoz
    asosida (til BUTUNLAY chiqarib tashlandi, `detected_lang` bu yerda
    FAQAT [SPK] logiga chiqarish uchun — qarorga HECH QANDAY ta'sir
    qilmaydi, tarjima yo'nalishi dialog.py'da mustaqil hal qilinadi).

    Qaytaradi: {"speaker": int, "uncertain": bool, "decision": str,
    "merges": [(from_id, into_id, sim), ...]}.

        "merges" — enrollment natijasida ixtiyoriy ikki spiker BIR ODAM
        ekani aniqlanib AVTOMATIK birlashtirilgan bo'lsa (qarang
        _check_and_merge), shu voqealar ro'yxati — dialog.py buni frontendga
        "speakers_merged" sifatida uzatishi va o'zining turn_history/
        last_turn_by_speaker holatini mos ravishda yangilashi SHART.

    Qaror daraxti (batafsil — [SPK] logiga qarang, chegaralarni real
    suhbat loglaridan sozlash uchun):
      1. sims = {speaker_id: eng yaqin prototip bilan o'xshashlik}.
      2. best = eng yuqori sims.
      3a. TURN (xom, trim qilinmagan) SPK_SHORT_TURN_S'dan qisqa VA
          best < SPK_ACCEPT VA avval kimdir gapirgan bo'lsa — ENG SO'NGGI
          gapirgan spikerga (uncertain=True), ovoz-o'xshashlikdan ko'ra
          "suhbat davomiyligi" ko'proq ishonchli signal (juda qisqa
          javoblarda ovoz-profil ishonchsiz).
      3b. best >= SPK_ACCEPT → shu spiker. Agar NUTQ (trim qilingan)
          >= SPK_MIN_ENROLL_SPEECH_S VA best < SPK_DUPLICATE_PROTOTYPE_SIM
          (ya'ni bu aynan takroriy namuna EMAS) VA (profilda hali
          SPK_BOOTSTRAP_PROTOTYPES'dan KAM prototip BOR YOKI best >=
          SPK_ENROLL_MIN_SIM) — YANGI prototip qo'shiladi.
      3c. SPK_NEW <= best < SPK_ACCEPT → eng yaqin spikerga, uncertain=True,
          prototip QO'SHILMAYDI.
      3d. best < SPK_NEW (yoki hali spiker yo'q):
          - nutq >= SPK_MIN_ENROLL_SPEECH_S → YANGI spiker DARHOL ochiladi
            (keyingi raqam, decision="new", uncertain=False) — "tasdiqlash"
            bosqichi YO'Q (bu eski, xato mexanizm edi — qarang fayl boshidagi
            izoh).
          - nutq qisqa → eng yaqin spikerga (bor bo'lsa) uncertain=True,
            yangi spiker OCHILMAYDI."""
    t0 = time.perf_counter()
    trimmed = _trim_trailing_silence(wav, sample_rate)
    trim_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[SPK][TRIM] wav_len={len(wav)} trimmed_len={len(trimmed)} took={trim_ms:.2f}ms")

    turn_s = len(wav) / sample_rate
    speech_s = len(trimmed) / sample_rate
    if trimmed.size > 0 and speech_s < MIN_TRIMMED_DURATION_S and speech_s < turn_s:
        logger.info(f"[SPK][TRIM] trim too aggressive ({turn_s:.2f}s -> {speech_s:.2f}s), using original audio")
        trimmed = wav
        speech_s = turn_s

    trimmed_rms = float(np.sqrt(np.mean(trimmed.astype(np.float64) ** 2))) if trimmed.size else 0.0

    embedding = None
    if trimmed.size > 0 and speech_s >= MIN_TRIMMED_DURATION_S:
        embedding = _embed(trimmed, sample_rate)

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
            else:
                best_id = None
                best_sim = -1.0

            decision: str
            result_speaker = None
            uncertain = False

            if embedding is None:
                # Audio juda qisqa/jim — hech qanday ovoz dalili yo'q.
                # Amalda deyarli bo'lmaydi (dialog.py'dagi MIN_TURN_SECONDS/
                # STT MIN_SPEECH_S allaqachon bundan oldin filtrlaydi), lekin
                # xavfsizlik uchun: oxirgi spikerga (bo'lsa), aks holda
                # (sessiyaning eng boshi) 1-spikerni prototipsiz ochamiz.
                decision = "no_embedding"
                uncertain = True
                if session["last_speaker"] is not None:
                    result_speaker = session["last_speaker"]
                else:
                    result_speaker = session["next_id"]
                    session["next_id"] += 1
                    speakers[result_speaker] = {"prototypes": []}

            elif turn_s < SPK_SHORT_TURN_S and best_sim < SPK_ACCEPT and session["last_speaker"] is not None:
                decision = "short"
                result_speaker = session["last_speaker"]
                uncertain = True

            elif best_sim >= SPK_ACCEPT:
                decision = "accept"
                result_speaker = best_id
                profile = speakers[best_id]
                if speech_s >= SPK_MIN_ENROLL_SPEECH_S and best_sim < SPK_DUPLICATE_PROTOTYPE_SIM:
                    is_bootstrap = len(profile["prototypes"]) < SPK_BOOTSTRAP_PROTOTYPES
                    if is_bootstrap or best_sim >= SPK_ENROLL_MIN_SIM:
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

            elif best_id is not None and SPK_NEW <= best_sim < SPK_ACCEPT:
                decision = "uncertain"
                result_speaker = best_id
                uncertain = True

            elif speech_s >= SPK_MIN_ENROLL_SPEECH_S:
                # best < SPK_NEW (yoki hali spiker yo'q) VA nutq yetarli
                # uzun — YANGI spiker DARHOL ochiladi, tasdiqlash YO'Q.
                new_id = session["next_id"]
                session["next_id"] += 1
                speakers[new_id] = {"prototypes": [{"seq": seq, "embedding": embedding}]}
                decision = "new"
                result_speaker = new_id

            else:
                # best < SPK_NEW (yoki spiker yo'q) VA nutq qisqa.
                if best_id is not None:
                    decision = "uncertain"
                    result_speaker = best_id
                    uncertain = True
                else:
                    # Sessiyaning eng boshi, lekin birinchi gap juda qisqa —
                    # tizim hech qachon "spiker yo'q" holatida qolib
                    # ketmasligi uchun 1-spiker shunchaki uncertain bilan
                    # ochiladi (prototip SPK_MIN_ENROLL_SPEECH_S talabi
                    # bajarilmagani uchun hali qo'shilmaydi).
                    new_id = session["next_id"]
                    session["next_id"] += 1
                    speakers[new_id] = {"prototypes": []}
                    decision = "new_short"
                    result_speaker = new_id
                    uncertain = True

            # Har bir enrollment/yangi-spikerdan KEYIN — ixtiyoriy ikki
            # spiker aslida bitta odam bo'lib qolmaganini tekshiramiz
            # (qarang SPK_MERGE_SIM izohi yuqorida).
            merges = _check_and_merge(session)
            result_speaker = _apply_remap(result_speaker, merges)

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
