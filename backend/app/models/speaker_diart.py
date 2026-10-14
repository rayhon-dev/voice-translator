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
# ---------------------------------------------------------------------------

# Shu va undan yuqori o'xshashlikda turn SHU spikerga ANIQ tegishli deb
# qabul qilinadi.
SPK_ACCEPT = float(os.environ.get("SPK_ACCEPT", "0.55"))
# Shundan past o'xshashlikda turn HECH QANDAY mavjud spikerga tegishli
# EMAS deb hisoblanadi (yangi ovoz nomzodi).
SPK_NEW = float(os.environ.get("SPK_NEW", "0.35"))
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
        # speaker_id(int) -> {"prototypes": [np.ndarray, ...]} — birinchi
        # (index 0) prototip "founding" namuna, HECH QACHON FIFO orqali
        # olib tashlanmaydi (qarang _add_prototype).
        "speakers": {},
        # Hali TASDIQLANMAGAN yangi-spiker nomzodi: {"tentative_id":,
        # "embedding":, "speech_s":, "seq":} yoki None. Bir vaqtning
        # o'zida FAQAT bitta nomzod kuzatiladi (qarang identify_speaker).
        "pending_candidate": None,
        "next_id": 1,
        # SPK_SHORT_TURN_S'dan qisqa, noaniq turnlar uchun fallback —
        # eng SO'NGGI qaytarilgan (pending/uncertain ham) spiker.
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
    return max(_cosine_sim(embedding, proto) for proto in profile["prototypes"])


def _add_prototype(profile: dict, embedding: np.ndarray) -> None:
    profile["prototypes"].append(embedding)
    if len(profile["prototypes"]) > SPK_MAX_PROTOTYPES:
        # index 0 — "founding" prototip, HAR DOIM saqlanadi; eng ESKI
        # (lekin founding EMAS) prototip index 1'da turadi.
        del profile["prototypes"][1]


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

    Qaytaradi: {"speaker": int, "uncertain": bool, "pending": bool,
    "decision": str, "confirmed_pending_seq": int|None}.

        "confirmed_pending_seq" — agar bu chaqiruv ILGARI "pending" deb
        belgilangan nomzodni HAQIQIY yangi spiker sifatida TASDIQLASA,
        o'sha ESKI turn'ning seq'i shu yerda qaytadi — dialog.py shu
        seq'ni YANGI spiker ID bilan qayta yuborishi (upsert) kerak,
        aks holda o'sha bubble abadiy "pending/uncertain" holatda qolib
        ketadi.

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
          (ya'ni bu aynan takroriy namuna EMAS) — YANGI prototip
          qo'shiladi (SPK_MAX_PROTOTYPES'dan oshsa, eng eskisi — founding
          BUNDAN MUSTASNO — olib tashlanadi).
      3c. SPK_NEW <= best < SPK_ACCEPT → eng yaqin spikerga, uncertain=True,
          prototip QO'SHILMAYDI.
      3d. best < SPK_NEW (yoki hali spiker yo'q):
          - nutq >= SPK_MIN_ENROLL_SPEECH_S:
            - spiker UMUMAN yo'q bo'lsa → BIRINCHI spiker sifatida
              DARHOL ochiladi (tasdiqlash shart emas — taqqoslaydigan
              hech kim yo'q).
            - aks holda, agar ALLAQACHON pending nomzod bo'lsa VA shu
              nomzod bilan o'xshashlik >= SPK_ACCEPT → nomzod TASDIQLANADI
              (haqiqiy, keyingi raqamli spiker bo'ladi;
              confirmed_pending_seq orqali eski turn ham yangilanadi).
            - aks holda → YANGI nomzod ochiladi (pending=True,
              uncertain=True), hali HAQIQIY spiker emas.
          - nutq qisqa → eng yaqin spikerga (bor bo'lsa) uncertain=True,
            yangi spiker/nomzod OCHILMAYDI."""
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

            cand_sim = None
            pending_candidate = session["pending_candidate"]
            if embedding is not None and pending_candidate is not None:
                cand_sim = _cosine_sim(embedding, pending_candidate["embedding"])

            if sims:
                best_id = max(sims, key=sims.get)
                best_sim = sims[best_id]
            else:
                best_id = None
                best_sim = -1.0

            decision: str
            result_speaker = None
            uncertain = False
            pending = False
            confirmed_pending_seq = None

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
                    _add_prototype(profile, embedding)
                    logger.info(
                        f"[SPK][ENROLL] speaker={best_id} yangi prototip qo'shildi "
                        f"(sim={best_sim:.3f} < {SPK_DUPLICATE_PROTOTYPE_SIM}) — "
                        f"jami prototiplar={len(profile['prototypes'])}"
                    )

            elif best_id is not None and SPK_NEW <= best_sim < SPK_ACCEPT:
                decision = "uncertain"
                result_speaker = best_id
                uncertain = True

            elif speech_s >= SPK_MIN_ENROLL_SPEECH_S:
                if not speakers:
                    new_id = session["next_id"]
                    session["next_id"] += 1
                    speakers[new_id] = {"prototypes": [embedding]}
                    decision = "new"
                    result_speaker = new_id
                elif pending_candidate is not None and cand_sim is not None and cand_sim >= SPK_ACCEPT:
                    new_id = pending_candidate["tentative_id"]
                    speakers[new_id] = {"prototypes": [pending_candidate["embedding"]]}
                    if speech_s >= SPK_MIN_ENROLL_SPEECH_S and cand_sim < SPK_DUPLICATE_PROTOTYPE_SIM:
                        _add_prototype(speakers[new_id], embedding)
                    decision = "confirm_new"
                    result_speaker = new_id
                    confirmed_pending_seq = pending_candidate["seq"]
                    session["pending_candidate"] = None
                    # MUHIM: tentative_id ENDI haqiqiy spiker sifatida
                    # "iste'mol qilindi" — next_id shuni hisobga olib
                    # oldinga surilishi SHART, aks holda keyingi pending
                    # nomzod xuddi shu ID'ni qayta ishlatib, ENDI MAVJUD
                    # spikerning prototiplarini ustiga YOZIB yuborgan
                    # bo'lardi (real testda topilgan xato).
                    session["next_id"] = max(session["next_id"], new_id + 1)
                else:
                    tentative_id = session["next_id"]
                    session["pending_candidate"] = {
                        "tentative_id": tentative_id,
                        "embedding": embedding,
                        "speech_s": speech_s,
                        "seq": seq,
                    }
                    decision = "pending"
                    result_speaker = tentative_id
                    uncertain = True
                    pending = True
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

            session["last_speaker"] = result_speaker

            sims_str = ", ".join(f"{sid}:{sim:.2f}" for sid, sim in sims.items())
            if cand_sim is not None:
                sims_str = f"{sims_str}, " if sims_str else ""
                sims_str += f"cand:{cand_sim:.2f}"
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
                "pending": pending,
                "decision": decision,
                "confirmed_pending_seq": confirmed_pending_seq,
            }
    except Exception as e:
        logger.error(f"[SPK][ERROR] identify_speaker raised {e!r}")
        raise


def has_known_speakers(session) -> bool:
    """dialog.py'dagi "fast path" uchun — kamida BITTA spiker uchun ovoz
    profili allaqachon mavjud bo'lsa, juda qisqa/juda past-energiyali
    (frontend VAD'ning past chegarasi tufayli shovqindan ham "nutq" deb
    yuborilgan bo'lishi mumkin bo'lgan) audio bo'laklarini STT/embedding'ga
    yubormasdan oldin saralab tashlash mumkin. Eski (A/B ikki slot
    to'lgan-to'lmaganini tekshiruvchi) `both_slots_bound()`dan farqli —
    endi spikerlar soni cheklanmagani uchun "kamida bitta spiker bor"
    yetarli signal."""
    with session["lock"]:
        return len(session["speakers"]) > 0


def apply_speaker_correction(session, wav: np.ndarray, sample_rate: int, target_speaker) -> int:
    """Har bir bubble'dagi "spikerni almashtirish" menyusi uchun —
    foydalanuvchi mavjud spikerlardan birini YOKI "Yangi spiker"ni
    tanlaganda chaqiriladi (dialog.py, `correct_speaker` action).
    `target_speaker` — mavjud spiker ID (int) yoki literal `"new"` satri.

    Foydalanuvchining o'zi tasdiqlagani — identify_speaker()dagi avtomatik
    qarordan HAM ishonchli signal, shuning uchun hech qanday o'xshashlik
    chegarasi tekshirilmasdan, embedding to'g'ridan-to'g'ri TO'G'RI
    spikerning prototiplariga qo'shiladi. Qaytaradi: YAKUNIY spiker ID
    (target_speaker="new" bo'lsa, yangi ajratilgan ID)."""
    trimmed = _trim_trailing_silence(wav, sample_rate)
    duration_s = len(trimmed) / sample_rate
    original_duration_s = len(wav) / sample_rate
    if trimmed.size > 0 and duration_s < MIN_TRIMMED_DURATION_S and duration_s < original_duration_s:
        trimmed = wav
        duration_s = original_duration_s

    with session["lock"]:
        if target_speaker == "new":
            target_speaker = session["next_id"]
            session["next_id"] += 1

        if duration_s < MIN_TRIMMED_DURATION_S:
            logger.info(
                f"[SPK][CORRECT] audio juda qisqa ({duration_s:.2f}s) — speaker={target_speaker} "
                f"uchun prototip QO'SHILMADI"
            )
            if target_speaker not in session["speakers"]:
                session["speakers"][target_speaker] = {"prototypes": []}
            session["last_speaker"] = target_speaker
            return target_speaker

        embedding = _embed(trimmed, sample_rate)
        profile = session["speakers"].get(target_speaker)
        if profile is None:
            session["speakers"][target_speaker] = {"prototypes": [embedding]}
        else:
            _add_prototype(profile, embedding)
        session["last_speaker"] = target_speaker

    logger.info(f"[SPK][CORRECT] speaker={target_speaker} — foydalanuvchi tuzatishi orqali prototip qo'shildi")
    return target_speaker


def remove_speaker(session, speaker_id) -> None:
    """delete_turn uchun — dialog.py, shu spikerni "ochgan" (yagona) turn
    o'chirilganda, shu spikerning ENDI hech qanday turn'ga tegishli
    emasligini tekshirgandan KEYIN chaqiradi. Spiker profilini, agar u
    hozirgi `pending_candidate`/`last_speaker` bo'lsa, shu holatlardan ham
    tozalaydi — aks holda o'chirilgan spiker "jim" holda sessiya holatida
    osilib qolgan bo'lardi."""
    with session["lock"]:
        if speaker_id in session["speakers"]:
            del session["speakers"][speaker_id]
            logger.info(f"[SPK][DELETE] speaker={speaker_id} profili o'chirildi")
        if session["last_speaker"] == speaker_id:
            session["last_speaker"] = None
        if session["pending_candidate"] is not None and session["pending_candidate"]["tentative_id"] == speaker_id:
            session["pending_candidate"] = None
