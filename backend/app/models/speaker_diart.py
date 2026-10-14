import collections
import datetime
import glob
import json
import os
import threading
import time

import numpy as np
import soundfile as sf
import torch

from app import diart_compat
from diart import models as m
from app.config import SAVE_DEBUG_AUDIO, DEBUG_AUDIO_DIR
import logging

logger = logging.getLogger(__name__)



_embedding_model = None

RMS_SILENCE_THRESHOLD = 0.003  # stt.py bilan bir xil qiymat
_TRIM_FRAME_MS = 50

# _trim_trailing_silence() 50ms'lik LAHZAVIY kadr RMS'ini tekshiradi
MIN_TRIMMED_DURATION_S = 0.5

# ---------------------------------------------------------------------------
# DIALOG_PLAN.md 2-bosqich: spiker aniqlash endi OVOZ + TIL signallarining
# BIRLASHGAN balli orqali hal qilinadi — eski alohida bosqichlar (avval
# FAQAT ovoz orqali klasterlash/bog'lash, til esa FAQAT klaster birinchi
# marta tashkil bo'lganda tekshirilardi, "ovoz orqali, til tekshirilmadi"
# yo'li) BUTUNLAY OLIB TASHLANDI. Sabab — real testda topilgan xato:
# ikkinchi spikerning ovozi birinchisiga tasodifan o'xshab qolgani (sim
# 0.45-0.66, eski SPEAKER_MATCH_THRESHOLD=0.4'dan yuqori) uchun uning
# nutqi YANGI klaster emas, birinchisiga qo'shilib ketgan edi — til farqli
# bo'lsa ham, mantiq buni TEKSHIRMAGANI uchun bu signal yo'qolgan edi.
#
# Yangi qaror: har bir turn uchun
#     score_X = w_lang * p_lang(X) + w_voice * voice_sim_norm(X)   (X = A, B)
# hisoblanadi va kattarog'i g'olib e'lon qilinadi — FAQAT agar ikkisi
# orasidagi farq (margin) AMBIGUOUS_MARGIN'dan katta bo'lsa (pastga
# qarang). `identify_speaker()`ga qarang.
# ---------------------------------------------------------------------------

# Bazaviy (teng) vaznlar — turn "til aniq hal qiluvchi" rejimida bo'lmaganda
# ishlatiladi (qarang LANG_DECISIVE_* va _effective_weights()). Boshlang'ich
# qiymat sifatida teng tanlangan — real eval_dialog.py natijalariga qarab
# keyinroq sozlanishi mumkin, shuning uchun konstanta sifatida yuqorida.
W_LANG = 0.5
W_VOICE = 0.5

# Til "hal qiluvchi" (deyarli yakuniy qarorni beradigan) rejimga o'tadigan
# shartlar: turn yetarlicha uzun (qisqa turnlarda til-aniqlash ishonchsiz
# bo'ladi) VA til ishonchi "noaniq oraliq"dan (pastga qarang) tashqarida —
# ya'ni STT ANIQ bir tilni ishonch bilan ko'rsatgan. Shu holatda ovoz
# deyarli e'tiborga olinmaydi (LANG_DECISIVE_W_VOICE juda kichik) — chunki
# STT'ning aniq natijasi ovoz-o'xshashlikdan ANCHA ishonchli signal.
LANG_DECISIVE_MIN_DURATION_S = 1.5
LANG_AMBIGUOUS_PROB_LOW = 0.4
LANG_AMBIGUOUS_PROB_HIGH = 0.6
LANG_DECISIVE_W_LANG = 0.9
LANG_DECISIVE_W_VOICE = 0.1

# Ovoz profili (markaz) FAQAT shu til-ishonchidan yuqori turnlardan
# tashkil topadi/yangilanadi — past/noaniq ishonchli turnlar o'ziga
# spiker tayinlanishi mumkin (agar margin yetarli bo'lsa), lekin ULARNING
# embeddingi markazga HECH QACHON qo'shilmaydi. Bu A3/A4'ni (sifatsiz
# "founding sample" markazni buzishi) ildizidan hal qiladi — eski
# MIN_RMS_FOR_NEW_CLUSTER/MIN_LANG_PROB_FOR_NEW_CLUSTER/RMS-asosidagi
# tekshiruvlar endi keraksiz, chunki til-ishonch ulardan ANCHA to'g'ridan
# to'g'ri va ishonchli signal.
CENTROID_UPDATE_MIN_LANG_PROB = 0.85

# Ikki ball orasidagi farq shundan kichik bo'lsa, natija "ambiguous" deb
# belgilanadi (A2) — noto'g'ri, lekin "ishonchli ko'ringan" qarordan ko'ra
# ochiq tanlashni so'rash xavfsizroq.
AMBIGUOUS_MARGIN = 0.15

# REAL TESTDA TOPILGAN MUAMMO (3-band): w_lang=0.90 bo'lgani uchun til
# signali amalda deyarli HAR DOIM g'olib bo'ladi — HATTO ovoz butunlay
# QARAMA-QARSHI narsani ko'rsatsa ham (masalan ovoz aniq A, til esa B
# tomonidan). Bunday holatda g'olibni (til bo'yicha) o'zgartirmaymiz —
# lekin uning centroidiga BOSHQA odamning ovozini "yuklab" qo'yish ham
# xavfli (keyingi turnlarda ovoz-aniqlashni butunlay buzadi). Shu sababli:
# ikkala profil ham mavjud bo'lganda, agar OVOZ g'olibi (voice_winner) TIL
# g'olibidan (lang_winner) farq qilsa VA ovoz ballari orasidagi farq
# yetarlicha katta (ishonchli) bo'lsa — demak ovoz signali "jim" emas, u
# HAQIQATDA boshqa odamni ko'rsatmoqda — markaz YANGILANMAYDI (hech kim
# uchun) va turn "conflict" deb belgilanadi (dialog.py buni frontendga
# uzatadi, qarang DialogBubble.jsx'dagi "⚠ spiker noaniq" belgisi).
VOICE_CONFLICT_MARGIN = 0.10

# Har bir A/B ovoz profili uchun so'nggi shuncha SIFATLI (CENTROID_UPDATE_
# MIN_LANG_PROB'dan yuqori ishonchli) embedding saqlanadi; markaz —
# shularning (+ doim saqlanadigan "founding" namunaning, pastga qarang)
# o'rtachasi. Eski qiymat (5) juda kichik edi — bitta "burchakka og'ib
# ketgan" namuna markazni sezilarli siljitib yuborishi mumkin edi; 15-20
# oraliq buferni "drift"ga ancha chidamli qiladi.
CENTROID_BUFFER_SIZE = 18


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


def _load_model():
    global _embedding_model
    if _embedding_model is None:
        model = m.EmbeddingModel.from_pyannote("pyannote/embedding")
        model.to(torch.device("cpu"))
        model.eval()
        _embedding_model = model
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
    bu funksiyaning o'zi disk I/O'ni kutmasdan darhol qaytadi.

    Fayl nomini (timestamp'ga asoslangani uchun) THREAD BOSHLANISHIDAN OLDIN,
    sinxron hisoblab, qaytaradi — shunda chaqiruvchi (identify_speaker)
    ushbu nomni debug_label_out'ga yozib qo'yishi mumkin, dialog.py esa keyinroq
    (speaker/lang/text ANIQLANGANDAN KEYIN) save_turn_label() orqali shu AYNAN
    faylga mos .json yaratishi uchun — disk yozishning o'zi hamon asinxron."""
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


def save_turn_label(wav_filename: str | None, speaker: str | None, lang: str | None, text: str | None) -> None:
    """dialog.py tomonidan, bitta turn uchun YAKUNIY natija (A/B spiker, til,
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
    logger.info(f"[SPK] pyannote embedding model loaded in {time.perf_counter() - t0:.2f}s")
    if not run_inference:
        return
    try:
        session = create_session()
        identify_speaker(
            session,
            np.random.randn(16000).astype(np.float32) * 0.05,
            16000,
            speaker_langs={"A": "en", "B": "ru"},
            detected_lang="en",
            lang_prob=0.9,
        )
        logger.info("[SPK] warm-up inference done")
    except Exception as e:
        logger.info(f"[SPK] warm-up inference skipped: {e!r}")


def create_session():
    return {
        "embedding_model": _load_model(),
        # "A"/"B" -> {"founding_embedding": np.ndarray, "embeddings": deque,
        #             "centroid": np.ndarray} — kalit faqat shu spiker
        # UCHUN ovoz profili tashkil topgandan keyin paydo bo'ladi (qarang
        # identify_speaker/enroll_speaker).
        "voices": {},
        "lock": threading.Lock(),
    }


def _embed(embedding_model, wav: np.ndarray) -> np.ndarray:
    waveform = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).reshape(1, 1, -1)
    with torch.no_grad():
        emb = embedding_model(waveform)
    emb = emb.squeeze(0).cpu().numpy().astype(np.float64)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def _recompute_voice_centroid(profile: dict) -> None:
    """Markaz — "founding" (asos) namuna + sirpanuvchi bufердаgi hamma
    namunalarning o'rtachasi. `founding_embedding` ATAYLAB `embeddings`
    deque'sidan TASHQARIDA saqlanadi (A3): deque `maxlen=CENTROID_BUFFER_
    SIZE` bo'lgani uchun eski namunalarni o'zi "unutadi", lekin founding
    namuna HECH QACHON shu tarzda chiqarib tashlanmaydi — klaster "kim
    ekanligi"ning barqaror langari sifatida doim markazga hissa qo'shadi."""
    vectors = [profile["founding_embedding"], *profile["embeddings"]]
    mean = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean)
    profile["centroid"] = mean / norm if norm > 0 else mean


def _voice_profile_score(session, speaker: str, embedding: np.ndarray | None) -> float:
    """[0,1] oraliqda — `speaker`ning OVOZ profili asosida ushbu embedding
    unga qanchalik mos kelishi (cosine o'xshashlik [-1,1]'dan chiziqli
    o'tkazilgan). Profil hali tashkil topmagan BO'LSA yoki `embedding` yo'q
    (audio juda qisqa/jim bo'lib trim qilingan) bo'lsa — NEUTRAL (0.5, ya'ni
    "hech qanday dalil yo'q") qaytaradi, shunda yakuniy qaror shu daraja
    bo'yicha butunlay til signaliga tortiladi."""
    profile = session["voices"].get(speaker)
    if profile is None or embedding is None:
        return 0.5
    sim = float(np.dot(embedding, profile["centroid"]))
    return min(1.0, max(0.0, (sim + 1.0) / 2.0))


def _lang_profile_scores(
    speaker_langs: dict, detected_lang: str | None, lang_prob: float
) -> tuple[float, float]:
    """(p_lang(A), p_lang(B)) qaytaradi. `detected_lang` A yoki B'ning
    TANLAGAN tiliga mos kelsa, shu tomonga `lang_prob` (stt.py'da
    ALLAQACHON 2 til orasida qayta normallashtirilgan — qarang
    _detect_language_restricted) beriladi, ikkinchi tomonga `1 - lang_prob`
    (faqat 2 nomzod bo'lgani uchun ehtimolliklar yig'indisi 1 bo'ladi).
    `detected_lang` ikkisiga ham mos kelmasa (STT candidate_languages
    orqali shu ikkisiga cheklangani uchun amalda bo'lmasligi kerak — faqat
    himoya sifatida) — ikkisiga ham NEUTRAL (0.5)."""
    if detected_lang == speaker_langs.get("A"):
        return lang_prob, 1.0 - lang_prob
    if detected_lang == speaker_langs.get("B"):
        return 1.0 - lang_prob, lang_prob
    return 0.5, 0.5


def _effective_weights(duration_s: float, lang_prob: float) -> tuple[float, float]:
    """(w_lang, w_voice). Til "hal qiluvchi" rejimga o'tadi (deyarli
    yakuniy qarorni belgilaydi) FAQAT turn >= LANG_DECISIVE_MIN_DURATION_S
    VA lang_prob "noaniq oraliq" [LANG_AMBIGUOUS_PROB_LOW,
    LANG_AMBIGUOUS_PROB_HIGH]'dan TASHQARIDA bo'lsa — ya'ni STT yetarlicha
    uzun audio'da ANIQ bir tilni ko'rsatgan. Aks holda (qisqa turn YOKI
    til ishonchi noaniq) bazaviy (teng) vaznlarga qaytiladi va ovoz
    amalda hal qiluvchi bo'lib qoladi."""
    lang_is_decisive = duration_s >= LANG_DECISIVE_MIN_DURATION_S and not (
        LANG_AMBIGUOUS_PROB_LOW <= lang_prob <= LANG_AMBIGUOUS_PROB_HIGH
    )
    if lang_is_decisive:
        return LANG_DECISIVE_W_LANG, LANG_DECISIVE_W_VOICE
    return W_LANG, W_VOICE


def identify_speaker(
    session,
    wav: np.ndarray,
    sample_rate: int,
    speaker_langs: dict,
    detected_lang: str | None,
    lang_prob: float,
    debug_label_out: dict | None = None,
) -> tuple[str, dict]:
    """DIALOG_PLAN.md 2-bosqich: bitta turn uchun A/B qarorini TIL va OVOZ
    signallarining birlashgan balli orqali chiqaradi:

        score_X = w_lang * p_lang(X) + w_voice * voice_sim_norm(X)   (X=A,B)

    Qaytaradi: (`"A"` | `"B"` | `"ambiguous"`, debug_info_dict).
    HECH QACHON `None`/"unknown_speaker" qaytarmaydi — bu suhbat ANIQ 2
    kishi orasida ekani ma'lum (speaker_langs orqali), shuning uchun har
    bir (bo'sh bo'lmagan) turn ALBATTA A, B yoki (agar ikkisi ham deyarli
    teng ball olsa) "ambiguous" bo'ladi. Chaqiruvchi (dialog.py) `detected_
    lang`/`lang_prob`ni BU FUNKSIYA chaqirilishidan OLDIN STT orqali olishi
    kerak — eski kodda STT faqat "founding sample" tekshiruvi uchun shartli
    chaqirilardi, endi til ASOSIY signallardan biri bo'lgani uchun HAR DOIM
    oldindan kerak.

    Eski alohida bosqichlar — FAQAT ovoz orqali klasterlash
    (SPEAKER_MATCH_THRESHOLD), keyin FAQAT birinchi marta klaster ochilganda
    til tekshirish, shundan keyin "ovoz orqali, til tekshirilmadi" doimiy
    bog'lash — BUTUNLAY OLIB TASHLANDI. Buning o'rniga har bir turn
    MUSTAQIL baholanadi (streak/pending holat saqlanmaydi)."""
    t0 = time.perf_counter()
    trimmed = _trim_trailing_silence(wav, sample_rate)
    trim_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[SPK][TRIM] wav_len={len(wav)} trimmed_len={len(trimmed)} took={trim_ms:.2f}ms")

    original_duration_s = len(wav) / sample_rate
    trimmed_duration_s = len(trimmed) / sample_rate
    if trimmed.size > 0 and trimmed_duration_s < MIN_TRIMMED_DURATION_S and trimmed_duration_s < original_duration_s:
        logger.info(f"[SPK][TRIM] trim too aggressive ({original_duration_s:.2f}s -> "
            f"{trimmed_duration_s:.2f}s), using original audio")
        trimmed = wav

    duration_s = len(trimmed) / sample_rate
    trimmed_rms = float(np.sqrt(np.mean(trimmed.astype(np.float64) ** 2))) if trimmed.size else 0.0

    try:
        with session["lock"]:
            # Audio juda qisqa/butunlay jim bo'lsa, ishonchli embedding
            # OLISH MUMKIN EMAS — lekin bu, eski koddan farqli, ENDI turnni
            # butunlay rad etishga sabab BO'LMAYDI: voice_sim ikki tomon
            # uchun ham NEUTRAL (0.5) qoladi va qaror butunlay til signaliga
            # tayanadi (agar u ham noaniq bo'lsa, margin tekshiruvi buni
            # "ambiguous" deb to'g'ri belgilaydi — pastga qarang).
            embedding = None
            if trimmed.size > 0 and duration_s >= MIN_TRIMMED_DURATION_S:
                embedding = _embed(session["embedding_model"], trimmed)

            voice_a = _voice_profile_score(session, "A", embedding)
            voice_b = _voice_profile_score(session, "B", embedding)
            lang_a, lang_b = _lang_profile_scores(speaker_langs, detected_lang, lang_prob)
            w_lang, w_voice = _effective_weights(duration_s, lang_prob)

            score_a = w_lang * lang_a + w_voice * voice_a
            score_b = w_lang * lang_b + w_voice * voice_b

            debug_info = {
                "score_a": score_a,
                "score_b": score_b,
                "voice_a": voice_a,
                "voice_b": voice_b,
                "lang_a": lang_a,
                "lang_b": lang_b,
                "w_lang": w_lang,
                "w_voice": w_voice,
            }

            logger.info(f"[SPK][SCORE] A={score_a:.3f} (lang={lang_a:.2f} voice={voice_a:.2f}) "
                f"B={score_b:.3f} (lang={lang_b:.2f} voice={voice_b:.2f}) "
                f"w_lang={w_lang:.2f} w_voice={w_voice:.2f} duration={duration_s:.2f}s "
                f"detected_lang={detected_lang!r} lang_prob={lang_prob:.2f}")

            margin = abs(score_a - score_b)
            if margin < AMBIGUOUS_MARGIN:
                logger.info(f"[SPK][DEBUG] ambiguous (margin={margin:.3f} < {AMBIGUOUS_MARGIN}) — "
                    f"foydalanuvchidan tanlash so'ralishi kerak")
                wav_filename = _save_debug_audio_async(trimmed, sample_rate, duration_s, trimmed_rms, "ambiguous")
                if debug_label_out is not None:
                    debug_label_out["wav_filename"] = wav_filename
                return "ambiguous", debug_info

            winner = "A" if score_a > score_b else "B"

            # 3-band: ikkala profil ham mavjud bo'lgandagina voice_winner
            # mazmunli (aks holda _voice_profile_score mavjud bo'lmagan
            # tomon uchun NEUTRAL 0.5 qaytaradi — "g'olib" tasodifiy bo'lib
            # qoladi, haqiqiy ziddiyat emas).
            conflict = False
            if "A" in session["voices"] and "B" in session["voices"]:
                voice_winner = "A" if voice_a > voice_b else "B"
                lang_winner = "A" if lang_a > lang_b else "B"
                voice_margin = abs(voice_a - voice_b)
                if voice_winner != lang_winner and voice_margin > VOICE_CONFLICT_MARGIN:
                    conflict = True
                    debug_info["conflict"] = True
                    logger.info(
                        f"[SPK][CONFLICT] lang→{lang_winner} voice→{voice_winner} "
                        f"margin={voice_margin:.3f} — centroid yangilanmadi"
                    )

            # Markaz FAQAT yuqori til-ishonchli turnlardan tashkil topadi/
            # yangilanadi (A3/A4) — g'olib ANIQLANGANDAN keyin, HAR DOIM,
            # g'olib qanday aniqlanganidan (til yoki ovoz hal qilganidan)
            # qat'i nazar shu BITTA shart tekshiriladi. 3-band: til/ovoz
            # ziddiyati bo'lsa, bu tekshiruv UMUMAN o'tkazilmaydi — hech
            # qaysi spiker centroidi yangilanmaydi (g'olibning o'zi esa
            # o'zgarishsiz, hali til bo'yicha qolaveradi).
            if conflict:
                pass
            elif embedding is not None and lang_prob >= CENTROID_UPDATE_MIN_LANG_PROB:
                profile = session["voices"].get(winner)
                if profile is None:
                    session["voices"][winner] = {
                        "founding_embedding": embedding,
                        "embeddings": collections.deque(maxlen=CENTROID_BUFFER_SIZE),
                        "centroid": embedding,
                    }
                    logger.info(f"[SPK][DEBUG] {winner!r} uchun YANGI ovoz profili tashkil topdi "
                        f"(lang_prob={lang_prob:.2f} >= {CENTROID_UPDATE_MIN_LANG_PROB})")
                else:
                    profile["embeddings"].append(embedding)
                    _recompute_voice_centroid(profile)
                    logger.info(f"[SPK][DEBUG] {winner!r} ovoz profili yangilandi "
                        f"(lang_prob={lang_prob:.2f} >= {CENTROID_UPDATE_MIN_LANG_PROB})")
            elif embedding is not None:
                logger.info(f"[SPK][DEBUG] {winner!r} g'olib, lekin markaz YANGILANMADI "
                    f"(lang_prob={lang_prob:.2f} < {CENTROID_UPDATE_MIN_LANG_PROB})")

            wav_filename = _save_debug_audio_async(
                trimmed, sample_rate, duration_s, trimmed_rms,
                f"speaker_{winner}_conflict" if conflict else f"speaker_{winner}",
            )
            if debug_label_out is not None:
                debug_label_out["wav_filename"] = wav_filename
            return winner, debug_info
    except Exception as e:
        logger.error(f"[SPK][ERROR] identify_speaker raised {e!r}")
        raise


def both_slots_bound(session) -> bool:
    """A VA B ikkisi ham allaqachon OVOZ PROFILIGA ega (founded) — ya'ni
    suhbatdagi ikkita spikerning ovoz "imzosi" aniq. dialog.py'dagi "fast
    path" uchun ishlatiladi: shu holatda juda qisqa/juda past-energiyali
    (frontend VAD'ning past chegarasi tufayli shovqindan ham "nutq" deb
    yuborilgan bo'lishi mumkin bo'lgan) audio bo'laklarini STT/embedding'ga
    yubormasdan oldin saralab tashlash mumkin."""
    with session["lock"]:
        return "A" in session["voices"] and "B" in session["voices"]


def enroll_speaker(session, speaker: str, wav: np.ndarray, sample_rate: int) -> None:
    """KEYINGI ISH uchun joy (DIALOG_PLAN.md 2-bosqich, ixtiyoriy band):
    til tanlangandan keyin, suhbat boshlanishidan OLDIN, har bir spiker
    3-4 soniya namunaviy gapirib "ro'yxatdan o'tishi" (enrollment) orqali
    ovoz profilini SUHBAT boshidanoq, til signaliga bog'liq bo'lmay turib
    tayyorlab qo'yish mumkin — bu sovuq-start muammosini (suhbatning
    birinchi bir-ikki turnida hali hech kim uchun ovoz profili yo'qligi,
    shuning uchun qaror butunlay til signaliga tayanishi) butunlay hal
    qiladi.

    Hozircha TO'LIQ ishlatilmaydi — na dialog.py (websocket oqimi), na
    frontend (UI qadam/tugma) buni chaqirmaydi; bu shunchaki tayyor,
    ishlaydigan backend funksiyasi, keyingi bosqichda ulash uchun.
    Enrollment paytida foydalanuvchi ANIQ KIM ekanini UI orqali (masalan
    "A, hozir 3 soniya gapiring" ko'rsatmasi bilan) o'zi tasdiqlagan
    bo'ladi — shuning uchun bu yerda til-ishonch tekshiruvi (CENTROID_
    UPDATE_MIN_LANG_PROB) shart emas, berilgan audio to'g'ridan-to'g'ri
    "founding" namuna sifatida qabul qilinadi."""
    embedding = _embed(session["embedding_model"], wav)
    with session["lock"]:
        session["voices"][speaker] = {
            "founding_embedding": embedding,
            "embeddings": collections.deque(maxlen=CENTROID_BUFFER_SIZE),
            "centroid": embedding,
        }
    logger.info(f"[SPK][ENROLL] {speaker!r} uchun ovoz profili enrollment orqali tashkil topdi")


def apply_speaker_correction(session, wav: np.ndarray, sample_rate: int, speaker: str) -> None:
    """DIALOG_PLAN.md 4-bosqich: har bir bubble'dagi "spikerni almashtirish"
    tugmasi uchun — foydalanuvchi "bu A emas, B edi" deb TUZATGANDA
    chaqiriladi (dialog.py, `correct_speaker` action). Foydalanuvchining
    o'zi tasdiqlagani — identify_speaker()dagi til/ovoz balli yoki
    CENTROID_UPDATE_MIN_LANG_PROB tekshiruvidan HAM ishonchli signal,
    shuning uchun enroll_speaker() kabi, hech qanday ishonch chegarasi
    tekshirilmasdan, embedding to'g'ridan-to'g'ri TO'G'RI spikerning
    markaziga qo'shiladi.

    Eski (noto'g'ri) spikerning markazidan bu namunani olib tashlash
    amalga oshirilmaydi (qaysi namuna ekanini qidirib topish/pop qilish
    keraksiz murakkablik qo'shardi) — bu YETARLI, chunki markaz so'nggi
    CENTROID_BUFFER_SIZE namunaning o'rtachasi, bitta noto'g'ri namunaning
    ta'siri keyingi to'g'ri namunalar bilan tezda suyultiriladi."""
    trimmed = _trim_trailing_silence(wav, sample_rate)
    duration_s = len(trimmed) / sample_rate
    original_duration_s = len(wav) / sample_rate
    if trimmed.size > 0 and duration_s < MIN_TRIMMED_DURATION_S and duration_s < original_duration_s:
        trimmed = wav
        duration_s = original_duration_s
    if duration_s < MIN_TRIMMED_DURATION_S:
        logger.info(
            f"[SPK][CORRECT] audio juda qisqa ({duration_s:.2f}s) — {speaker!r} markazi yangilanmadi"
        )
        return

    with session["lock"]:
        embedding = _embed(session["embedding_model"], trimmed)
        profile = session["voices"].get(speaker)
        if profile is None:
            session["voices"][speaker] = {
                "founding_embedding": embedding,
                "embeddings": collections.deque(maxlen=CENTROID_BUFFER_SIZE),
                "centroid": embedding,
            }
        else:
            profile["embeddings"].append(embedding)
            _recompute_voice_centroid(profile)
    logger.info(f"[SPK][CORRECT] {speaker!r} markazi foydalanuvchi tuzatishi orqali yangilandi")
