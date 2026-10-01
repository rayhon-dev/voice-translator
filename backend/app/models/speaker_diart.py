import collections
import threading
import time

import numpy as np
import torch

from app import diart_compat  # noqa: F401 — `diart` import qilinishidan OLDIN kerak
from diart import models as m


_embedding_model = None

RMS_SILENCE_THRESHOLD = 0.003  # stt.py bilan bir xil qiymat
_TRIM_FRAME_MS = 50

# _trim_trailing_silence() 50ms'lik LAHZAVIY kadr RMS'ini tekshiradi — bu
# stt.py'dagi BUTUN KLIP bo'yicha o'rtacha RMS'dan ancha "shovqinli" metrika.
# Real mikrofon audiosida (past gain, tabiiy pauzalar, gap oxiridagi pasayish)
# ko'plab haqiqiy nutq kadrlari ham shu chegaradan past bo'lib qolishi mumkin
# — natijada orqadan-oldinga skaner deyarli butun gapni "jimlik" deb kesib
# tashlashi mumkin edi (real testda kuzatildi: duration=1.98s'dan
# trimmed=0.08s qolgan). Shuning uchun natija shubhali darajada qisqa chiqsa,
# pastdagi floor uni rad etib, original audio'ga qaytadi.
MIN_TRIMMED_DURATION_S = 0.5

# Kosinus o'xshashlik shundan yuqori bo'lsa — bu ovoz mavjud klasterlardan
# biriga tegishli deb hisoblanadi. Real mikrofon bilan nazorat qilingan
# sinovdan (5 marta bir odam / 5 marta boshqa odam, [SPK][SIM] loglari
# qo'lda solishtirilgan) olingan qiymat: bir xil odam ichidagi o'xshashlik
# 0.488-0.660, turli odamlar orasida -0.06-0.133 chiqdi — ikkisi orasida
# 0.13-0.49 bo'sh oraliq bor edi. Eski 0.75 shu oraliqdan ham yuqorida
# bo'lgani uchun bir xil odamning IKKINCHI gapini ham rad etardi. 0.4 — shu
# bo'shliqning o'rtasiga yaqin, pastroq (turli-odamlar) tomondan ehtiyot
# marjasi bilan tanlangan.
SPEAKER_MATCH_THRESHOLD = 0.4

# Shundan QISQA (trim qilingandan keyingi) gap klaster markazini
# YANGILAMAYDI — faqat mavjud markazlarga solishtirib A/B'ga tayinlash uchun
# ishlatiladi. Sabab: qisqa gapning embeddingi beqaror bo'ladi (akademik
# adabiyotda ham markazni faqat "sifatli"/uzunroq namunalar bilan yangilash
# tavsiya etiladi — arxiv.org/pdf/2402.00067dagi rho_update'ga o'xshash
# g'oya). Bitta beqaror namuna markazni "buzib qo'ysa", o'sha odamning
# KEYINGI gapi ham eski-buzilgan markazga yomon mos kelib, xato ravishda
# "boshqa odam" deb aniqlanishi mumkin edi — aynan shu servisning eski
# muammosi shu yerdan kelib chiqqan.
MIN_DURATION_FOR_CENTROID_UPDATE = 2.0  # soniya

# Har bir A/B klaster uchun so'nggi shuncha SIFATLI (yuqoridagi chegaradan
# uzun) embedding saqlanadi; markaz — shularning o'rtachasi (pastga qarang,
# `_recompute_centroid`). Bitta statik/EMA vektor o'rniga sirpanuvchi bufer
# ishlatilishining sababi: bufer eski namunalarni o'zi "unutadi" (N dan
# eskisi chiqib ketadi), shuning ustiga EMA qo'shish ikki marta
# silliqlashtirish bo'lar va yangi sifatli namunaning ta'sirini keraksiz
# kamaytirar edi. Oddiy o'rtacha esa bitta yomon namunaning markazga
# ta'sirini ham ≤1/N bilan chegaralaydi va tushunish/debug qilish osonroq.
CENTROID_BUFFER_SIZE = 5

# Doimiy slot bog'lashda (label_map) qo'shimcha ehtiyot chorasi sifatida
# LANGUAGE_MATCH_MIN_PROB'dan (dialog.py, 0.15) biroz yuqoriroq — bitta
# tasodifiy past-ishonchli shovqin turi butun sessiya uchun noto'g'ri doimiy
# bog'lanib qolmasligi uchun.
BINDING_MIN_PROB = 0.2
REQUIRED_STREAK = 1  # yangi bog'lash uchun kerakli izchil mos kelishlar soni

# Nima uchun diart'ning to'liq SpeakerDiarization pipeline'i emas, faqat
# embedding modeli ishlatiladi: o'sha pipeline haqiqiy UZLUKSIZ audio oqimi
# uchun mo'ljallangan (har chaqiriqda kichik qadam bilan ilgarilab boruvchi,
# bir-biriga ulanib turadigan chunk'lar kutiladi — ichida shunga tayangan
# bufer/aggregatsiya holati bor). Bizda esa har bir "turn" alohida, uzuq-yuluq
# (orasida jimlik bo'lgan) audio bo'lagi va u har doim t=0'dan boshlanadi deb
# yuborilar edi — bu pipeline'ning ichki online-klasterlash holatini
# chalkashtirib, amalda ikkala haqiqiy so'zlovchini ham bitta klasterga
# ("speaker0") birlashtirib yuborishiga olib keldi (real test paytida
# kuzatildi). Shu sababli endi har bir navbat uchun shunchaki BITTA embedding
# olinadi va sessiya davomida o'zimiz oddiy kosinus-o'xshashlik asosidagi
# klasterlashni saqlaymiz (eski Resemblyzer tizimiga konseptual yaqin, faqat
# enrollment bosqichisiz va yaxshiroq — pyannote — embedding modeli bilan).


def _trim_trailing_silence(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    """Audio oxiridagi jim (RMS past) qismni kesib tashlaydi."""
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


def warmup(run_inference: bool = True) -> None:
    t0 = time.perf_counter()
    _load_model()
    print(f"[SPK] pyannote embedding model loaded in {time.perf_counter() - t0:.2f}s", flush=True)
    if not run_inference:
        return
    try:
        session = create_session()
        get_raw_speaker_label(session, np.random.randn(16000).astype(np.float32) * 0.05, 16000)
        print("[SPK] warm-up inference done", flush=True)
    except Exception as e:
        print(f"[SPK] warm-up inference skipped: {e!r}", flush=True)


def create_session():
    return {
        "embedding_model": _load_model(),
        # [{"raw_label": str, "embeddings": deque[np.ndarray] (L2-normalized,
        #   faqat MIN_DURATION_FOR_CENTROID_UPDATE'dan uzun gaplardan),
        #   "centroid": np.ndarray (L2-normalized, embeddings o'rtachasi)}]
        "clusters": [],
        "label_map": {},
        "pending_matches": {},
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


def _recompute_centroid(cluster: dict) -> None:
    mean = np.mean(cluster["embeddings"], axis=0)
    norm = np.linalg.norm(mean)
    cluster["centroid"] = mean / norm if norm > 0 else mean


def get_raw_speaker_label(session, wav: np.ndarray, sample_rate: int) -> tuple[str | None, dict[str, float]]:
    """Ikkinchi qiymat (`sim_debug`) — bu turnning embeddingi HOZIRDA mavjud
    bo'lgan har bir klaster markaziga (A/B'ga bog'langan bo'lsa shu nom bilan,
    aks holda raw_label bilan) qanchalik o'xshashligi, {nom: cosine_sim}
    ko'rinishida. Faqat diagnostika/logging uchun — chaqiruvchi SPEAKER_MATCH_THRESHOLD
    qayerda bo'lishi kerakligini real ma'lumot bilan baholashi uchun qaytariladi."""
    t0 = time.perf_counter()
    trimmed = _trim_trailing_silence(wav, sample_rate)
    trim_ms = (time.perf_counter() - t0) * 1000
    print(
        f"[SPK][TRIM] wav_len={len(wav)} trimmed_len={len(trimmed)} took={trim_ms:.2f}ms",
        flush=True,
    )
    if trimmed.size == 0:
        print(f"[SPK][TRIM] wav_len={len(wav)} fully silent after trailing-silence trim — skipping", flush=True)
        return None, {}

    print(
        f"[SPK][AUDIO] duration={len(wav)/sample_rate:.2f}s "
        f"trimmed_duration={len(trimmed)/sample_rate:.2f}s rms={np.sqrt(np.mean(wav**2)):.4f}",
        flush=True,
    )

    # Trim natijasi gumonli darajada qisqa (floor'dan kam) VA asl audio'dan
    # sezilarli qisqartirilgan bo'lsa — bu haqiqiy trailing silence emas,
    # trim'ning haqiqiy nutqni noto'g'ri "jimlik" deb kesib tashlagani
    # (yuqoridagi MIN_TRIMMED_DURATION_S izohiga qarang). Shunday holatda
    # original (kesilmagan) audio ishlatiladi.
    original_duration_s = len(wav) / sample_rate
    trimmed_duration_s = len(trimmed) / sample_rate
    if trimmed_duration_s < MIN_TRIMMED_DURATION_S and trimmed_duration_s < original_duration_s:
        print(
            f"[SPK][TRIM] trim too aggressive ({original_duration_s:.2f}s -> "
            f"{trimmed_duration_s:.2f}s), using original audio",
            flush=True,
        )
        trimmed = wav

    duration_s = len(trimmed) / sample_rate

    if duration_s < MIN_TRIMMED_DURATION_S:
        print(
            f"[SPK][TRIM] duration={duration_s:.2f}s < {MIN_TRIMMED_DURATION_S}s even after "
            f"fallback — too short for embedding, skipping",
            flush=True,
        )
        return None, {}

    try:
        with session["lock"]:
            embedding = _embed(session["embedding_model"], trimmed)

            best_label = None
            best_sim = -1.0
            sim_debug: dict[str, float] = {}
            for cluster in session["clusters"]:
                sim = float(np.dot(embedding, cluster["centroid"]))
                display_name = session["label_map"].get(cluster["raw_label"], cluster["raw_label"])
                sim_debug[display_name] = sim
                if sim > best_sim:
                    best_sim = sim
                    best_label = cluster["raw_label"]

            if sim_debug:
                sim_str = " ".join(f"vs {name}={sim:.3f}" for name, sim in sim_debug.items())
                print(f"[SPK][SIM] {sim_str}", flush=True)

            if best_label is not None and best_sim >= SPEAKER_MATCH_THRESHOLD:
                if duration_s >= MIN_DURATION_FOR_CENTROID_UPDATE:
                    for cluster in session["clusters"]:
                        if cluster["raw_label"] == best_label:
                            cluster["embeddings"].append(embedding)
                            _recompute_centroid(cluster)
                            break
                    print(
                        f"[SPK][DEBUG] matched existing cluster {best_label!r} (sim={best_sim:.3f}, "
                        f"duration={duration_s:.2f}s) — centroid updated "
                        f"label_map={session['label_map']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[SPK][DEBUG] matched existing cluster {best_label!r} (sim={best_sim:.3f}, "
                        f"duration={duration_s:.2f}s < {MIN_DURATION_FOR_CENTROID_UPDATE}s) — "
                        f"centroid NOT updated (too short) label_map={session['label_map']}",
                        flush=True,
                    )
                return best_label, sim_debug

            if len(session["clusters"]) < 2:
                if duration_s < MIN_DURATION_FOR_CENTROID_UPDATE:
                    # Qisqa/tasodifiy navbat (masalan bitta "Thank you" kabi
                    # shovqin/orqa fon) bilan bo'sh klaster slotini "chiqindi"
                    # namuna bilan abadiy band qilib qo'yib bo'lmaydi — real
                    # testda aynan shu holat kuzatildi: 0.90s'lik tasodifiy
                    # gap ikkinchi slotni egalladi va undan keyin HAQIQIY
                    # ikkinchi spiker hech qanday klasterga yetarlicha
                    # o'xshamay, doimiy "unknown_speaker" bo'lib qoldi. Shu
                    # sababli klaster ochish ham markaz-yangilash bilan BIR
                    # XIL davomiylik talabiga bo'ysundiriladi — qisqa, mos
                    # kelmagan ovoz shunchaki rad etiladi, keyinroq uzunroq
                    # sifatli namuna kelganda klaster to'g'ri ochiladi.
                    print(
                        f"[SPK][DEBUG] potential new cluster rejected (best existing sim={best_sim:.3f}, "
                        f"duration={duration_s:.2f}s < {MIN_DURATION_FOR_CENTROID_UPDATE}s) — "
                        f"too short to found a cluster, treating as unknown_speaker",
                        flush=True,
                    )
                    return None, sim_debug

                new_label = f"speaker{len(session['clusters'])}"
                session["clusters"].append({
                    "raw_label": new_label,
                    "embeddings": collections.deque([embedding], maxlen=CENTROID_BUFFER_SIZE),
                    "centroid": embedding,
                })
                print(
                    f"[SPK][DEBUG] new cluster {new_label!r} created (best existing sim={best_sim:.3f}, "
                    f"duration={duration_s:.2f}s, founding sample) "
                    f"label_map={session['label_map']}",
                    flush=True,
                )
                return new_label, sim_debug

            # Ikkala slot ham band va o'xshashlik chegaradan past — bu ovoz
            # begona (notanish uchinchi odam yoki juda buzilgan namuna) deb
            # rad etiladi. Til BU YERDA UMUMAN tekshirilmaydi — faqat masofa.
            print(
                f"[SPK][DEBUG] weak match (best sim={best_sim:.3f} < {SPEAKER_MATCH_THRESHOLD}) "
                f"and both slots already taken — rejecting as unknown_speaker",
                flush=True,
            )
            return None, sim_debug
    except Exception as e:
        print(f"[SPK][ERROR] embedding/clustering raised {e!r}", flush=True)
        raise


def resolve_speaker(
    session,
    raw_label: str,
    speaker_langs: dict,
    detected_lang: str | None,
    lang_prob: float,
    min_prob: float = 0.5,
) -> tuple[str | None, str | None]:
    """`raw_label` (ovoz-klasterlash natijasi, `get_raw_speaker_label`) ni
    A/B'ga bog'laydi. Til ENDI faqat SUHBAT BOSHIDA, hali hech qaysi A/B'ga
    bog'lanmagan yangi `raw_label` uchun BIR MARTALIK bog'lashda ishlatiladi
    (chunki ovozning o'zi hali hech narsani bilmaydi — birinchi klaster uchun
    markaz yo'q). Bog'langandan KEYIN til bu qarorga umuman aralashmaydi —
    `raw_label` allaqachon `get_raw_speaker_label` ichida ovoz-masofasi
    bo'yicha aniqlangan, shuning uchun bu yerda faqat oddiy lookup qilinadi.
    Sabab: til har safar qayta tekshirilib ovoz-bog'lamani bekor qilishi
    (eski xatti-harakat) — ovoz-embedding endi (duration-gated, ko'p-namunali
    bufer bilan) ancha barqaror bo'lgani uchun endi kerak emas, aksincha bir
    marta to'g'ri bog'langan odamni tilga qarab boshqa slotga "o'g'irlab"
    ketish xavfini tug'dirardi (masalan odam ikkinchi tilga o'tib qolsa)."""

    label_map = session["label_map"]
    pending = session["pending_matches"]
    with session["lock"]:
        if raw_label in label_map:
            bound_speaker = label_map[raw_label]
            print(
                f"[SPK][DEBUG] raw_label={raw_label!r} bound to {bound_speaker!r} "
                f"(ovoz orqali, til tekshirilmadi)",
                flush=True,
            )
            return bound_speaker, None

        # --- bootstrap: hali bog'lanmagan raw_label uchun BIR MARTALIK til orqali bog'lash ---
        if detected_lang is None or lang_prob < min_prob:
            print(
                f"[SPK][DEBUG] raw_label={raw_label!r} (hali bog'lanmagan) til "
                f"unresolved/low-confidence (detected={detected_lang!r} prob={lang_prob:.2f} "
                f"< {min_prob}) — boshlang'ich bog'lash uchun yetarli emas, navbat tashlab yuborildi",
                flush=True,
            )
            pending.pop(raw_label, None)
            return None, "low_confidence_language"

        assigned = set(label_map.values())
        open_slots = [s for s in ("A", "B") if s not in assigned]

        if not open_slots:
            # Amalda yuz bermasligi kerak: get_raw_speaker_label 2 tadan
            # ortiq klaster yaratmaydi va notanish ovozni allaqachon None
            # qaytarib rad etadi. Himoya sifatida saqlanadi.
            print(f"[SPK][DEBUG] raw_label={raw_label!r} not mappable, A/B slots full", flush=True)
            return None, "unknown_speaker"

        candidates = [s for s in open_slots if speaker_langs.get(s) == detected_lang]
        if len(candidates) != 1:
            print(
                f"[SPK][DEBUG] raw_label={raw_label!r} detected_lang={detected_lang!r} "
                f"does not uniquely match an open slot (open={open_slots}, "
                f"speaker_langs={speaker_langs}) — pending streak reset",
                flush=True,
            )
            pending.pop(raw_label, None)
            return None, "language_mismatch"

        candidate_speaker = candidates[0]

        if lang_prob >= BINDING_MIN_PROB:
            entry = pending.get(raw_label)
            if entry is not None and entry["lang"] == detected_lang:
                entry["count"] += 1
            else:
                entry = {"lang": detected_lang, "count": 1}
                pending[raw_label] = entry

            if entry["count"] >= REQUIRED_STREAK:
                label_map[raw_label] = candidate_speaker
                del pending[raw_label]
                print(
                    f"[SPK][DEBUG] raw_label={raw_label!r} bound to {candidate_speaker!r} "
                    f"via detected_lang={detected_lang!r} after streak={REQUIRED_STREAK}",
                    flush=True,
                )
            else:
                print(
                    f"[SPK][DEBUG] raw_label={raw_label!r} pending bind to {candidate_speaker!r} "
                    f"(streak={entry['count']}/{REQUIRED_STREAK}, prob={lang_prob:.2f}) "
                    f"— turn still processed, slot not locked yet",
                    flush=True,
                )
        else:
            print(
                f"[SPK][DEBUG] raw_label={raw_label!r} prob={lang_prob:.2f} below "
                f"BINDING_MIN_PROB={BINDING_MIN_PROB} — pending streak reset, turn still processed",
                flush=True,
            )
            pending.pop(raw_label, None)

        return candidate_speaker, None
