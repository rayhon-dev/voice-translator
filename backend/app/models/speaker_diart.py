import threading
import time

import numpy as np
import torch

from app import diart_compat  # noqa: F401 — `diart` import qilinishidan OLDIN kerak
from diart import models as m


_embedding_model = None

RMS_SILENCE_THRESHOLD = 0.003  # stt.py bilan bir xil qiymat
_TRIM_FRAME_MS = 50

# Kosinus o'xshashlik shundan yuqori bo'lsa — bu ovoz mavjud klasterlardan
# biriga tegishli deb hisoblanadi. VAQTINCHALIK qiymat: real inson ovozi bilan
# hali alohida kalibrlanmagan (bu servis avvalgi diart streaming pipeline'idan
# nearest-centroid klasterlashga o'tkazilgach yozilgan — quyidagi "Nima uchun"
# izohiga qarang).
SPEAKER_MATCH_THRESHOLD = 0.75
CENTROID_UPDATE_RATE = 0.3  # mos kelgan klaster markazini yangi embedding tomon EMA bilan siljitish tezligi

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
        "clusters": [],  # [{"raw_label": str, "centroid": np.ndarray (L2-normalized)}]
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


def get_raw_speaker_label(session, wav: np.ndarray, sample_rate: int) -> str | None:
    t0 = time.perf_counter()
    trimmed = _trim_trailing_silence(wav, sample_rate)
    trim_ms = (time.perf_counter() - t0) * 1000
    print(
        f"[SPK][TRIM] wav_len={len(wav)} trimmed_len={len(trimmed)} took={trim_ms:.2f}ms",
        flush=True,
    )
    if trimmed.size == 0:
        print(f"[SPK][TRIM] wav_len={len(wav)} fully silent after trailing-silence trim — skipping", flush=True)
        return None

    print(
        f"[SPK][AUDIO] duration={len(wav)/sample_rate:.2f}s "
        f"trimmed_duration={len(trimmed)/sample_rate:.2f}s rms={np.sqrt(np.mean(wav**2)):.4f}",
        flush=True,
    )

    try:
        with session["lock"]:
            embedding = _embed(session["embedding_model"], trimmed)

            best_label = None
            best_sim = -1.0
            for cluster in session["clusters"]:
                sim = float(np.dot(embedding, cluster["centroid"]))
                if sim > best_sim:
                    best_sim = sim
                    best_label = cluster["raw_label"]

            if best_label is not None and best_sim >= SPEAKER_MATCH_THRESHOLD:
                for cluster in session["clusters"]:
                    if cluster["raw_label"] == best_label:
                        updated = (1 - CENTROID_UPDATE_RATE) * cluster["centroid"] + CENTROID_UPDATE_RATE * embedding
                        norm = np.linalg.norm(updated)
                        cluster["centroid"] = updated / norm if norm > 0 else updated
                        break
                print(
                    f"[SPK][DEBUG] matched existing cluster {best_label!r} (sim={best_sim:.3f}) "
                    f"label_map={session['label_map']}",
                    flush=True,
                )
                return best_label

            if len(session["clusters"]) < 2:
                new_label = f"speaker{len(session['clusters'])}"
                session["clusters"].append({"raw_label": new_label, "centroid": embedding})
                print(
                    f"[SPK][DEBUG] new cluster {new_label!r} created (best existing sim={best_sim:.3f}) "
                    f"label_map={session['label_map']}",
                    flush=True,
                )
                return new_label

            # Bu dialog aynan IKKI kishilik (A va B oldindan tanlangan) — shuning
            # uchun "notanish uchinchi odam" degan holat yo'q. Ikkala slot ham
            # band bo'lganda o'xshashlik chegaradan past bo'lsa ham (masalan
            # qisqa/shovqinli navbatda ovoz tembri biroz siljib qolganda), eng
            # yaqin klasterga baribir bog'laymiz — aks holda haqiqiy A/B'ning
            # keyingi har bir navbati asossiz "unknown_speaker" deb tashlab
            # yuborilardi (real testda aniq shu kuzatildi: har ikki spikerning
            # ham FAQAT birinchi gapi o'tib, qolganlari yo'qolgan edi). Markaz
            # (centroid) esa past ishonchli namuna bilan YANGILANMAYDI — aks
            # holda bitta yomon namuna klasterni asta-sekin "buzib" yuborishi
            # mumkin edi.
            print(
                f"[SPK][DEBUG] weak match (best sim={best_sim:.3f} < {SPEAKER_MATCH_THRESHOLD}) "
                f"but both slots already taken — forcing nearest cluster {best_label!r} "
                f"(centroid not updated)",
                flush=True,
            )
            return best_label
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
    """Til (restricted candidate-detection orqali aniqlangan) endi HAR BIR
    navbat uchun (allaqachon bog'langan spikerlar uchun ham) tekshiriladi va
    asosiy signal hisoblanadi — ovoz-klasterlash (`raw_label`) esa yordamchi.
    Sabab: real foydalanishda ovoz-embedding qisqa navbatlarda ba'zan xato
    klasterga bog'lab qo'yishi (yoki umuman past ishonch bilan mos kelishi)
    kuzatildi, tilni tekshirmasdan unga ko'r-ko'rona ishonish esa boshqa
    odamning nutqini noto'g'ri tilda "tarjima qilib" chiqarib yuborardi."""

    label_map = session["label_map"]
    pending = session["pending_matches"]
    with session["lock"]:
        if detected_lang is None or lang_prob < min_prob:
            print(
                f"[SPK][DEBUG] raw_label={raw_label!r} language unresolved/low-confidence "
                f"(detected={detected_lang!r} prob={lang_prob:.2f} < {min_prob}) — "
                f"shovqin/notanish nutq deb hisoblanmoqda, navbat tashlab yuborildi",
                flush=True,
            )
            pending.pop(raw_label, None)
            return None, "low_confidence_language"

        lang_matches = [s for s in ("A", "B") if speaker_langs.get(s) == detected_lang]

        if raw_label in label_map:
            bound_speaker = label_map[raw_label]
            if len(lang_matches) == 1 and lang_matches[0] != bound_speaker:
                print(
                    f"[SPK][DEBUG] raw_label={raw_label!r} ovoz-klaster {bound_speaker!r}ga "
                    f"bog'langan edi, lekin detected_lang={detected_lang!r} (prob={lang_prob:.2f}) "
                    f"aslida {lang_matches[0]!r}ga mos keladi — tilga ishonib {lang_matches[0]!r} "
                    f"deb qabul qilinmoqda (ovoz-embedding qisqa navbatda xato bo'lishi mumkin)",
                    flush=True,
                )
                return lang_matches[0], None
            print(
                f"[SPK][DEBUG] raw_label={raw_label!r} bound to {bound_speaker!r}, "
                f"detected_lang={detected_lang!r} (prob={lang_prob:.2f}) mos keladi",
                flush=True,
            )
            return bound_speaker, None

        assigned = set(label_map.values())
        open_slots = [s for s in ("A", "B") if s not in assigned]

        if not open_slots:
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
