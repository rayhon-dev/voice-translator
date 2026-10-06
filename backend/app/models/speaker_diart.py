import collections
import threading
import time

import numpy as np
import torch

from app import diart_compat  
from diart import models as m


_embedding_model = None

RMS_SILENCE_THRESHOLD = 0.003  # stt.py bilan bir xil qiymat
_TRIM_FRAME_MS = 50

# _trim_trailing_silence() 50ms'lik LAHZAVIY kadr RMS'ini tekshiradi
MIN_TRIMMED_DURATION_S = 0.5

# Kosinus o'xshashlik shundan yuqori bo'lsa — bu ovoz mavjud klasterlardan
# biriga tegishli deb hisoblanadi.
SPEAKER_MATCH_THRESHOLD = 0.4


MIN_DURATION_FOR_CENTROID_UPDATE = 2.0  # soniya

CENTROID_BUFFER_SIZE = 5


BINDING_MIN_PROB = 0.2
REQUIRED_STREAK = 1  # yangi bog'lash uchun kerakli izchil mos kelishlar soni



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

        assigned = set(label_map.values())
        open_slots = [s for s in ("A", "B") if s not in assigned]

        if not open_slots:
            print(f"[SPK][DEBUG] raw_label={raw_label!r} not mappable, A/B slots full", flush=True)
            return None, "unknown_speaker"

        if len(open_slots) == 1:
            candidate_speaker = open_slots[0]
            label_map[raw_label] = candidate_speaker
            pending.pop(raw_label, None)
            print(
                f"[SPK][DEBUG] raw_label={raw_label!r} bound to {candidate_speaker!r} "
                f"(yagona bo'sh slot, til tekshirilmadi)",
                flush=True,
            )
            return candidate_speaker, None

        if detected_lang is None or lang_prob < min_prob:
            print(
                f"[SPK][DEBUG] raw_label={raw_label!r} (hali bog'lanmagan) til "
                f"unresolved/low-confidence (detected={detected_lang!r} prob={lang_prob:.2f} "
                f"< {min_prob}) — boshlang'ich bog'lash uchun yetarli emas, navbat tashlab yuborildi",
                flush=True,
            )
            pending.pop(raw_label, None)
            return None, "low_confidence_language"

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
