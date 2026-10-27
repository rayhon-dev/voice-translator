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

SPK_EMBEDDING_MODEL = os.environ.get("SPK_EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM")
SPK_EMBEDDING_FALLBACK_MODEL = os.environ.get("SPK_EMBEDDING_FALLBACK_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)

SPK_DEVICE = os.environ.get("SPK_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

_embedding_backend: str | None = None  # "pyannote" | "speechbrain"
_embedding_model = None  # pyannote Inference OBYEKTI yoki speechbrain EncoderClassifier
_embedding_model_name: str | None = None
_embedding_load_lock = threading.Lock()


SPK_ACCEPT = float(os.environ.get("SPK_ACCEPT", "0.55"))

SPK_MIN_MARGIN = float(os.environ.get("SPK_MIN_MARGIN", "0.08"))
SPK_NEW = float(os.environ.get("SPK_NEW", "0.40"))
SPK_MIN_ENROLL_SPEECH_S = float(os.environ.get("SPK_MIN_ENROLL_SPEECH_S", "2.0"))
# Bootstrap can use a confident short answer without relaxing later enrollment.
SPK_BOOTSTRAP_MIN_SPEECH_S = float(os.environ.get("SPK_BOOTSTRAP_MIN_SPEECH_S", "1.5"))
SPK_NEW_IMMEDIATE_PAIR_MASS = float(os.environ.get("SPK_NEW_IMMEDIATE_PAIR_MASS", "0.95"))
SPK_NEW_IMMEDIATE_NEAR_RATIO = float(os.environ.get("SPK_NEW_IMMEDIATE_NEAR_RATIO", "0.75"))
SPK_MAX_PROTOTYPES = int(os.environ.get("SPK_MAX_PROTOTYPES", "10"))
SPK_SHORT_TURN_S = float(os.environ.get("SPK_SHORT_TURN_S", "1.5"))

SPK_DUPLICATE_PROTOTYPE_SIM = 0.95

SPK_ENROLL_MIN_SIM = float(os.environ.get("SPK_ENROLL_MIN_SIM", "0.65"))

SPK_BOOTSTRAP_PROTOTYPES = int(os.environ.get("SPK_BOOTSTRAP_PROTOTYPES", "3"))
SPK_BOOTSTRAP_ACCEPT = float(os.environ.get("SPK_BOOTSTRAP_ACCEPT", "0.45"))
SPK_SHORT_CONTINUE_SIM = float(os.environ.get("SPK_SHORT_CONTINUE_SIM", "0.30"))
SPK_SHORT_CONTINUE_MAX_SPEECH_S = float(os.environ.get("SPK_SHORT_CONTINUE_MAX_SPEECH_S", "3.0"))

SPK_MERGE_SIM = float(os.environ.get("SPK_MERGE_SIM", "0.65"))

SPK_ENROLL_MIN_PURITY = float(os.environ.get("SPK_ENROLL_MIN_PURITY", "0.6"))

SPK_NEW_MIN_NEAR_FIELD_RATIO = float(os.environ.get("SPK_NEW_MIN_NEAR_FIELD_RATIO", "0.5"))
SPK_NEW_CONFIRM_TURNS = max(2, int(os.environ.get("SPK_NEW_CONFIRM_TURNS", "2")))
SPK_NEW_MIN_PAIR_MASS = float(os.environ.get("SPK_NEW_MIN_PAIR_MASS", "0.90"))


def _new_speaker_evidence(purity: float, near_field_ratio: float | None, pair_mass: float | None):
    reasons = []
    if purity < SPK_ENROLL_MIN_PURITY:
        reasons.append(f"purity={purity:.3f}<{SPK_ENROLL_MIN_PURITY}")
    if near_field_ratio is not None and near_field_ratio < SPK_NEW_MIN_NEAR_FIELD_RATIO:
        reasons.append(f"near_field_ratio={near_field_ratio:.3f}<{SPK_NEW_MIN_NEAR_FIELD_RATIO}")
    if pair_mass is not None and pair_mass < SPK_NEW_MIN_PAIR_MASS:
        reasons.append(f"pair_mass={pair_mass:.3f}<{SPK_NEW_MIN_PAIR_MASS}")
    return (not reasons), reasons

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
        "speakers": {},
        "next_id": 1,
        "last_speaker": None,
        "unknown_buffer": [],
        "lock": threading.Lock(),
    }


def _embed(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    model = _load_model()
    waveform = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).reshape(1, -1)
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
    return float(np.dot(a, b))


def _speaker_best_sim(profile: dict, embedding: np.ndarray) -> float:
    if not profile["prototypes"]:
        return -1.0
    return max(_cosine_sim(embedding, p["embedding"]) for p in profile["prototypes"])


def _add_prototype(profile: dict, embedding: np.ndarray, seq: int) -> None:
    profile["prototypes"].append({"seq": seq, "embedding": embedding})
    if len(profile["prototypes"]) > SPK_MAX_PROTOTYPES:
        del profile["prototypes"][1]


def _pop_prototypes_by_seqs(profile: dict, seqs) -> list[dict]:
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
    if embedding is None:
        return None, -1.0, None
    with session["lock"]:
        return _best_match_with_margin(session, embedding)


def _process_unknown_buffer(session) -> list[tuple[int, int]]:
    _trim_unknown_buffer(session)
    buf = session["unknown_buffer"]
    reassigned: list[tuple[int, int]] = []

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
            buffer_reassigned: list[tuple[int, int]] = []

            if embedding is None:
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
                        logger.info(
                            f"[SPK][ENROLL] skipped sim={best_sim:.3f} < {SPK_ENROLL_MIN_SIM} "
                            f"speaker={best_id}"
                        )

            elif best_sim >= SPK_ACCEPT:
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
                new_id = session["next_id"]
                session["next_id"] += 1
                speakers[new_id] = {"prototypes": [{"seq": seq, "embedding": embedding}]}
                decision = "new"
                result_speaker = new_id

            else:
                decision = "unknown"
                result_speaker = None
                uncertain = True
                if speech_s >= SPK_MIN_ENROLL_SPEECH_S:
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
                        decision = "buffer_assigned"
                        result_speaker = r_speaker
                        uncertain = False

            if decision != "unknown" and session["unknown_buffer"]:
                buffer_reassigned.extend(_process_unknown_buffer(session))

            merges = _check_and_merge(session)
            result_speaker = _apply_remap(result_speaker, merges)
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
                "reassigned": reassigned,
            }
    except Exception as e:
        logger.error(f"[SPK][ERROR] identify_speaker raised {e!r}")
        raise


def has_known_speakers(session) -> bool:
    with session["lock"]:
        return len(session["speakers"]) > 0


def apply_speaker_correction(
    session, wav: np.ndarray, sample_rate: int, seqs: list[int], old_speaker, target_speaker
) -> dict:
    trimmed = _trim_trailing_silence(wav, sample_rate)
    duration_s = len(trimmed) / sample_rate
    original_duration_s = len(wav) / sample_rate
    if trimmed.size > 0 and duration_s < MIN_TRIMMED_DURATION_S and duration_s < original_duration_s:
        trimmed = wav
        duration_s = original_duration_s

    primary_seq = seqs[0]

    with session["lock"]:
        removed_speaker = None

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

        chain_merges = _check_and_merge(session)

    final_into = _apply_remap(into_speaker, chain_merges)
    logger.info(f"[SPK][MERGE] reason=user from={from_speaker} into={final_into}")
    return {"merged": True, "merges": [(from_speaker, into_speaker, 1.0)] + chain_merges}
