import time

from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from app.config import SPEAKER_SIMILARITY_THRESHOLD

# Resemblyzer works internally at 16 kHz. Import the real value if we can so the
# duration maths stays correct if the library ever changes it.
try:
    from resemblyzer.hparams import sampling_rate as RESEMBLYZER_SR
except Exception:  # pragma: no cover - defensive
    RESEMBLYZER_SR = 16000

# Resemblyzer's partial-utterance window is 1.6 s. With less than roughly this
# much *voiced* audio (after silence trimming) the resulting embedding is not
# speaker-discriminative — every clip lands near the model's "generic voice"
# centroid and unrelated speakers score 0.8+ against each other. Below this we
# refuse to register/compare rather than pollute the reference set.
MIN_VOICED_SECONDS = 1.5

# embed_utterance() averages embeddings over 1.6 s windows across the ENTIRE
# clip. The turns coming from live_transcribe.py are the whole 13-19 s session
# (pauses, breaths, room noise, changing delivery) which washes the embedding
# toward a generic centroid and makes different speakers score alike. So we
# embed only the LAST few seconds of voiced audio — the tail of a turn is
# usually continuous speech (the speaker finishing their sentence).
IDENTIFY_WINDOW_SECONDS = 5


_encoder = None
_speakers = {}


def _load_encoder():
    global _encoder
    if _encoder is None:
        # Must appear EXACTLY ONCE per process. Resemblyzer's VoiceEncoder
        # defaults to CPU here (no device arg passed), so first load + every
        # embed is CPU work.
        print("[SPK] Loading Resemblyzer VoiceEncoder...", flush=True)
        t0 = time.perf_counter()
        _encoder = VoiceEncoder()
        print(
            f"[SPK] VoiceEncoder loaded in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

        # preprocess_wav() only trims silence when webrtcvad is importable.
        # Without it, leading/trailing pauses stay in the clip and dilute the
        # embedding — a common reason different speakers score alike.
        try:
            import webrtcvad  # noqa: F401

            print(
                "[SPK] webrtcvad: available — preprocess_wav WILL trim silence",
                flush=True,
            )
        except Exception:
            print(
                "[SPK] webrtcvad: NOT installed — preprocess_wav will NOT trim "
                "silence, so pauses stay in the audio and embeddings are "
                "diluted. `pip install webrtcvad` is strongly recommended.",
                flush=True,
            )
    return _encoder


def warmup(run_inference: bool = True) -> None:
    """Load the encoder now, and optionally embed a dummy clip so the first
    real /identify-speaker call is fast. Does NOT touch the _speakers registry."""
    encoder = _load_encoder()
    if not run_inference:
        return
    try:
        encoder.embed_utterance(np.zeros(int(RESEMBLYZER_SR * 2), dtype=np.float32))
        print("[SPK] warm-up inference done", flush=True)
    except Exception as e:  # pragma: no cover - diagnostics only
        print(f"[SPK] warm-up inference skipped: {e!r}", flush=True)


def reset_speakers():
    global _speakers
    _speakers = {}
    print("[SPK] speaker registry reset", flush=True)


def _load_waveform(audio_path: str):
    """Decode the file to a 16 kHz mono float32 waveform via librosa's
    ffmpeg/audioread backend (which handles webm/opus, unlike libsndfile's
    metadata-only get_duration()). Returns (wav, duration_seconds)."""
    import librosa

    wav, _ = librosa.load(audio_path, sr=RESEMBLYZER_SR, mono=True)
    return wav.astype(np.float32), len(wav) / RESEMBLYZER_SR


def identify_speaker(audio_path: str) -> str:
    encoder = _load_encoder()

    # Decode the file ourselves once so we get a correct raw duration
    # (librosa.get_duration(path=...) returns 0.00s for webm here), then hand
    # the array to preprocess_wav for normalisation + silence trimming.
    t0 = time.perf_counter()
    try:
        raw_wav, raw_s = _load_waveform(audio_path)
        wav = preprocess_wav(raw_wav, source_sr=RESEMBLYZER_SR)
    except Exception as e:
        print(
            f"[SPK] waveform pre-load failed ({e!r}); falling back to path decode",
            flush=True,
        )
        wav = preprocess_wav(audio_path)
        raw_s = None
    prep_s = time.perf_counter() - t0
    voiced_s = len(wav) / RESEMBLYZER_SR

    raw_str = f"{raw_s:.2f}s" if raw_s is not None else "unknown"
    print(
        f"[SPK] audio={audio_path} raw_duration={raw_str} "
        f"voiced_after_preprocess={voiced_s:.2f}s (decode+preprocess={prep_s:.2f}s)",
        flush=True,
    )

    # Guard: too little usable speech to produce a reliable embedding.
    if voiced_s < MIN_VOICED_SECONDS:
        print(
            f"[SPK] WARNING: only {voiced_s:.2f}s of voiced audio "
            f"(need >= {MIN_VOICED_SECONDS:.1f}s). Skipping embedding — "
            f"returning 'unknown' and NOT registering a speaker. Record a "
            f"longer utterance (aim for 3-5s of continuous speech).",
            flush=True,
        )
        return "unknown"

    # Keep only the final IDENTIFY_WINDOW_SECONDS of the voiced waveform.
    # numpy slicing with [-n:] safely returns the whole array if it is shorter.
    window_samples = int(IDENTIFY_WINDOW_SECONDS * RESEMBLYZER_SR)
    before_n = len(wav)
    wav = wav[-window_samples:]
    after_n = len(wav)
    print(
        f"[SPK] identify window: {before_n} -> {after_n} samples "
        f"({before_n / RESEMBLYZER_SR:.2f}s -> {after_n / RESEMBLYZER_SR:.2f}s; "
        f"target = last {IDENTIFY_WINDOW_SECONDS}s)",
        flush=True,
    )

    t1 = time.perf_counter()
    embedding = encoder.embed_utterance(wav)
    embed_s = time.perf_counter() - t1
    emb_norm = float(np.linalg.norm(embedding))
    print(
        f"[SPK] embed_utterance={embed_s:.2f}s embedding_dim={embedding.shape[0]} "
        f"embedding_L2_norm={emb_norm:.4f}",
        flush=True,
    )

    if not _speakers:
        _speakers["A"] = embedding
        print("[SPK] registered first speaker as 'A'", flush=True)
        return "A"

    best_label = None
    best_score = -1.0

    for label, ref_embedding in _speakers.items():
        score = np.dot(embedding, ref_embedding)
        print(f"[SPK] similarity vs '{label}' = {score:.3f}", flush=True)

        if score > best_score:
            best_score = score
            best_label = label

    margin = best_score - SPEAKER_SIMILARITY_THRESHOLD
    print(
        f"[SPK] best={best_label!r} score={best_score:.3f} "
        f"threshold={SPEAKER_SIMILARITY_THRESHOLD} margin={margin:+.3f}",
        flush=True,
    )

    if best_score > SPEAKER_SIMILARITY_THRESHOLD:
        print(f"[SPK] -> same speaker as {best_label!r}", flush=True)
        return best_label

    if "B" not in _speakers:
        _speakers["B"] = embedding
        print("[SPK] -> below threshold; registered new speaker 'B'", flush=True)
        return "B"

    print(
        f"[SPK] -> below threshold but A and B both exist; returning closest "
        f"{best_label!r}",
        flush=True,
    )
    return best_label
