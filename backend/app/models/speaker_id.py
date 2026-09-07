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
.
MIN_VOICED_SECONDS = 1.5


IDENTIFY_WINDOW_SECONDS = 5


_encoder = None
_speakers = {}


def _load_encoder():
    global _encoder
    if _encoder is None:
        # Loaded once per process. VoiceEncoder runs on CPU here (no device arg).
        t0 = time.perf_counter()
        _encoder = VoiceEncoder()
        print(
            f"[SPK] Resemblyzer VoiceEncoder loaded in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

        # preprocess_wav() only trims silence when webrtcvad is importable;
        # without it, pauses stay in the clip and dilute embeddings.
        try:
            import webrtcvad  
        except Exception:
            print(
                "[SPK] webrtcvad not installed; preprocess_wav will not trim "
                "silence and speaker embeddings will be less reliable",
                flush=True,
            )
    return _encoder


def warmup(run_inference: bool = True) -> None:
    encoder = _load_encoder()
    if not run_inference:
        return
    try:
        encoder.embed_utterance(np.zeros(int(RESEMBLYZER_SR * 2), dtype=np.float32))
        print("[SPK] warm-up inference done", flush=True)
    except Exception as e:  
        print(f"[SPK] warm-up inference skipped: {e!r}", flush=True)


def reset_speakers():
    global _speakers
    _speakers = {}
    print("[SPK] speaker registry reset", flush=True)


def _load_waveform(audio_path: str) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(audio_path, sr=RESEMBLYZER_SR, mono=True)
    return wav.astype(np.float32)


def identify_speaker(audio_path: str) -> str:
    encoder = _load_encoder()
    t0 = time.perf_counter()

    # Decode ourselves (librosa handles webm), then normalise + trim silence.
    try:
        wav = preprocess_wav(_load_waveform(audio_path), source_sr=RESEMBLYZER_SR)
    except Exception as e:
        print(f"[SPK] waveform decode failed ({e!r}); using path decode", flush=True)
        wav = preprocess_wav(audio_path)
    voiced_s = len(wav) / RESEMBLYZER_SR

    if voiced_s < MIN_VOICED_SECONDS:
        print(
            f"[SPK] only {voiced_s:.1f}s of voiced audio "
            f"(< {MIN_VOICED_SECONDS}s); returning 'unknown'",
            flush=True,
        )
        return "unknown"

    
    window_samples = int(IDENTIFY_WINDOW_SECONDS * RESEMBLYZER_SR)
    embedding = encoder.embed_utterance(wav[-window_samples:])

    if not _speakers:
        _speakers["A"] = embedding
        print(f"[SPK] registered first speaker 'A' ({time.perf_counter() - t0:.2f}s)", flush=True)
        return "A"

    best_label = None
    best_score = -1.0
    for label, ref_embedding in _speakers.items():
        score = np.dot(embedding, ref_embedding)
        if score > best_score:
            best_score = score
            best_label = label

    if best_score > SPEAKER_SIMILARITY_THRESHOLD:
        print(
            f"[SPK] speaker={best_label!r} score={best_score:.3f} "
            f"(> {SPEAKER_SIMILARITY_THRESHOLD}, {time.perf_counter() - t0:.2f}s)",
            flush=True,
        )
        return best_label

    if "B" not in _speakers:
        _speakers["B"] = embedding
        print(
            f"[SPK] registered new speaker 'B' "
            f"(best score {best_score:.3f} <= {SPEAKER_SIMILARITY_THRESHOLD})",
            flush=True,
        )
        return "B"

    print(
        f"[SPK] speaker={best_label!r} score={best_score:.3f} "
        f"(below threshold, A and B already registered)",
        flush=True,
    )
    return best_label
