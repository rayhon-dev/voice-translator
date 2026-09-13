import time

from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from app.config import UNKNOWN_SPEAKER_THRESHOLD

# Resemblyzer works internally at 16 kHz. Import the real value if we can so the
# duration maths stays correct if the library ever changes it.
try:
    from resemblyzer.hparams import sampling_rate as RESEMBLYZER_SR
except Exception: 
    RESEMBLYZER_SR = 16000

MIN_VOICED_SECONDS = 1.5


IDENTIFY_WINDOW_SECONDS = 5


_encoder = None
_speakers = {}


def _load_encoder():
    global _encoder
    if _encoder is None:
        t0 = time.perf_counter()
        _encoder = VoiceEncoder()
        print(
            f"[SPK] Resemblyzer VoiceEncoder loaded in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

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


def _prepare_wav(audio_path: str) -> np.ndarray:
    try:
        return preprocess_wav(_load_waveform(audio_path), source_sr=RESEMBLYZER_SR)
    except Exception as e:
        print(f"[SPK] waveform decode failed ({e!r}); using path decode", flush=True)
        return preprocess_wav(audio_path)


def enroll_speaker(audio_path: str, label: str) -> bool:
    encoder = _load_encoder()

    wav = _prepare_wav(audio_path)
    voiced_s = len(wav) / RESEMBLYZER_SR

    if voiced_s < MIN_VOICED_SECONDS:
        print(
            f"[SPK] enroll {label!r} failed: only {voiced_s:.1f}s voiced audio "
            f"(< {MIN_VOICED_SECONDS}s)",
            flush=True,
        )
        return False

    window_samples = int(IDENTIFY_WINDOW_SECONDS * RESEMBLYZER_SR)
    embedding = encoder.embed_utterance(wav[-window_samples:])

    _speakers[label] = embedding
    print(f"[SPK] enrolled speaker {label!r} ({voiced_s:.1f}s voiced audio)", flush=True)
    return True


def identify_from_enrolled(audio_path: str) -> str:
    if len(_speakers) < 2:
        print(
            "[SPK] identify_from_enrolled called before both speakers enrolled",
            flush=True,
        )
        return "unknown"

    encoder = _load_encoder()

    wav = _prepare_wav(audio_path)
    voiced_s = len(wav) / RESEMBLYZER_SR

    if voiced_s < MIN_VOICED_SECONDS:
        print(
            f"[SPK] turn only {voiced_s:.1f}s of voiced audio "
            f"(< {MIN_VOICED_SECONDS}s); returning 'unknown'",
            flush=True,
        )
        return "unknown"

    window_samples = int(IDENTIFY_WINDOW_SECONDS * RESEMBLYZER_SR)
    embedding = encoder.embed_utterance(wav[-window_samples:])

    best_label = None
    best_score = -1.0
    for label, ref_embedding in _speakers.items():
        score = np.dot(embedding, ref_embedding)
        if score > best_score:
            best_score = score
            best_label = label

    if best_score < UNKNOWN_SPEAKER_THRESHOLD:
        print(
            f"[SPK] turn does not match A or B (best={best_score:.3f} "
            f"< {UNKNOWN_SPEAKER_THRESHOLD}) -> unknown",
            flush=True,
        )
        return "unknown"

    print(f"[SPK] turn matched to {best_label!r} (score={best_score:.3f})", flush=True)
    return best_label