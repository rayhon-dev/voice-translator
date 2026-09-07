import time

import numpy as np
from faster_whisper import WhisperModel
from app.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

# --- anti-hallucination tuning ----------------------------------------------
# All three guards below are independent; a segment/clip only survives if it
# passes every one. Numbers are deliberately equal to faster-whisper's own
# internal defaults where such a default exists, so genuine speech is never
# touched — see the explanation in the accompanying notes.

# Drop a segment if Whisper itself thinks it is probably not speech.
NO_SPEECH_PROB_THRESHOLD = 0.6      # matches faster-whisper's no_speech_threshold
# Drop a segment whose tokens were decoded with low average confidence.
AVG_LOGPROB_THRESHOLD = -1.0        # matches faster-whisper's log_prob_threshold
# Skip Whisper entirely if the whole clip is quieter than this RMS level
# (waveform is float32 in [-1, 1]). Conservative: well below even quiet speech.
RMS_SILENCE_THRESHOLD = 0.003
# Use faster-whisper's built-in Silero VAD to strip silence before decoding.
USE_VAD_FILTER = True

SAMPLE_RATE = 16000

_model = None


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


# vad_filter=True needs onnxruntime (faster-whisper runs Silero VAD via ONNX).
_ONNX_AVAILABLE = _onnxruntime_available()


def get_model():
    global _model
    if _model is None:
        # This line must appear EXACTLY ONCE in the server logs for the whole
        # process lifetime. If you see it on every /transcribe or every
        # WebSocket final-pass, the module-level cache is being defeated.
        print(
            f"[STT] Loading Whisper model... size={WHISPER_MODEL_SIZE!r} "
            f"device={WHISPER_DEVICE!r} compute_type={WHISPER_COMPUTE_TYPE!r}",
            flush=True,
        )
        t0 = time.perf_counter()
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        load_s = time.perf_counter() - t0

        # faster-whisper runs on CTranslate2. This is the most reliable way to
        # confirm CUDA is actually available to the inference backend (it does
        # not necessarily mean this model instance is on the GPU, but if this
        # is 0 then device="cuda" silently fell back to CPU).
        try:
            import ctranslate2

            cuda_count = ctranslate2.get_cuda_device_count()
        except Exception as e:  # pragma: no cover - diagnostics only
            cuda_count = f"<unavailable: {e!r}>"

        print(
            f"[STT] Whisper model loaded in {load_s:.2f}s | "
            f"requested device={WHISPER_DEVICE!r} | "
            f"ctranslate2 CUDA device count={cuda_count}",
            flush=True,
        )
        print(
            f"[STT] vad_filter: "
            + (
                "enabled (onnxruntime present)"
                if (USE_VAD_FILTER and _ONNX_AVAILABLE)
                else "DISABLED — "
                + (
                    "USE_VAD_FILTER is False"
                    if not USE_VAD_FILTER
                    else "onnxruntime not installed; run `pip install onnxruntime`"
                )
            ),
            flush=True,
        )
    return _model


def warmup(run_inference: bool = True) -> None:
    """Load the model now, and optionally run one tiny inference so the CUDA
    kernels / cuDNN autotuning (and the Silero VAD model) load at startup
    instead of on the first user request."""
    model = get_model()
    if not run_inference:
        return
    try:
        # 1 second of silence at 16 kHz — enough to force the first real
        # forward pass through the decoder (and the VAD, if available).
        segments, _ = model.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            language="en",
            vad_filter=USE_VAD_FILTER and _ONNX_AVAILABLE,
            condition_on_previous_text=False,
        )
        for _ in segments:  # generator is lazy; consume it to actually run
            pass
        print("[STT] warm-up inference done", flush=True)
    except Exception as e:  # pragma: no cover - diagnostics only
        print(f"[STT] warm-up inference skipped: {e!r}", flush=True)


def _decode_waveform(audio_path: str):
    """Decode to 16 kHz mono float32. Prefer faster-whisper's own decoder
    (PyAV — already a dependency, so no new install) so what we RMS-check is
    exactly what Whisper would otherwise decode. Returns an ndarray or None."""
    try:
        from faster_whisper.audio import decode_audio

        return decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
    except Exception:
        try:
            import librosa

            wav, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            return wav.astype(np.float32)
        except Exception as e:
            print(
                f"[STT] could not pre-decode audio for RMS check: {e!r}",
                flush=True,
            )
            return None


def transcribe_audio(audio_path: str) -> str:
    model = get_model()

    # --- Guard 1: cheap energy pre-check, no Whisper involved --------------
    audio = _decode_waveform(audio_path)
    rms = None
    if audio is not None and audio.size:
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
        if rms < RMS_SILENCE_THRESHOLD:
            print(
                f"[STT] audio too quiet, skipping transcription "
                f"(rms={rms:.5f} < {RMS_SILENCE_THRESHOLD})",
                flush=True,
            )
            return ""

    # Feed the already-decoded array to Whisper when we have it (avoids a
    # second decode); otherwise let faster-whisper read the file itself.
    source = audio if (audio is not None and audio.size) else audio_path

    # --- Guard 2: faster-whisper's built-in Silero VAD --------------------
    use_vad = USE_VAD_FILTER and _ONNX_AVAILABLE
    t0 = time.perf_counter()
    try:
        segments, info = model.transcribe(
            source,
            language="en",
            vad_filter=use_vad,
            # Do NOT feed the previous (possibly hallucinated) output back in
            # as the decoder prompt — a well-known hallucination-loop trigger
            # for chunked / repeated transcription like the live WS path.
            condition_on_previous_text=False,
        )
        seg_list = list(segments)
    except Exception as e:
        if use_vad:
            print(
                f"[STT] transcribe with vad_filter failed ({e!r}); "
                f"retrying without VAD",
                flush=True,
            )
            use_vad = False
            segments, info = model.transcribe(
                source, language="en", condition_on_previous_text=False
            )
            seg_list = list(segments)
        else:
            raise
    elapsed = time.perf_counter() - t0

    # --- Guard 3: per-segment confidence filtering -----------------------
    kept_parts = []
    dropped = 0
    for seg in seg_list:
        nsp = getattr(seg, "no_speech_prob", 0.0)
        alp = getattr(seg, "avg_logprob", 0.0)
        if nsp > NO_SPEECH_PROB_THRESHOLD or alp < AVG_LOGPROB_THRESHOLD:
            dropped += 1
            print(
                f"[STT] DROP segment [{seg.start:.1f}-{seg.end:.1f}s] "
                f"no_speech_prob={nsp:.2f} avg_logprob={alp:.2f} "
                f"text={seg.text.strip()!r}",
                flush=True,
            )
            continue
        kept_parts.append(seg.text.strip())

    text = "".join(kept_parts).strip()

    audio_s = getattr(info, "duration", None)
    vad_s = getattr(info, "duration_after_vad", None)
    audio_str = f"{audio_s:.1f}s" if audio_s else "?"
    vad_str = f"{vad_s:.1f}s" if vad_s is not None else "?"
    rms_str = f"{rms:.5f}" if rms is not None else "n/a"
    print(
        f"[STT] transcribe={elapsed:.2f}s vad={'on' if use_vad else 'off'} "
        f"audio_dur={audio_str} vad_dur={vad_str} rms={rms_str} "
        f"segments_kept={len(kept_parts)} segments_dropped={dropped} "
        f"-> {len(text)} chars",
        flush=True,
    )

    return text
