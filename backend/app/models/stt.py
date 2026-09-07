import time

import numpy as np
from faster_whisper import WhisperModel
from app.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE


NO_SPEECH_PROB_THRESHOLD = 0.6      
AVG_LOGPROB_THRESHOLD = -1.0     
RMS_SILENCE_THRESHOLD = 0.003
USE_VAD_FILTER = True

SAMPLE_RATE = 16000

_model = None


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  

        return True
    except Exception:
        return False


_ONNX_AVAILABLE = _onnxruntime_available()


def get_model():
    global _model
    if _model is None:
        # Loaded once per process 
        t0 = time.perf_counter()
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print(
            f"[STT] Whisper model loaded in {time.perf_counter() - t0:.2f}s "
            f"(size={WHISPER_MODEL_SIZE!r}, device={WHISPER_DEVICE!r}, "
            f"compute_type={WHISPER_COMPUTE_TYPE!r})",
            flush=True,
        )
        if USE_VAD_FILTER and not _ONNX_AVAILABLE:
            print(
                "[STT] vad_filter requested but onnxruntime is not installed; "
                "running without VAD",
                flush=True,
            )
    return _model


def warmup(run_inference: bool = True) -> None:
    model = get_model()
    if not run_inference:
        return
    try:
        segments, _ = model.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            language="en",
            vad_filter=USE_VAD_FILTER and _ONNX_AVAILABLE,
            condition_on_previous_text=False,
        )
        for _ in segments:  
            pass
        print("[STT] warm-up inference done", flush=True)
    except Exception as e: 
        print(f"[STT] warm-up inference skipped: {e!r}", flush=True)


def _decode_waveform(audio_path: str):
    try:
        from faster_whisper.audio import decode_audio

        return decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
    except Exception:
        try:
            import librosa

            wav, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            return wav.astype(np.float32)
        except Exception as e:
            print(f"[STT] could not pre-decode audio for RMS check: {e!r}", flush=True)
            return None


def transcribe_audio(audio_path: str) -> str:
    model = get_model()

    audio = _decode_waveform(audio_path)
    rms = None
    if audio is not None and audio.size:
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
        if rms < RMS_SILENCE_THRESHOLD:
            print("[STT] audio below RMS threshold — skipping transcription", flush=True)
            return ""

   
    source = audio if (audio is not None and audio.size) else audio_path

    use_vad = USE_VAD_FILTER and _ONNX_AVAILABLE
    t0 = time.perf_counter()
    try:
        segments, _ = model.transcribe(
            source,
            language="en",
            vad_filter=use_vad,
            condition_on_previous_text=False,
        )
        seg_list = list(segments)
    except Exception as e:
        if use_vad:
            print(f"[STT] vad_filter failed ({e!r}); retrying without VAD", flush=True)
            use_vad = False
            segments, _ = model.transcribe(
                source, language="en", condition_on_previous_text=False
            )
            seg_list = list(segments)
        else:
            raise
    elapsed = time.perf_counter() - t0

    kept_parts = []
    dropped = 0
    for seg in seg_list:
        nsp = getattr(seg, "no_speech_prob", 0.0)
        alp = getattr(seg, "avg_logprob", 0.0)
        if nsp > NO_SPEECH_PROB_THRESHOLD or alp < AVG_LOGPROB_THRESHOLD:
            dropped += 1
            continue
        kept_parts.append(seg.text.strip())

    text = "".join(kept_parts).strip()

    msg = f"[STT] {len(text)} chars in {elapsed:.2f}s"
    if dropped:
        msg += f" ({dropped} low-confidence segment(s) filtered)"
    print(msg, flush=True)

    return text
