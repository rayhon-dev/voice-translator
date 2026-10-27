import difflib
import os
import time
import traceback

import numpy as np
import requests
import soundfile as sf
import torch  # noqa: F401 — ctranslate2 uchun libcublas.so.12'ni oldindan yuklaydi
from faster_whisper import WhisperModel
from faster_whisper.vad import get_speech_timestamps
from app.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    STT_PROVIDER,
    ELEVENLABS_API_KEY,
    SUPPORTED_LANGUAGES,
)
import logging

logger = logging.getLogger(__name__)


NO_SPEECH_PROB_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0
RMS_SILENCE_THRESHOLD = 0.003
USE_VAD_FILTER = True

MIN_PAIR_MASS = float(os.environ.get("MIN_PAIR_MASS", "0.45"))

MIN_LANG_PROB = float(os.environ.get("MIN_LANG_PROB", "0.50"))

SHORT_TURN_STRICT_S = float(os.environ.get("SHORT_TURN_STRICT_S", "3.0"))
MIN_PAIR_MASS_SHORT = float(os.environ.get("MIN_PAIR_MASS_SHORT", "0.75"))

FOREIGN_RIVAL_MIN_PROB = float(os.environ.get("FOREIGN_RIVAL_MIN_PROB", "0.15"))
FOREIGN_RIVAL_RATIO = float(os.environ.get("FOREIGN_RIVAL_RATIO", "0.4"))
LANG_FAMILY = {
    "ru": {"uk", "be", "bg", "sr", "hr", "bs", "mk", "sl", "pl", "cs", "sk", "lv", "lt"},
}

SHORT_DOMINANT_MIN_PAIR_MASS = float(os.environ.get("SHORT_DOMINANT_MIN_PAIR_MASS", "0.6"))
SHORT_DOMINANCE_RATIO = float(os.environ.get("SHORT_DOMINANCE_RATIO", "5.0"))

AMBIGUOUS_LANG_PROB = float(os.environ.get("AMBIGUOUS_LANG_PROB", "0.7"))

DECODE_RESCUE_MIN_PAIR_MASS = float(os.environ.get("DECODE_RESCUE_MIN_PAIR_MASS", "0.4"))
DECODE_RESCUE_MIN_SCORE = float(os.environ.get("DECODE_RESCUE_MIN_SCORE", "-0.6"))
DECODE_RESCUE_MARGIN = float(os.environ.get("DECODE_RESCUE_MARGIN", "0.35"))


CONTINUATION_MIN_PAIR_MASS = float(os.environ.get("CONTINUATION_MIN_PAIR_MASS", "0.3"))

SEG_LANG_CHECK_BELOW = float(os.environ.get("SEG_LANG_CHECK_BELOW", "0.97"))
SEG_LANG_MIN_S = float(os.environ.get("SEG_LANG_MIN_S", "1.0"))
SEG_MIN_PAIR_MASS = float(os.environ.get("SEG_MIN_PAIR_MASS", "0.5"))

COMPRESSION_RATIO_THRESHOLD = 2.4

PHANTOM_PHRASES = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to my channel",
    "subtitles by the amara.org community",
    "субтитры создавал dimatorzok",
    "редактор субтитров а.семкин",
    "корректор а.egorova",
    "продолжение следует",
    "thank you",
    "thank you.",
    "thanks for watching.",
    "bye",
    "bye.",
    "you",
    "спасибо",
    "спасибо за внимание",
    "благодарю за внимание",
    "ставьте лайки и подписывайтесь",
]

PHANTOM_PHRASE_FUZZY_THRESHOLD = 0.85


def _is_phantom_phrase(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    return any(
        difflib.SequenceMatcher(None, normalized, phrase).ratio() >= PHANTOM_PHRASE_FUZZY_THRESHOLD
        for phrase in PHANTOM_PHRASES
    )


PHANTOM_SHORT_TURN_S = float(os.environ.get("PHANTOM_SHORT_TURN_S", "3.0"))
PHANTOM_NO_SPEECH_PROB_THRESHOLD = float(os.environ.get("PHANTOM_NO_SPEECH_PROB_THRESHOLD", "0.3"))
PHANTOM_AVG_LOGPROB_THRESHOLD = float(os.environ.get("PHANTOM_AVG_LOGPROB_THRESHOLD", "-0.7"))


LOW_CONF_SHORT_MAX_WORDS = int(os.environ.get("LOW_CONF_SHORT_MAX_WORDS", "2"))
LOW_CONF_SHORT_MAX_S = float(os.environ.get("LOW_CONF_SHORT_MAX_S", "2.5"))
LOW_CONF_SHORT_AVG_LOGPROB = float(os.environ.get("LOW_CONF_SHORT_AVG_LOGPROB", "-0.7"))


def _is_low_conf_short_drop(text: str, duration_s: float, avg_logprob: float) -> bool:
    word_count = len(text.strip().split())
    if word_count == 0 or word_count > LOW_CONF_SHORT_MAX_WORDS:
        return False
    return duration_s < LOW_CONF_SHORT_MAX_S and avg_logprob < LOW_CONF_SHORT_AVG_LOGPROB


def _is_phantom_drop(text: str, duration_s: float, no_speech_prob: float, avg_logprob: float) -> bool:
    if not _is_phantom_phrase(text):
        return False
    return (
        duration_s < PHANTOM_SHORT_TURN_S
        or no_speech_prob > PHANTOM_NO_SPEECH_PROB_THRESHOLD
        or avg_logprob < PHANTOM_AVG_LOGPROB_THRESHOLD
    )

WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))
WHISPER_TEMPERATURE = float(os.environ.get("WHISPER_TEMPERATURE", "0.0"))
WHISPER_CONDITION_ON_PREVIOUS_TEXT = (
    os.environ.get("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false").strip().lower() == "true"
)

WHISPER_INITIAL_PROMPT_ENABLED = (
    os.environ.get("WHISPER_INITIAL_PROMPT_ENABLED", "false").strip().lower() == "true"
)
WHISPER_INITIAL_PROMPTS = {
    "en": os.environ.get(
        "WHISPER_INITIAL_PROMPT_EN",
        "The speaker mainly speaks English but may use some Russian or Uzbek words. "
        "Write those words exactly as spoken, without translating them.",
    ),
    "ru": os.environ.get(
        "WHISPER_INITIAL_PROMPT_RU",
        "Говорящий в основном говорит по-русски, но может использовать отдельные "
        "английские или узбекские слова. Записывай их так, как они произнесены, без перевода.",
    ),
}

REPETITION_PENALTY = 1.3
NO_REPEAT_NGRAM_SIZE = 3

MIN_SPEECH_S = float(os.environ.get("MIN_SPEECH_S", "0.3"))

SEGMENT_REL_RMS = float(os.environ.get("SEGMENT_REL_RMS", "0.3"))

DEBUG_STT = False

SAMPLE_RATE = 16000

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_MODEL_ID = "scribe_v2"
ELEVENLABS_TIMEOUT_S = 30

ELEVENLABS_LANG_CODE_MAP = {
    "eng": "en",
    "rus": "ru",
    "kor": "ko",
}

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
        logger.info(f"[STT] Whisper model loaded in {time.perf_counter() - t0:.2f}s "
            f"(size={WHISPER_MODEL_SIZE!r}, device={WHISPER_DEVICE!r}, "
            f"compute_type={WHISPER_COMPUTE_TYPE!r})")
        if USE_VAD_FILTER and not _ONNX_AVAILABLE:
            logger.info("[STT] vad_filter requested but onnxruntime is not installed; "
                "running without VAD")
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
        logger.info("[STT] warm-up inference done")
    except Exception as e: 
        logger.info(f"[STT] warm-up inference skipped: {e!r}")


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
            logger.info(f"[STT] could not pre-decode audio for RMS check: {e!r}")
            return None


def _decode_waveform_bytes(chunk: bytes):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
        tmp.write(chunk)
        tmp.flush()
        return _decode_waveform(tmp.name)


def _decode_pcm_bytes(chunk: bytes) -> np.ndarray:
    return np.frombuffer(chunk, dtype=np.float32).copy()


def _write_wav(path: str, wav: np.ndarray) -> None:
    sf.write(path, wav, SAMPLE_RATE)


def _run_transcribe(model, source, language: str | None, use_vad: bool, initial_prompt: str | None = None):
    t0 = time.perf_counter()
    try:
        segments, info = model.transcribe(
            source,
            language=language,
            vad_filter=use_vad,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            temperature=WHISPER_TEMPERATURE,
            beam_size=WHISPER_BEAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            initial_prompt=initial_prompt,
        )
        return list(segments), info
    except Exception as e:
        first_attempt_elapsed = time.perf_counter() - t0
        if not use_vad:
            raise
        logger.error(f"[STT][VAD_RETRY] vad_filter failed after {first_attempt_elapsed:.2f}s "
            f"(language={language!r}, exception={e!r}); retrying without VAD\n"
            f"{traceback.format_exc()}")
        t_retry0 = time.perf_counter()
        segments, info = model.transcribe(
            source,
            language=language,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            temperature=WHISPER_TEMPERATURE,
            beam_size=WHISPER_BEAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            initial_prompt=initial_prompt,
        )
        seg_list = list(segments)
        logger.info(f"[STT][VAD_RETRY] retry (no VAD) took {time.perf_counter() - t_retry0:.2f}s "
            f"(first attempt had already spent {first_attempt_elapsed:.2f}s)")
        return seg_list, info


LID_TRIM_FRAME_S = 0.02
LID_TRIM_REL = 0.1
LID_TRIM_FLOOR_RMS = 0.005


def _trim_leading_quiet(audio: np.ndarray) -> np.ndarray:
    frame = int(LID_TRIM_FRAME_S * SAMPLE_RATE)
    n = audio.size // frame
    if n == 0:
        return audio
    rms = np.sqrt(np.mean(np.square(audio[: n * frame].astype(np.float64)).reshape(n, frame), axis=1))
    loud = rms >= max(LID_TRIM_FLOOR_RMS, LID_TRIM_REL * float(np.percentile(rms, 90)))
    if not loud.any():
        return audio
    return audio[int(np.argmax(loud)) * frame:]


def _detect_language_restricted(
    model, audio: np.ndarray, candidates: list[str]
) -> tuple[str, float, dict]:
    features = model.feature_extractor(_trim_leading_quiet(audio))
    segment = features[:, : model.feature_extractor.nb_max_frames]
    encoder_output = model.encode(segment)
    results = model.model.detect_language(encoder_output)[0]
    all_probs = {token[2:-2]: prob for token, prob in results}
    cand_probs = {
        lang: all_probs.get(lang, 0.0) + sum(
            all_probs.get(rel, 0.0) for rel in LANG_FAMILY.get(lang, ()) if rel not in candidates
        )
        for lang in candidates
    }

    if not all_probs:
        diagnostics = {"pair_mass": 0.0, "raw_top_lang": None, "raw_top_prob": 0.0, "top5": []}
        return candidates[0], 1.0 / len(candidates), diagnostics

    sorted_all = sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True)
    raw_top_lang, raw_top_prob = sorted_all[0]
    diagnostics = {
        "pair_mass": sum(cand_probs.values()),
        "raw_top_lang": raw_top_lang,
        "raw_top_prob": raw_top_prob,
        "top5": sorted_all[:5],
    }

    best_lang = max(cand_probs, key=cand_probs.get)
    diagnostics["best_lang"] = best_lang
    diagnostics["best_prob"] = cand_probs[best_lang]
    rivals = {
        lang: p for lang, p in all_probs.items()
        if lang not in candidates and not any(lang in LANG_FAMILY.get(c, ()) for c in candidates)
    }
    diagnostics["rival_lang"] = max(rivals, key=rivals.get) if rivals else None
    diagnostics["rival_prob"] = rivals.get(diagnostics["rival_lang"], 0.0)
    total = diagnostics["pair_mass"]
    if total <= 0:
        return best_lang, 1.0 / len(candidates), diagnostics

    return best_lang, cand_probs[best_lang] / total, diagnostics


def _language_gate(
    lang_diag: dict, candidates: list[str], duration_s: float, continuation_lang: str | None = None
) -> str | None:
    pair_mass = lang_diag["pair_mass"]
    best_prob = lang_diag.get("best_prob", 0.0)
    rival_prob = lang_diag.get("rival_prob", 0.0)

    if (
        continuation_lang is not None
        and lang_diag.get("best_lang") == continuation_lang
        and pair_mass >= CONTINUATION_MIN_PAIR_MASS
    ):
        return None

    is_short = duration_s < SHORT_TURN_STRICT_S
    if pair_mass < (MIN_PAIR_MASS_SHORT if is_short else MIN_PAIR_MASS):
        dominant = (
            is_short
            and pair_mass >= SHORT_DOMINANT_MIN_PAIR_MASS
            and best_prob >= SHORT_DOMINANCE_RATIO * rival_prob
        )
        if not dominant:
            return "low_pair_mass_short" if is_short else "low_pair_mass"

    if rival_prob >= FOREIGN_RIVAL_MIN_PROB and rival_prob >= FOREIGN_RIVAL_RATIO * best_prob:
        return "foreign_rival"
    return None


def _decode_score(seg_list) -> float:
    total, weighted = 0.0, 0.0
    for seg in seg_list:
        dur = max(0.01, float(getattr(seg, "end", 0.0)) - float(getattr(seg, "start", 0.0)))
        total += dur
        weighted += dur * float(getattr(seg, "avg_logprob", -10.0))
    return weighted / total if total else -10.0


def _segment_language_filter(
    model, audio: np.ndarray, seg_list, candidates: list[str]
) -> tuple[list[bool], list[dict | None]]:
    foreign = [False] * len(seg_list)
    diagnostics: list[dict | None] = [None] * len(seg_list)
    for i, seg in enumerate(seg_list):
        seg_start_s = getattr(seg, "start", None)
        seg_end_s = getattr(seg, "end", None)
        if seg_start_s is None or seg_end_s is None:
            continue
        if seg_end_s - seg_start_s < SEG_LANG_MIN_S:
            continue
        start_idx = max(0, int(seg_start_s * SAMPLE_RATE))
        end_idx = min(audio.size, int(seg_end_s * SAMPLE_RATE))
        if end_idx <= start_idx:
            continue
        seg_audio = audio[start_idx:end_idx]
        _, _, seg_diag = _detect_language_restricted(model, seg_audio, candidates)
        diagnostics[i] = seg_diag
        if seg_diag["pair_mass"] < SEG_MIN_PAIR_MASS:
            foreign[i] = True
    return foreign, diagnostics


def _fast_vad_speech_duration(audio: np.ndarray) -> float:
    try:
        segments = get_speech_timestamps(audio)
    except Exception as e:
        logger.info(f"[STT][VAD] get_speech_timestamps failed: {e!r} — tekshiruv o'tkazib yuborildi")
        return float("inf")
    return sum(seg["end"] - seg["start"] for seg in segments) / SAMPLE_RATE


SPEECH_RMS_FRAME_MS = 50
SPEECH_RMS_PERCENTILE = 0.9


def compute_speech_rms(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float | None:
    if audio is None or audio.size == 0:
        return None
    try:
        spans = get_speech_timestamps(audio)
    except Exception as e:
        logger.info(f"[STT][VAD] compute_speech_rms: get_speech_timestamps failed: {e!r}")
        spans = []
    if not spans:
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    frame_len = max(1, int(sample_rate * SPEECH_RMS_FRAME_MS / 1000))
    frame_rms: list[float] = []
    for seg in spans:
        start, end = seg["start"], seg["end"]
        for f_start in range(start, end, frame_len):
            f_end = min(f_start + frame_len, end)
            if f_end <= f_start:
                continue
            frame = audio[f_start:f_end]
            frame_rms.append(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))))
    if not frame_rms:
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    frame_rms.sort()
    idx = min(len(frame_rms) - 1, int(len(frame_rms) * SPEECH_RMS_PERCENTILE))
    return frame_rms[idx]


def transcribe_audio(
    audio_path: str,
    language: str | None = None,
    candidate_languages: list[str] | None = None,
    continuation_lang: str | None = None,
) -> tuple[str, str | None, float, bool, dict]:
    if _ONNX_AVAILABLE:
        audio_for_vad = _decode_waveform(audio_path)
        if audio_for_vad is not None and audio_for_vad.size:
            t_vad0 = time.perf_counter()
            speech_s = _fast_vad_speech_duration(audio_for_vad)
            if speech_s < MIN_SPEECH_S:
                logger.info(
                    f"[STT][VAD] speech={speech_s:.2f}s — skipped in "
                    f"{(time.perf_counter() - t_vad0) * 1000:.0f}ms"
                )
                return "", None, 0.0, False, {"seg_lang_check": False, "seg_lang_check_ms": 0.0}

    if STT_PROVIDER == "elevenlabs":
        return transcribe_audio_elevenlabs(audio_path, language_hint=language, candidate_languages=candidate_languages)
    return _transcribe_audio_whisper(audio_path, language, candidate_languages, continuation_lang)


def transcribe_audio_elevenlabs(
    audio_path: str,
    language_hint: str | None = None,
    candidate_languages: list[str] | None = None,
) -> tuple[str, str | None, float, bool, dict]:
    empty_diag = {"seg_lang_check": False, "seg_lang_check_ms": 0.0}

    if not ELEVENLABS_API_KEY:
        logger.info("[STT][ELEVENLABS] ELEVENLABS_API_KEY sozlanmagan (.env'ni tekshiring) — "
            "bo'sh natija qaytarilyapti")
        return "", None, 0.0, False, empty_diag

    audio = _decode_waveform(audio_path)
    if audio is not None and audio.size:
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
        if rms < RMS_SILENCE_THRESHOLD:
            logger.info(f"[STT][ELEVENLABS] audio below RMS threshold (rms={rms:.5f} < "
                f"{RMS_SILENCE_THRESHOLD}) — skipping transcription")
            return "", None, 0.0, False, empty_diag

    t0 = time.perf_counter()
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            data = {"model_id": ELEVENLABS_MODEL_ID}
            if language_hint:
                data["language_code"] = language_hint
            response = requests.post(
                ELEVENLABS_STT_URL,
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                files=files,
                data=data,
                timeout=ELEVENLABS_TIMEOUT_S,
            )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        logger.info(f"[STT][ELEVENLABS] so'rov muvaffaqiyatsiz bo'ldi: {e!r} — bo'sh natija "
            f"qaytarilyapti (tizim davom etadi)")
        return "", None, 0.0, False, empty_diag

    text = (result.get("text") or "").strip()
    raw_lang_code = result.get("language_code") or None
    detected_lang = ELEVENLABS_LANG_CODE_MAP.get(raw_lang_code, raw_lang_code) if raw_lang_code else None
    lang_prob = float(result.get("language_probability") or 0.0)

    valid_langs = candidate_languages if candidate_languages else SUPPORTED_LANGUAGES
    unsupported_language = False
    if detected_lang not in valid_langs:
        logger.info(f"[STT][ELEVENLABS] detected_lang={detected_lang!r} (raw={raw_lang_code!r}) "
            f"{valid_langs}'dan tashqarida — ishonch 0.0'ga tushirildi")
        lang_prob = 0.0
        if candidate_languages:
            unsupported_language = True
            logger.info(
                f"[STT][LANG][ELEVENLABS] unsupported_language reason=out_of_pair detected_lang={detected_lang!r} "
                f"candidate_languages={candidate_languages} ichida yo'q — turn tashlab yuborildi"
            )
    elif candidate_languages and lang_prob < MIN_LANG_PROB:
        unsupported_language = True
        logger.info(
            f"[STT][LANG][ELEVENLABS] unsupported_language reason=low_lang_prob lang_prob={lang_prob:.3f} "
            f"< {MIN_LANG_PROB} detected_lang={detected_lang!r} — turn tashlab yuborildi"
        )
    if _is_phantom_phrase(text):
        logger.info(f"[STT][ELEVENLABS] phantom-ibora aniqlandi (text={text!r}) — matn bo'shatildi")
        text = ""

    elapsed = time.perf_counter() - t0
    logger.info(f"[STT][ELEVENLABS] {len(text)} chars in {elapsed:.2f}s "
        f"detected_lang={detected_lang} (raw={raw_lang_code!r}) ({lang_prob:.2f})")
    return text, detected_lang, lang_prob, unsupported_language, empty_diag


def _transcribe_audio_whisper(
    audio_path: str,
    language: str | None = None,
    candidate_languages: list[str] | None = None,
    continuation_lang: str | None = None,
) -> tuple[str, str | None, float, bool, dict]:
    model = get_model()

    empty_diag = {"seg_lang_check": False, "seg_lang_check_ms": 0.0}

    audio = _decode_waveform(audio_path)
    rms = None
    if audio is not None and audio.size:
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
        if rms < RMS_SILENCE_THRESHOLD:
            logger.info(f"[STT] audio below RMS threshold (rms={rms:.5f} < {RMS_SILENCE_THRESHOLD}) — skipping transcription")
            return "", None, 0.0, False, empty_diag

    source = audio if (audio is not None and audio.size) else audio_path

    use_vad = USE_VAD_FILTER and _ONNX_AVAILABLE
    t0 = time.perf_counter()

    candidates = [c for c in dict.fromkeys(candidate_languages or []) if c]

    if language is None and len(candidates) >= 2 and audio is not None and audio.size:
        detected_lang, lang_prob, lang_diag = _detect_language_restricted(model, audio, candidates)

        pair_mass = lang_diag["pair_mass"]
        raw_top_lang = lang_diag["raw_top_lang"]
        raw_top_prob = lang_diag["raw_top_prob"]
        top5_str = ", ".join(f"{lang}={p:.3f}" for lang, p in lang_diag["top5"])
        logger.info(
            f"[STT][LANG] pair=({','.join(candidates)}) pair_mass={pair_mass:.3f} "
            f"raw_top={raw_top_lang}({raw_top_prob:.3f}) top5=[{top5_str}]"
        )

        turn_duration_s = audio.size / SAMPLE_RATE
        reject_reason = _language_gate(lang_diag, candidates, turn_duration_s, continuation_lang)
        rival_checked = None
        if reject_reason == "foreign_rival" and lang_diag.get("rival_lang"):
            prompt = WHISPER_INITIAL_PROMPTS.get(detected_lang) if WHISPER_INITIAL_PROMPT_ENABLED else None
            cand_segs, cand_info = _run_transcribe(model, source, detected_lang, use_vad, initial_prompt=prompt)
            rival_segs, _ = _run_transcribe(model, source, lang_diag["rival_lang"], use_vad)
            cand_score, rival_score = _decode_score(cand_segs), _decode_score(rival_segs)
            logger.info(
                f"[STT][LANG] rival_decode {detected_lang}={cand_score:.3f} "
                f"{lang_diag['rival_lang']}={rival_score:.3f}"
            )
            if cand_score > rival_score:
                reject_reason = None
                rival_checked = (cand_segs, cand_info)
        if (
            reject_reason in ("low_pair_mass", "low_pair_mass_short")
            and pair_mass >= DECODE_RESCUE_MIN_PAIR_MASS
        ):
            # Qarang DECODE_RESCUE izohi.
            decoded = {}
            for cand in candidates:
                prompt = WHISPER_INITIAL_PROMPTS.get(cand) if WHISPER_INITIAL_PROMPT_ENABLED else None
                segs, seg_info = _run_transcribe(model, source, cand, use_vad, initial_prompt=prompt)
                decoded[cand] = (_decode_score(segs), segs, seg_info)
            ranked = sorted(decoded, key=lambda c: decoded[c][0], reverse=True)
            best_score = decoded[ranked[0]][0]
            second_score = decoded[ranked[1]][0] if len(ranked) > 1 else -10.0
            logger.info(
                "[STT][LANG] decode_rescue "
                + ", ".join(f"{c}={decoded[c][0]:.3f}" for c in ranked)
            )
            if best_score >= DECODE_RESCUE_MIN_SCORE and best_score - second_score >= DECODE_RESCUE_MARGIN:
                reject_reason = None
                detected_lang = ranked[0]
                rival_checked = (decoded[ranked[0]][1], decoded[ranked[0]][2])
        if reject_reason is not None:
            logger.info(
                f"[STT][LANG] unsupported_language reason={reject_reason} "
                f"pair_mass={pair_mass:.3f} duration={turn_duration_s:.2f}s "
                f"raw_top={raw_top_lang!r}({raw_top_prob:.3f}) continuation={continuation_lang!r} "
                f"pair=({','.join(candidates)}) — turn tashlab yuborildi"
            )
            return "", None, 0.0, True, empty_diag

        if lang_prob < MIN_LANG_PROB:
            logger.info(
                f"[STT][LANG] unsupported_language reason=low_lang_prob lang_prob={lang_prob:.3f} "
                f"< {MIN_LANG_PROB} detected_lang={detected_lang!r} pair_mass={pair_mass:.3f} "
                f"pair=({','.join(candidates)}) — turn tashlab yuborildi"
            )
            return "", None, 0.0, True, empty_diag

        turn_pair_mass = pair_mass
        initial_prompt = WHISPER_INITIAL_PROMPTS.get(detected_lang) if WHISPER_INITIAL_PROMPT_ENABLED else None
        if rival_checked is not None:
            seg_list, info = rival_checked
        else:
            seg_list, info = _run_transcribe(model, source, detected_lang, use_vad, initial_prompt=initial_prompt)
        if (
            lang_prob < AMBIGUOUS_LANG_PROB
            and rival_checked is None
            and not (continuation_lang is not None and detected_lang == continuation_lang)
        ):
            scores = {detected_lang: _decode_score(seg_list)}
            for alt_lang in candidates:
                if alt_lang == detected_lang:
                    continue
                alt_prompt = WHISPER_INITIAL_PROMPTS.get(alt_lang) if WHISPER_INITIAL_PROMPT_ENABLED else None
                alt_segs, alt_info = _run_transcribe(model, source, alt_lang, use_vad, initial_prompt=alt_prompt)
                scores[alt_lang] = _decode_score(alt_segs)
                if scores[alt_lang] > scores[detected_lang]:
                    detected_lang, seg_list, info = alt_lang, alt_segs, alt_info
            logger.info(
                f"[STT][LANG] ambiguous lang_prob={lang_prob:.3f} decode_scores="
                + ", ".join(f"{l}={v:.3f}" for l, v in scores.items())
                + f" -> {detected_lang}"
            )
    else:
        turn_pair_mass = None
        initial_prompt = (
            WHISPER_INITIAL_PROMPTS.get(language) if (WHISPER_INITIAL_PROMPT_ENABLED and language) else None
        )
        seg_list, info = _run_transcribe(model, source, language, use_vad, initial_prompt=initial_prompt)
        detected_lang = getattr(info, "language", None)
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)

    elapsed = time.perf_counter() - t0

    seg_foreign: list[bool] = [False] * len(seg_list)
    seg_lang_diag: list[dict | None] = [None] * len(seg_list)
    seg_lang_check_performed = False
    seg_lang_check_ms = 0.0
    if (
        turn_pair_mass is not None
        and turn_pair_mass < SEG_LANG_CHECK_BELOW
        and len(seg_list) > 1
        and audio is not None
        and audio.size
    ):
        t_seg0 = time.perf_counter()
        seg_foreign, seg_lang_diag = _segment_language_filter(model, audio, seg_list, candidates)
        seg_lang_check_performed = True
        seg_lang_check_ms = (time.perf_counter() - t_seg0) * 1000
        if any(seg_foreign):
            logger.info(
                f"[STT][LANG] seg_lang_check turn_pair_mass={turn_pair_mass:.3f} "
                f"dropped {sum(seg_foreign)}/{len(seg_list)} segment(s) in {seg_lang_check_ms:.1f}ms"
            )

    seg_rms_list: list[float | None] = []
    if audio is not None and audio.size:
        for seg in seg_list:
            seg_start_s = getattr(seg, "start", None)
            seg_end_s = getattr(seg, "end", None)
            if seg_start_s is None or seg_end_s is None:
                seg_rms_list.append(None)
                continue
            start_idx = max(0, int(seg_start_s * SAMPLE_RATE))
            end_idx = min(audio.size, int(seg_end_s * SAMPLE_RATE))
            if end_idx <= start_idx:
                seg_rms_list.append(None)
                continue
            seg_audio = audio[start_idx:end_idx]
            seg_rms_list.append(float(np.sqrt(np.mean(np.square(seg_audio.astype(np.float64))))))
    else:
        seg_rms_list = [None] * len(seg_list)

    valid_seg_rms = [r for r in seg_rms_list if r is not None]
    max_seg_rms = max(valid_seg_rms) if valid_seg_rms else None

    kept_parts = []
    raw_parts = []
    last_kept_nsp, last_kept_alp = 0.0, 0.0
    fallback_candidates = []
    dropped = 0
    for idx, (seg, seg_rms) in enumerate(zip(seg_list, seg_rms_list)):
        nsp = getattr(seg, "no_speech_prob", 0.0)
        alp = getattr(seg, "avg_logprob", 0.0)
        cr = getattr(seg, "compression_ratio", 0.0)
        seg_start = getattr(seg, "start", None)
        seg_end = getattr(seg, "end", None)
        seg_text = seg.text.strip()
        raw_parts.append(seg_text)

        is_far_field = (
            seg_rms is not None
            and max_seg_rms is not None
            and max_seg_rms > 0
            and seg_rms < SEGMENT_REL_RMS * max_seg_rms
        )
        if is_far_field:
            logger.info(
                f"[STT][DROP] reason=far_field text={seg_text!r} rms={seg_rms:.4f} "
                f"max_rms={max_seg_rms:.4f} start={seg_start} end={seg_end}"
            )
            dropped += 1
            continue

        if seg_foreign[idx]:
            seg_diag = seg_lang_diag[idx] or {}
            logger.info(
                f"[STT][DROP] reason=foreign_segment text={seg_text!r} "
                f"seg_pair_mass={seg_diag.get('pair_mass', 0.0):.3f} "
                f"raw_top={seg_diag.get('raw_top_lang')!r}({seg_diag.get('raw_top_prob', 0.0):.3f})"
            )
            dropped += 1
            continue

        seg_duration_s = (
            (seg_end - seg_start) if (seg_start is not None and seg_end is not None) else 0.0
        )
        is_phantom = _is_phantom_drop(seg_text, seg_duration_s, nsp, alp)
        if not is_phantom:
            fallback_candidates.append(seg_text)

        is_low_confidence = nsp > NO_SPEECH_PROB_THRESHOLD and alp < AVG_LOGPROB_THRESHOLD
        is_long_compression = cr > COMPRESSION_RATIO_THRESHOLD
        if is_low_confidence or is_long_compression or is_phantom:
            reasons = []
            if is_low_confidence:
                reasons.append(
                    f"low_confidence (no_speech_prob={nsp:.3f} > {NO_SPEECH_PROB_THRESHOLD} "
                    f"AND avg_logprob={alp:.3f} < {AVG_LOGPROB_THRESHOLD})"
                )
            if is_long_compression:
                reasons.append(f"compression_ratio {cr:.2f} > {COMPRESSION_RATIO_THRESHOLD}")
            if is_phantom:
                reasons.append("phantom")
            logger.info(
                f"[STT][DROP] text={seg_text!r} no_speech_prob={nsp:.3f} avg_logprob={alp:.3f} "
                f"compression_ratio={cr:.2f} start={seg_start} end={seg_end} "
                f"reason=({'; '.join(reasons)})"
            )
            dropped += 1
            continue

        if DEBUG_STT:
            logger.info(f"[STT][DEBUG] segment KEPT    [{seg_start}-{seg_end}] text={seg_text!r} "
                f"no_speech_prob={nsp:.3f} avg_logprob={alp:.3f}")
        kept_parts.append(seg_text)
        last_kept_nsp, last_kept_alp = nsp, alp

    text = "".join(kept_parts).strip()

    if not text and fallback_candidates:
        fallback_text = "".join(fallback_candidates).strip()
        if fallback_text:
            if DEBUG_STT:
                logger.info("[STT][DEBUG] fallback used: all segments filtered, using raw transcript")
            text = fallback_text

    if not text and seg_lang_check_performed and any(seg_foreign):
        logger.info(
            f"[STT][LANG] unsupported_language reason=all_segments_foreign "
            f"pair=({','.join(candidates)}) — turn tashlab yuborildi"
        )
        return (
            "",
            None,
            0.0,
            True,
            {
                "seg_lang_check": seg_lang_check_performed,
                "seg_lang_check_ms": seg_lang_check_ms,
                "pair_mass": turn_pair_mass,
            },
        )

    turn_duration_s = (audio.size / SAMPLE_RATE) if (audio is not None and audio.size) else 0.0
    if _is_phantom_drop(text, turn_duration_s, last_kept_nsp, last_kept_alp):
        logger.info(
            f"[STT][DROP] reason=phantom text={text!r} turn_duration={turn_duration_s:.2f}s "
            f"no_speech_prob={last_kept_nsp:.3f} avg_logprob={last_kept_alp:.3f} — final text emptied"
        )
        text = ""
    elif _is_low_conf_short_drop(text, turn_duration_s, last_kept_alp) and not (
        continuation_lang is not None and detected_lang == continuation_lang
    ):
        logger.info(
            f"[STT][DROP] reason=low_conf_short text={text!r} turn_duration={turn_duration_s:.2f}s "
            f"avg_logprob={last_kept_alp:.3f} — final text emptied"
        )
        text = ""

    msg = f"[STT] {len(text)} chars in {elapsed:.2f}s detected_lang={detected_lang} ({lang_prob:.2f})"
    if dropped:
        msg += f" ({dropped} segment(s) filtered)"
    if seg_lang_check_performed:
        msg += f" seg_lang_check={seg_lang_check_ms:.1f}ms"
    logger.info(msg)

    return (
        text,
        detected_lang,
        lang_prob,
        False,
        {
            "seg_lang_check": seg_lang_check_performed,
            "seg_lang_check_ms": seg_lang_check_ms,
            "pair_mass": turn_pair_mass,
        },
    )
