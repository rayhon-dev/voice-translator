import time
import traceback

import numpy as np
import soundfile as sf
import torch  # noqa: F401 — ctranslate2 uchun libcublas.so.12'ni oldindan yuklaydi
from faster_whisper import WhisperModel
from app.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE


NO_SPEECH_PROB_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0
RMS_SILENCE_THRESHOLD = 0.003
USE_VAD_FILTER = True

# Muvozanatli fallback: temp=0.0 muvaffaqiyatsiz bo'lsa (compression_ratio/logprob
# mezonlaridan o'tmasa), faster-whisper faqat BITTA marta temp=0.4 bilan qayta urinadi
# (default 6-qiymatli (0.0..1.0) fallback ro'yxati o'rniga) — bu eng katta kechikish
# manbai edi (har bir qo'shimcha temperature — butun segmentni qayta decode qilish).
TRANSCRIBE_TEMPERATURE = (0.0, 0.4)

DEBUG_STT = False

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


def _decode_waveform_bytes(chunk: bytes):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
        tmp.write(chunk)
        tmp.flush()
        return _decode_waveform(tmp.name)


def _write_wav(path: str, wav: np.ndarray) -> None:
    sf.write(path, wav, SAMPLE_RATE)


def _run_transcribe(model, source, language: str | None, use_vad: bool):
    """model.transcribe() ni chaqiradi; vad_filter xato bersa (masalan qisqa
    audio'da faster-whisper'ning ichki til aniqlash kodi bo'sh to'plamda max()
    chaqirishi mumkin) — VAD'siz BITTA marta qayta uriniladi."""
    t0 = time.perf_counter()
    try:
        segments, info = model.transcribe(
            source,
            language=language,
            vad_filter=use_vad,
            condition_on_previous_text=False,
            temperature=TRANSCRIBE_TEMPERATURE,
        )
        return list(segments), info
    except Exception as e:
        first_attempt_elapsed = time.perf_counter() - t0
        if not use_vad:
            raise
        print(
            f"[STT][VAD_RETRY] vad_filter failed after {first_attempt_elapsed:.2f}s "
            f"(language={language!r}, exception={e!r}); retrying without VAD\n"
            f"{traceback.format_exc()}",
            flush=True,
        )
        t_retry0 = time.perf_counter()
        segments, info = model.transcribe(
            source,
            language=language,
            condition_on_previous_text=False,
            temperature=TRANSCRIBE_TEMPERATURE,
        )
        seg_list = list(segments)
        print(
            f"[STT][VAD_RETRY] retry (no VAD) took {time.perf_counter() - t_retry0:.2f}s "
            f"(first attempt had already spent {first_attempt_elapsed:.2f}s)",
            flush=True,
        )
        return seg_list, info


def _detect_language_restricted(model, audio: np.ndarray, candidates: list[str]) -> tuple[str, float]:
    """Whisper'ning o'zining til-aniqlash (language-ID) boshini ishlatadi va
    faqat berilgan nomzod tillar orasidan eng ehtimolini tanlaydi.

    Ilgari sinovdan o'tgan yondashuv — ikkala tilni ham majburan to'liq decode
    qilib, avg_logprob'larini solishtirish — juda shovqinli chiqdi (masalan
    real testda en=-0.685 va ru=-0.682 kabi, deyarli tasodifiy natija berardi),
    chunki forced-decode noto'g'ri tilda ham "ishonchli ko'ringan" matn
    generatsiya qilishi mumkin. Bu funksiya esa shu vazifa uchun maxsus
    o'qitilgan klassifikator boshidan (`model.detect_language`) foydalanadi —
    faqat BITTA yengil forward pass (to'liq decode emas) talab qiladi.

    E'TIBOR: qaytarilgan ehtimollik (xom, ~99 tilli taqsimotdan) real
    mikrofon audiosida past bo'lishi mumkin (masalan haqiqiy aniq inglizcha
    gap uchun ham 0.2-0.5 atrofida chiqishi kuzatilgan) — shuning uchun bu
    qiymatni mutlaq chegara sifatida emas, faqat ma'lumot/pending-streak
    uchun ishlating. Asosiy qabul/rad qarori shu funksiya ICHIDA, global
    argmax tekshiruvi orqali qabul qilinadi (pastga qarang)."""
    features = model.feature_extractor(audio)
    segment = features[:, : model.feature_extractor.nb_max_frames]
    encoder_output = model.encode(segment)
    results = model.model.detect_language(encoder_output)[0]
    all_probs = {token[2:-2]: prob for token, prob in results}
    cand_probs = {lang: all_probs.get(lang, 0.0) for lang in candidates}

    print(f"[STT][LANGID] candidates={cand_probs}", flush=True)

    if not all_probs:
        return candidates[0], 0.0

    best_lang = max(cand_probs, key=cand_probs.get)
    global_best_lang = max(all_probs, key=all_probs.get)
    print(
        f"[STT][LANGID] global_top={global_best_lang!r} "
        f"({all_probs.get(global_best_lang, 0.0):.3f})",
        flush=True,
    )

    # E'TIBOR: mutlaq foiz chegarasi ISHLATILMAYDI (masalan "0.75 dan yuqori
    # bo'lsin"). Real (studiya emas, mikrofon orqali, aksentli, fon shovqini
    # bilan) nutqda XOM ehtimollik to'g'ri til uchun ham past chiqishi mumkin
    # (real testda aniq inglizcha gap uchun en=0.26, aniq ruscha gap uchun
    # ru=0.24 kabi chiqdi — 0.75'dan ancha past). Shuning uchun buning o'rniga
    # "shu nomzod til to'liq ~99 tilli taqsimotda ENG YUQORI ehtimolli
    # tanlovmi" tekshiriladi — bu mutlaq qiymatga emas, NISBIY tartibga
    # asoslangani uchun yozib olish sharoitidan (past ovoz, shovqin) deyarli
    # ta'sirlanmaydi, lekin haqiqatan ham boshqa (uchinchi) tilni ishonchli
    # ajratib beradi.
    if global_best_lang not in candidates:
        return None, cand_probs[best_lang]

    return best_lang, cand_probs[best_lang]


def transcribe_audio(
    audio_path: str,
    language: str | None = None,
    candidate_languages: list[str] | None = None,
) -> tuple[str, str | None, float]:
    """`candidate_languages` berilsa (va `language` berilmagan bo'lsa), Whisper'ning
    ~99 tilli avtomatik aniqlashi o'rniga faqat shu ro'yxatdagi tillar orasida
    (Whisper'ning language-ID boshi orqali) tanlanadi, so'ng transkripsiya shu
    bitta tanlangan til bilan majburlanadi. Dialog rejimida so'zlovchilar tilini
    oldindan tanlagani uchun bu — qisqa/shovqinli navbatlarda avtomatik aniqlash
    tasodifiy boshqa tilni (masalan gruzincha yoki bengalcha) tanlab qo'yishining
    oldini oladi."""

    model = get_model()

    audio = _decode_waveform(audio_path)
    rms = None
    if audio is not None and audio.size:
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
        if rms < RMS_SILENCE_THRESHOLD:
            print(f"[STT] audio below RMS threshold (rms={rms:.5f} < {RMS_SILENCE_THRESHOLD}) — skipping transcription", flush=True)
            return "", None, 0.0


    source = audio if (audio is not None and audio.size) else audio_path

    use_vad = USE_VAD_FILTER and _ONNX_AVAILABLE
    t0 = time.perf_counter()

    candidates = [c for c in dict.fromkeys(candidate_languages or []) if c]

    if language is None and len(candidates) >= 2 and audio is not None and audio.size:
        detected_lang, lang_prob = _detect_language_restricted(model, audio, candidates)
        seg_list, info = _run_transcribe(model, source, detected_lang, use_vad)
    else:
        seg_list, info = _run_transcribe(model, source, language, use_vad)
        detected_lang = getattr(info, "language", None)
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)

    elapsed = time.perf_counter() - t0

    kept_parts = []
    raw_parts = []
    dropped = 0
    for seg in seg_list:
        nsp = getattr(seg, "no_speech_prob", 0.0)
        alp = getattr(seg, "avg_logprob", 0.0)
        seg_start = getattr(seg, "start", None)
        seg_end = getattr(seg, "end", None)
        seg_text = seg.text.strip()
        raw_parts.append(seg_text)

        if nsp > NO_SPEECH_PROB_THRESHOLD or alp < AVG_LOGPROB_THRESHOLD:
            reasons = []
            if nsp > NO_SPEECH_PROB_THRESHOLD:
                reasons.append(f"no_speech_prob {nsp:.3f} > {NO_SPEECH_PROB_THRESHOLD}")
            if alp < AVG_LOGPROB_THRESHOLD:
                reasons.append(f"avg_logprob {alp:.3f} < {AVG_LOGPROB_THRESHOLD}")
            if DEBUG_STT:
                print(
                    f"[STT][DEBUG] segment DROPPED [{seg_start}-{seg_end}] text={seg_text!r} "
                    f"no_speech_prob={nsp:.3f} avg_logprob={alp:.3f} reason=({'; '.join(reasons)})",
                    flush=True,
                )
            dropped += 1
            continue

        if DEBUG_STT:
            print(
                f"[STT][DEBUG] segment KEPT    [{seg_start}-{seg_end}] text={seg_text!r} "
                f"no_speech_prob={nsp:.3f} avg_logprob={alp:.3f}",
                flush=True,
            )
        kept_parts.append(seg_text)

    text = "".join(kept_parts).strip()

    if not text and raw_parts:
        fallback_text = "".join(raw_parts).strip()
        if fallback_text:
            if DEBUG_STT:
                print(
                    "[STT][DEBUG] fallback used: all segments filtered, using raw transcript",
                    flush=True,
                )
            text = fallback_text

    msg = f"[STT] {len(text)} chars in {elapsed:.2f}s detected_lang={detected_lang} ({lang_prob:.2f})"
    if dropped:
        msg += f" ({dropped} low-confidence segment(s) filtered)"
    print(msg, flush=True)

    return text, detected_lang, lang_prob
