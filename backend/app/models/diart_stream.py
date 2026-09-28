import threading
import time
import wave

import numpy as np
import torch

from app import diart_compat  # noqa: F401 — `diart` import qilinishidan OLDIN kerak
from diart import models as m
from diart.blocks import SpeakerDiarization, SpeakerDiarizationConfig
from diart.blocks.clustering import OnlineSpeakerClustering
from diart.mapping import SpeakerMapBuilder
from diart.operators import rearrange_audio_stream
from rx.subject import Subject

PCM_DUMP_PATH = "/tmp/session_pcm.wav"

# --- VAQTINCHALIK DIAGNOSTIKA: har bir diart qadamida local spiker embeddingi
# bilan mavjud global klasterlar orasidagi haqiqiy kosinus-masofani logga
# chiqaradi (delta_new bilan solishtirish uchun). Diart kutubxonasining o'z
# faylini o'zgartirmasdan, faqat shu metodni "o'rab" (wrap) qo'yamiz — qaror
# qabul qilish mantig'i (original identify()) o'zgarishsiz qoladi, faqat
# tashqarisidan kuzatiladi. Kalibratsiya tugagach shu blok olib tashlansin. ---
_original_identify = OnlineSpeakerClustering.identify


def _identify_with_distance_logging(self, segmentation, embeddings):
    if self.centers is not None:
        try:
            emb_np = embeddings.detach().cpu().numpy()
            active_speakers = np.where(
                np.max(segmentation.data, axis=0) >= self.tau_active
            )[0]
            if len(active_speakers) > 0:
                dist_map = SpeakerMapBuilder.dist(emb_np, self.centers, self.metric)
                active_centers = sorted(self.active_centers)
                for spk in active_speakers:
                    dist_str = ", ".join(
                        f"c{c}={dist_map.mapping_matrix[spk, c]:.3f}" for c in active_centers
                    )
                    print(
                        f"[DIART][DIST] local_spk={spk} distances=[{dist_str}] "
                        f"delta_new={self.delta_new}",
                        flush=True,
                    )
        except Exception as e:
            print(f"[DIART][DIST][ERROR] {e!r}", flush=True)
    return _original_identify(self, segmentation, embeddings)


OnlineSpeakerClustering.identify = _identify_with_distance_logging

SAMPLE_RATE = 16000
DIART_DURATION = 5.0
DIART_STEP = 0.5
MAX_SPEAKERS = 3
DELTA_NEW = 0.7  # yangi so'zlovchi deb topish uchun kosinus-masofa chegarasi.
# Real sinovlarda: 1.5/1.0/0.8/0.75 — hammasi bitta klasterga birlashtirdi (YUQORI);
# 0.6 — A/B'ning o'z ovozi bo'linib ketdi (PAST); 0.65'da begona odam ba'zan o'tib
# ketdi. To'g'ri qiymat 0.65-0.75 oralig'ida, juda tor ko'rinmoqda — shu ikkisi
# orasida davom ettirilmoqda.

RHO_UPDATE = 0.2  # local spiker yangi klaster/yangilanish uchun 5s oynaning kamida
# shu ulushida (0.2 = ~1s) faol bo'lishi kerak. Diart default'i 0.3 (~1.5s) — qisqa
# navbatlar uchun juda qattiq bo'lishi mumkin edi; 0.15 esa ortiqcha bo'linishga
# hissa qo'shgan bo'lishi mumkin (juda oson yangi klaster yaratilardi).
TIMELINE_HORIZON_SECONDS = 120.0

_segmentation_model = None
_embedding_model = None


def _load_models():
    global _segmentation_model, _embedding_model
    if _segmentation_model is None:
        _segmentation_model = m.SegmentationModel.from_pyannote("pyannote/segmentation")
    if _embedding_model is None:
        _embedding_model = m.EmbeddingModel.from_pyannote("pyannote/embedding")
    return _segmentation_model, _embedding_model


def warmup(run_inference: bool = True) -> None:
    t0 = time.perf_counter()
    _load_models()
    print(f"[DIART] segmentation+embedding models loaded in {time.perf_counter() - t0:.2f}s", flush=True)
    if not run_inference:
        return
    try:
        session = create_stream_session()
        silence = np.zeros(int(DIART_STEP * SAMPLE_RATE), dtype=np.int16).tobytes()
        steps = int(DIART_DURATION / DIART_STEP) + 1
        for _ in range(steps):
            feed_pcm_chunk(session, silence)
        close_session(session)
        print("[DIART] warm-up inference done", flush=True)
    except Exception as e:
        print(f"[DIART] warm-up inference skipped: {e!r}", flush=True)


def create_stream_session():
    segmentation, embedding = _load_models()
    config = SpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        duration=DIART_DURATION,
        step=DIART_STEP,
        delta_new=DELTA_NEW,
        rho_update=RHO_UPDATE,
        max_speakers=MAX_SPEAKERS,
        device=torch.device("cpu"),
        sample_rate=SAMPLE_RATE,
    )
    wav_writer = None
    try:
        wav_writer = wave.open(PCM_DUMP_PATH, "wb")
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(SAMPLE_RATE)
    except Exception as e:
        print(f"[DIART][WAV-DUMP][ERROR] could not open {PCM_DUMP_PATH}: {e!r}", flush=True)

    session = {
        "pipeline": SpeakerDiarization(config),
        "subject": Subject(),
        "timeline": [],
        "seen_labels": set(),
        "wav_writer": wav_writer,
        "lock": threading.Lock(),
    }

    def on_window(window):
        t0 = time.perf_counter()
        outputs = session["pipeline"]([window])
        took_ms = (time.perf_counter() - t0) * 1000
        print(f"[DIART][STEP] took={took_ms:.1f}ms", flush=True)
        if took_ms > 500:
            print(
                f"[DIART][STEP][WARN] step took {took_ms:.1f}ms > 500ms budget — "
                f"real-time bilan orqada qolishi mumkin",
                flush=True,
            )
        annotation, _ = outputs[0]
        with session["lock"]:
            for segment, _, label in annotation.itertracks(yield_label=True):
                if label not in session["seen_labels"]:
                    session["seen_labels"].add(label)
                    print(
                        f"[DIART][NEW-CLUSTER] {label!r} birinchi marta ko'rindi "
                        f"(t={segment.start:.2f}s) — jami klasterlar: {sorted(session['seen_labels'])}",
                        flush=True,
                    )
                session["timeline"].append(
                    {"start": segment.start, "end": segment.end, "raw_label": label}
                )
            _prune_timeline(session)

    session["subject"].pipe(
        rearrange_audio_stream(duration=DIART_DURATION, step=DIART_STEP, sample_rate=SAMPLE_RATE)
    ).subscribe(on_window)

    return session


def _prune_timeline(session):
    tl = session["timeline"]
    if not tl:
        return
    cutoff = tl[-1]["end"] - TIMELINE_HORIZON_SECONDS
    while tl and tl[0]["end"] < cutoff:
        tl.pop(0)


def feed_pcm_chunk(session, pcm_int16_bytes: bytes) -> None:
    wav_writer = session.get("wav_writer")
    if wav_writer is not None:
        try:
            wav_writer.writeframes(pcm_int16_bytes)
        except Exception as e:
            print(f"[DIART][WAV-DUMP][ERROR] write failed: {e!r}", flush=True)

    wav = np.frombuffer(pcm_int16_bytes, dtype="<i2").astype(np.float32) / 32768.0
    chunk = wav.reshape(1, -1)
    session["subject"].on_next(chunk)


def close_session(session) -> None:
    wav_writer = session.get("wav_writer")
    if wav_writer is not None:
        try:
            wav_writer.close()
            print(f"[DIART][WAV-DUMP] session PCM saqlandi: {PCM_DUMP_PATH}", flush=True)
        except Exception as e:
            print(f"[DIART][WAV-DUMP][ERROR] close failed: {e!r}", flush=True)
        session["wav_writer"] = None


def match_turn_to_speaker(session, turn_start: float, turn_end: float) -> str | None:
    with session["lock"]:
        totals: dict[str, float] = {}
        for entry in session["timeline"]:
            overlap = min(entry["end"], turn_end) - max(entry["start"], turn_start)
            if overlap > 0:
                totals[entry["raw_label"]] = totals.get(entry["raw_label"], 0.0) + overlap
    if not totals:
        print(f"[DIART-MATCH] turn=[{turn_start:.2f},{turn_end:.2f}] no diart overlap yet", flush=True)
        return None

    best_label = max(totals, key=totals.get)
    turn_duration = max(turn_end - turn_start, 1e-6)
    share = totals[best_label] / turn_duration
    rounded_totals = {k: round(v, 2) for k, v in totals.items()}
    print(
        f"[DIART-MATCH] turn=[{turn_start:.2f},{turn_end:.2f}] raw_label={best_label!r} "
        f"share={share:.2f} totals={rounded_totals}",
        flush=True,
    )
    return best_label
