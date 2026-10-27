import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


try:
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    _VAD_AVAILABLE = True
except Exception:
    _VAD_AVAILABLE = False

SAMPLE_RATE = 16000


SEG_WIN_S = float(os.environ.get("SEG_WIN_S", "2.5"))
SEG_HOP_S = float(os.environ.get("SEG_HOP_S", "1.25"))


SEG_SILENCE_GAP_S = float(os.environ.get("SEG_SILENCE_GAP_S", "0.6"))

SPK_CHANGE_SIM = float(os.environ.get("SPK_CHANGE_SIM", "0.45"))

SUBSEG_MIN_S = float(os.environ.get("SUBSEG_MIN_S", "1.5"))

FAST_PATH_MAX_DURATION_S = SEG_WIN_S


def _single_segment(wav: np.ndarray, sample_rate: int):
    duration_s = len(wav) / sample_rate
    return [{"start": 0.0, "end": duration_s, "wav": wav, "purity": 1.0}]


def speech_spans(wav: np.ndarray, min_silence_duration_ms: float | None = None) -> list[tuple[float, float]]:
    gap_ms = SEG_SILENCE_GAP_S * 1000 if min_silence_duration_ms is None else min_silence_duration_ms
    if not _VAD_AVAILABLE:
        return [(0.0, len(wav) / SAMPLE_RATE)]
    try:
        opts = VadOptions(min_silence_duration_ms=int(gap_ms))
        raw = get_speech_timestamps(wav, opts)
    except Exception as e:
        logger.info(f"[SEG] get_speech_timestamps failed: {e!r} — butun bo'lak bitta oraliq deb olinadi")
        return [(0.0, len(wav) / SAMPLE_RATE)]
    if not raw:
        return [(0.0, len(wav) / SAMPLE_RATE)]
    return [(seg["start"] / SAMPLE_RATE, seg["end"] / SAMPLE_RATE) for seg in raw]


def _speech_spans(wav: np.ndarray) -> list[tuple[float, float]]:
    return speech_spans(wav)


def windows_for_span(
    start_s: float, end_s: float, win_s: float | None = None, hop_s: float | None = None
) -> list[tuple[float, float]]:
    win_s = SEG_WIN_S if win_s is None else win_s
    hop_s = SEG_HOP_S if hop_s is None else hop_s
    span_len = end_s - start_s
    if span_len <= win_s:
        return [(start_s, end_s)]
    windows = []
    w_start = start_s
    while w_start < end_s:
        w_end = min(w_start + win_s, end_s)
        windows.append((w_start, w_end))
        if w_end >= end_s:
            break
        w_start += hop_s
    
    if len(windows) >= 2 and (windows[-1][1] - windows[-1][0]) < hop_s:
        prev_start, _ = windows[-2]
        _, last_end = windows[-1]
        windows[-2] = (prev_start, last_end)
        windows.pop()
    return windows


def _windows_for_span(start_s: float, end_s: float) -> list[tuple[float, float]]:
    return windows_for_span(start_s, end_s)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


SPLIT_SEARCH_RADIUS_S = 0.4


def _quietest_point(wav: np.ndarray, sample_rate: int, around_s: float) -> float:
    frame = int(0.02 * sample_rate)
    lo = max(0, int((around_s - SPLIT_SEARCH_RADIUS_S) * sample_rate))
    hi = min(len(wav), int((around_s + SPLIT_SEARCH_RADIUS_S) * sample_rate))
    n = (hi - lo) // frame
    if n <= 0:
        return around_s
    frames = wav[lo: lo + n * frame].astype(np.float64).reshape(n, frame)
    quietest = int(np.argmin(np.mean(frames ** 2, axis=1)))
    return (lo + quietest * frame + frame / 2) / sample_rate


def segment_turn(wav: np.ndarray, sample_rate: int, seq: int | None = None) -> list[dict]:

    t0 = time.perf_counter()
    duration_s = len(wav) / sample_rate

    if duration_s <= FAST_PATH_MAX_DURATION_S:
        return _single_segment(wav, sample_rate)
    
    spans = _speech_spans(wav)
    if len(spans) <= 1 and duration_s <= FAST_PATH_MAX_DURATION_S:
        return _single_segment(wav, sample_rate)

   
    from app.models import speaker_diart

    all_windows: list[tuple[int, float, float]] = []
    for span_idx, (s_start, s_end) in enumerate(spans):
    
        if len(spans) == 1 and duration_s <= 2 * SEG_WIN_S:
            middle = _quietest_point(wav, sample_rate, (s_start + s_end) / 2)
            windows = [(s_start, middle), (middle, s_end)]
        else:
            windows = _windows_for_span(s_start, s_end)
        for w_start, w_end in windows:
            all_windows.append((span_idx, w_start, w_end))

    if len(all_windows) <= 1:
        return _single_segment(wav, sample_rate)

   
    embeddings = []
    for _, w_start, w_end in all_windows:
        start_idx = max(0, int(w_start * sample_rate))
        end_idx = min(len(wav), int(w_end * sample_rate))
        win_wav = wav[start_idx:end_idx]
        if win_wav.size == 0:
            embeddings.append(None)
            continue
        try:
            embeddings.append(speaker_diart._embed(win_wav, sample_rate))
        except Exception as e:
            logger.error(f"[SEG] window embedding failed: {e!r} — segmentatsiya bekor qilinadi")
            return _single_segment(wav, sample_rate)

    
    groups: list[list[int]] = [[0]]
    centroid_sum = embeddings[0].copy() if embeddings[0] is not None else None
    centroid_count = 1 if embeddings[0] is not None else 0
    for i in range(1, len(all_windows)):
        is_boundary = False
        if embeddings[i] is not None and centroid_count > 0:
            centroid = centroid_sum / np.linalg.norm(centroid_sum)
            sim = _cosine_sim(centroid, embeddings[i])
            threshold = min(SPK_CHANGE_SIM, 0.30) if duration_s <= 2 * SEG_WIN_S else SPK_CHANGE_SIM
            if sim < threshold:
                is_boundary = True
        if is_boundary:
            groups.append([i])
            centroid_sum = embeddings[i].copy() if embeddings[i] is not None else None
            centroid_count = 1 if embeddings[i] is not None else 0
        else:
            groups[-1].append(i)
            if embeddings[i] is not None:
                centroid_sum = embeddings[i].copy() if centroid_sum is None else centroid_sum + embeddings[i]
                centroid_count += 1

    def group_bounds(idxs: list[int]) -> tuple[float, float]:
        first = all_windows[idxs[0]]
        last = all_windows[idxs[-1]]
        return first[1], last[2]

    def group_purity(idxs: list[int]) -> float:

        embs = [embeddings[i] for i in idxs if embeddings[i] is not None]
        if len(embs) <= 1:
            return 1.0
        centroid = sum(embs)
        centroid = centroid / np.linalg.norm(centroid)
        return min(_cosine_sim(centroid, e) for e in embs)

    
    merged_any = True
    while merged_any:
        merged_any = False
        for i, idxs in enumerate(groups):
            g_start, g_end = group_bounds(idxs)
            if (g_end - g_start) >= SUBSEG_MIN_S or len(groups) == 1:
                continue
            if i > 0:
                groups[i - 1] = groups[i - 1] + idxs
                del groups[i]
            else:
                groups[i + 1] = idxs + groups[i + 1]
                del groups[i]
            merged_any = True
            break


    cuts = [0.0]
    for left, right in zip(groups, groups[1:]):
        _, ls, le = all_windows[left[-1]]
        _, rs, re = all_windows[right[0]]
        cuts.append((ls + le + rs + re) / 4)
    cuts.append(duration_s)
    subsegments = []
    for group_index, idxs in enumerate(groups):
        g_start, g_end = cuts[group_index:group_index + 2]
        start_idx = max(0, int(g_start * sample_rate))
        end_idx = min(len(wav), int(g_end * sample_rate))
        subsegments.append(
            {
                "start": g_start,
                "end": g_end,
                "wav": wav[start_idx:end_idx],
                "purity": group_purity(idxs),
            }
        )

    took_ms = (time.perf_counter() - t0) * 1000
    sub_desc = [(f"{s['start']:.2f}-{s['end']:.2f}", round(s["purity"], 3)) for s in subsegments]
    changes = [round(s["start"], 2) for s in subsegments[1:]]
    logger.info(
        f"[SEG] seq={seq} windows={len(all_windows)} changes={changes} "
        f"subsegments={sub_desc} took={took_ms:.1f}ms"
    )

    return subsegments if subsegments else _single_segment(wav, sample_rate)


def language_recovery_parts(wav, sample_rate, window_s, min_s):
    """Bounded, disjoint retry windows; never retry the unchanged chunk."""
    duration = len(wav) / sample_rate
    if duration < 2 * min_s:
        return []
    spans = _speech_spans(wav)
    parts = []
    for start, end in spans:
        length = end - start
        if length < min_s:
            continue
        count = max(1, int(np.ceil(length / window_s)))
        if len(spans) == 1:
            count = max(2, count)
        count = min(count, int(length / min_s))
        for i in range(count):
            a = int((start + length * i / count) * sample_rate)
            b = int((start + length * (i + 1) / count) * sample_rate)
            parts.append(wav[a:b])
    if len(parts) == 1 and len(parts[0]) == len(wav):
        return []
    return parts
