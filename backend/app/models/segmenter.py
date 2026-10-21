"""Mentor demo'da topilgan B-guruh muammosi (5-band): bitta WebSocket
audio bo'lagi ("turn") hozircha BITTA spiker va BITTA til deb qayta
ishlanadi. Bu quyidagi real holatlarda noto'g'ri natijaga olib keladi:
  - Foydalanuvchi asosan inglizcha gapirib, OXIRIDA o'zbekcha bir necha
    so'z qo'shsa — BUTUN bo'lak begona til deb tashlab yuborilardi.
  - Ikki kishi orasida PAUZASIZ navbat almashsa (bitta uzun bo'lak ichida)
    — ikkinchisi birinchisining spikeriga/tiliga "yopishtirilardi".
  - Aralash bo'lakdan olingan BITTA embedding ikki ovozni o'rtachalab,
    spiker profilini "ifloslantirardi".

Shu modul bitta turn'ning XOM audiosini, agar kerak bo'lsa, bir nechta
QISM (sub-segment)ga ajratadi — har biri keyinroq dialog.py tomonidan
MUSTAQIL (o'z tili, o'z spikeri, o'z tarjimasi bilan) qayta ishlanadi.

YONDASHUV TANLOVI (5-band: "qaysi biri ishonchliroq, sinab tanla"):
ikkita muqobil usul ko'rib chiqildi —
  (A) Qo'shni OYNALAR orasidagi COSINE O'XSHASHLIKKA chegara qo'yish
      (SPK_CHANGE_SIM) — oddiy, O(n) ketma-ket o'tish, hech qanday
      klasterlash kerak emas.
  (B) Har bir oynani MAVJUD global spiker klasterlariga (speaker_diart.py)
      bog'lab, "label" o'zgargan joyda kesish.
(B) jozibali ko'rinadi (global spiker ID'lari bilan to'g'ridan-to'g'ri
mos keladi), lekin ikkita jiddiy kamchiligi bor: 1) bu bosqichda hali
BIRON bir spiker ANIQLANMAGAN (identify_speaker() navbatdagi BOSQICHDA,
HAR BIR qism uchun ALOHIDA chaqiriladi — segmenter o'zi spiker-klaster
holatiga yozishi KERAK EMAS, aks holda ikki mas'uliyat aralashib ketardi);
2) global chegaralar (SPK_ACCEPT/SPK_NEW) BOSHQA maqsad uchun — "bu ALLAQACHON
ma'lum odam" — sozlangan, "ketma-ket ikki oyna bir xil odammi" degan
LOKAL savol uchun unchalik mos emas (ayniqsa hali profilsiz YANGI spiker
uchun). Shu sababli (A) tanlandi — sodda, mas'uliyati aniq ajratilgan,
va amalda yetarli (qarang quyidagi SPK_CHANGE_SIM izohi).
"""
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

# faster_whisper.vad — stt.py'da ALLAQACHON xuddi shu maqsadda (tezkor,
# faqat-VAD "nutq bormi" tekshiruvi uchun) ishlatiladi — shu yerda AYNAN
# o'sha funksiya, endi "qayerda nutq BOR" (vaqt oralig'i, soniyada EMAS,
# NAMUNADA) uchun qayta ishlatiladi.
try:
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    _VAD_AVAILABLE = True
except Exception:
    _VAD_AVAILABLE = False

SAMPLE_RATE = 16000

# --- 5-band: oyna parametrlari ---
# EMPIRIK TOPILMA (debug_audio'dagi HAQIQIY yozuvlar bilan sinovda):
# 5-bandda taklif qilingan SEG_WIN_S=1.5/SEG_HOP_S=0.75 bilan BIR XIL
# odamning (hech qanday spiker almashinuvisiz) 8s monologi NOTO'G'RI 3-4
# qismga bo'linib ketdi — sabab, ushbu loyihadagi embedding modeli
# (pyannote WeSpeaker, "window=whole") 1.5s'lik juda qisqa kesmalarda
# PROSODIYA/UNDOSH tovlanishidan kelib chiqadigan shovqinga nisbatan
# o'zi (bir xil odam ICHIDA) qo'shni oynalar orasida 0.3-0.8 oralig'ida
# "sakraydigan" o'xshashlik berdi — SPK_CHANGE_SIM=0.45'dan chegarani
# ishonchli ajratib bo'lmadi. Oynani 2.5s/1.25s'ga oshirgandan keyin
# (qarang scratchpad/seg_debug_sims.py sinovi) BIR XIL odamning qo'shni
# oynalari HAR DOIM >= 0.55 chiqdi — SPK_CHANGE_SIM'dan yetarlicha baland,
# chegara ishonchli bo'ldi. Shu sababli standart qiymatlar shu yerda
# 5-bandda YOZILGANIDAN farqli — lekin BIR XIL .env mexanizmi orqali
# istalgan vaqtda qayta sozlanishi mumkin.
SEG_WIN_S = float(os.environ.get("SEG_WIN_S", "2.5"))
SEG_HOP_S = float(os.environ.get("SEG_HOP_S", "1.25"))

# Ikki Silero-nutq segmenti orasidagi shuncha (yoki undan uzoqroq) jimlik
# bo'lsagina ular ALOHIDA "nutq oralig'i" (span) deb hisoblanadi (qarang
# get_speech_timestamps chaqiruvidagi `min_silence_duration_ms`) — bu
# FAQAT oynalar QAYERDA yaratilishiga ta'sir qiladi (bir span ICHIDA
# oyna jimlikni "kesib o'tmaydi"); o'ZI avtomatik qism-chegarasi EMAS
# (qarang pastdagi segment_turn'dagi 1-muammo izohi — buni ATAYLAB olib
# tashladik, aks holda bir kishining jumla orasidagi nafas-pauzasi ham
# "spiker almashdi" deb bo'lib yuborilardi).
SEG_SILENCE_GAP_S = float(os.environ.get("SEG_SILENCE_GAP_S", "0.6"))

# Qo'shni oynalar orasidagi cosine o'xshashlik shundan PAST bo'lsa — spiker
# o'zgargan deb hisoblanadi (qarang yuqoridagi "YONDASHUV TANLOVI"). Oddiy
# A1/A2 (bir xil odam, boshqa jumla) o'xshashligi odatda 0.6+ (speaker_diart.py
# SPK_ACCEPT=0.55 bilan taqqoslang), TURLI odam esa odatda <0.35 — 0.45 shu
# ikkisi orasidagi, xavfsiz chegara.
SPK_CHANGE_SIM = float(os.environ.get("SPK_CHANGE_SIM", "0.45"))

# Shu qiymatdan QISQA qism mustaqil STT/tarjima uchun yetarli emas (juda
# qisqa WAV'da Whisper/embedding ishonchsiz) — eng YAQIN qo'shnisiga
# qo'shib yuboriladi.
SUBSEG_MIN_S = float(os.environ.get("SUBSEG_MIN_S", "1.0"))

# 10-band (tezlik): bitta oynadan KO'P sig'maydigan ("oddiy, bir kishilik")
# bo'lakda butun oyna/embedding/klasterlash quvuri UMUMAN ishga tushmaydi —
# bitta bo'lak, bitta qism, qo'shimcha GPU chaqiruvisiz (0ms qo'shimcha
# xarajat). 2x SEG_WIN_S — kamida IKKITA (qisman ustma-ust tushmaydigan)
# oyna sig'ishi uchun minimal uzunlik.
FAST_PATH_MAX_DURATION_S = 2 * SEG_WIN_S


def _single_segment(wav: np.ndarray, sample_rate: int):
    duration_s = len(wav) / sample_rate
    return [{"start": 0.0, "end": duration_s, "wav": wav, "purity": 1.0}]


def _speech_spans(wav: np.ndarray) -> list[tuple[float, float]]:
    """Silero VAD orqali nutq oraliqlarini (soniyada, [start,end) juftlar,
    vaqt bo'yicha tartiblangan) qaytaradi. Xato/mavjud bo'lmasa — xavfsiz
    fallback: BUTUN wav bitta "nutq oralig'i" deb hisoblanadi (pastdagi
    oyna-mantiq o'zgarishsiz davom etadi, faqat Silero-chegara signali
    yo'qoladi — SPK_CHANGE_SIM baribir ishlayveradi)."""
    if not _VAD_AVAILABLE:
        return [(0.0, len(wav) / SAMPLE_RATE)]
    try:
        opts = VadOptions(min_silence_duration_ms=int(SEG_SILENCE_GAP_S * 1000))
        raw = get_speech_timestamps(wav, opts)
    except Exception as e:
        logger.info(f"[SEG] get_speech_timestamps failed: {e!r} — butun bo'lak bitta oraliq deb olinadi")
        return [(0.0, len(wav) / SAMPLE_RATE)]
    if not raw:
        return [(0.0, len(wav) / SAMPLE_RATE)]
    return [(seg["start"] / SAMPLE_RATE, seg["end"] / SAMPLE_RATE) for seg in raw]


def _windows_for_span(start_s: float, end_s: float) -> list[tuple[float, float]]:
    """Bitta nutq oralig'ini SEG_WIN_S uzunlikdagi, SEG_HOP_S qadamli
    oynalarga bo'ladi. Oxirgi oyna `end_s`ga TEKISLANADI (uzunligi
    SEG_WIN_S'dan qisqaroq bo'lishi mumkin) — agar natijada juda qisqa
    (< SEG_HOP_S) bo'lib qolsa, OLDINGI oyna bilan birlashtiriladi (shu
    oynaning chegarasi `end_s`ga cho'ziladi)."""
    span_len = end_s - start_s
    if span_len <= SEG_WIN_S:
        return [(start_s, end_s)]
    windows = []
    w_start = start_s
    while w_start < end_s:
        w_end = min(w_start + SEG_WIN_S, end_s)
        windows.append((w_start, w_end))
        if w_end >= end_s:
            break
        w_start += SEG_HOP_S
    # Oxirgi oyna juda qisqa bo'lib qolgan bo'lsa (masalan span uzunligi
    # SEG_HOP_S'ga teng bo'linmaydi) — oldingisiga qo'shib yuboramiz.
    if len(windows) >= 2 and (windows[-1][1] - windows[-1][0]) < SEG_HOP_S:
        prev_start, _ = windows[-2]
        _, last_end = windows[-1]
        windows[-2] = (prev_start, last_end)
        windows.pop()
    return windows


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def segment_turn(wav: np.ndarray, sample_rate: int, seq: int | None = None) -> list[dict]:
    """Bitta turn'ning XOM audiosini (kerak bo'lsa) bir nechta QISM
    (sub-segment)ga ajratadi. Qaytaradi: vaqt bo'yicha tartiblangan
    `[{"start": soniya, "end": soniya, "wav": np.ndarray, "purity": float}, ...]`
    — HAR DOIM kamida bitta element (hech qachon bo'sh ro'yxat emas).

    FAQAT bitta qism chiqsa (ODDIY holat — yakka spiker, pauzasiz, qisqa
    bo'lak) — MAVJUD yo'l (dialog.py'dagi process_turn) O'ZGARISHSIZ
    ishlaydi, chunki natija aynan bitta, butun-wav'ni qamrab oluvchi
    yozuv bo'ladi (qarang FAST_PATH_MAX_DURATION_S va _single_segment)."""
    t0 = time.perf_counter()
    duration_s = len(wav) / sample_rate

    # 10-band: tezkor yo'l — oyna/embedding quvuri UMUMAN ishga tushmaydi.
    if duration_s <= FAST_PATH_MAX_DURATION_S:
        return _single_segment(wav, sample_rate)

    # Silero VAD (get_speech_timestamps) 16kHz kutadi — dialog.py bu yerga
    # HAR DOIM SAMPLE_RATE=16000'dagi audio uzatadi (stt.py'dagi
    # SAMPLE_RATE bilan BIR XIL), shuning uchun qayta namunalash shart emas.
    spans = _speech_spans(wav)
    if len(spans) <= 1 and duration_s <= FAST_PATH_MAX_DURATION_S:
        return _single_segment(wav, sample_rate)

    # Lazy import — speaker_diart.py (og'ir model) FAQAT haqiqatan bir
    # nechta oyna kerak bo'lgandagina yuklanadi (import vaqtida EMAS).
    from app.models import speaker_diart

    # Har bir span ichida oynalar — (span_idx, win_start, win_end).
    all_windows: list[tuple[int, float, float]] = []
    for span_idx, (s_start, s_end) in enumerate(spans):
        for w_start, w_end in _windows_for_span(s_start, s_end):
            all_windows.append((span_idx, w_start, w_end))

    if len(all_windows) <= 1:
        return _single_segment(wav, sample_rate)

    # Har bir oyna uchun embedding (qarang modul docstring'idagi (A) vs
    # (B) — bu yerda speaker_diart.py'ning klaster HOLATIGA yozilmaydi,
    # FAQAT uning embedding modelidan (_embed) foydalaniladi).
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

    # Chegaralarni belgilash: FAQAT embedding o'xshashligi orqali (qarang
    # pastdagi izoh) — Silero span-chegarasi (jimlik) O'ZI chegara
    # YARATMAYDI, faqat qaysi oynalar yaralishiga (_windows_for_span)
    # ta'sir qiladi (bir kishi gap orasida nafas olib to'xtasa, bu HALI
    # spiker almashinuvi EMAS).
    #
    # REAL (debug_audio) audio bilan IKKI muammo topildi va tuzatildi:
    #   1. "Silero jimlik = avtomatik chegara" qoidasi (5-bandning
    #      so'zma-so'z o'qilishi) BIR KISHINING jumla orasidagi oddiy
    #      nafas-pauzasini ham "spiker almashdi" deb bo'lib yuborardi —
    #      "oddiy bo'lak o'zgarishsiz qolsin" talabiga zid. Shu sababli
    #      bu qoida OLIB TASHLANDI — pauza endi FAQAT oyna-joylashtirishga
    #      ta'sir qiladi (_windows_for_span), chegara qarori ESA har doim
    #      HAQIQIY ovoz-o'xshashligiga asoslanadi.
    #   2. Qo'shni-OYNA (i-1 vs i) solishtirish BIR XIL odamning
    #      monologida ham vaqti-vaqti bilan SPK_CHANGE_SIM'dan past
    #      chiqib, yolg'ON chegara hosil qilardi. Tuzatish: har bir oyna
    #      endi qo'shnisi bilan EMAS, joriy qism ICHIDAGI barcha
    #      oynalarning O'RTACHA (markaz) embeddingi bilan solishtiriladi —
    #      bitta oynadagi tasodifiy shovqin markazni unchalik siljitmaydi,
    #      HAQIQIY spiker almashinuvi esa (markazdan sezilarli farqli
    #      embedding) baribir aniq ko'rinadi (qarang SEG_WIN_S izohi
    #      yuqorida — oyna o'lchami ham shu kuzatish asosida oshirildi).
    groups: list[list[int]] = [[0]]
    centroid_sum = embeddings[0].copy() if embeddings[0] is not None else None
    centroid_count = 1 if embeddings[0] is not None else 0
    for i in range(1, len(all_windows)):
        is_boundary = False
        if embeddings[i] is not None and centroid_count > 0:
            centroid = centroid_sum / np.linalg.norm(centroid_sum)
            sim = _cosine_sim(centroid, embeddings[i])
            if sim < SPK_CHANGE_SIM:
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
        """Qismning "tozaligi" — oynalar o'zaro naqadar bir-biriga
        o'xshaydi. EMPIRIK TOPILMA (xuddi SPK_WIN_S/boundary-aniqlash
        kabi, qarang yuqoridagi izohlar): HAR BIR juftlikning (pairwise)
        ENG KICHIGini olish — hatto HAQIQIY bitta-spikerli qismda ham,
        bu loyihadagi embedding modeli 2-3 oyna narigi JUFTLIKLAR uchun
        ba'zan past (0.3-0.4) qiymat berardi, garchi qo'shni oynalar HAR
        DOIM barqaror yuqori (0.55+) bo'lsa ham — xom pairwise-min shu
        sababli haqiqiy toza qismlarni ham "iflos" deb belgilardi. Shu
        sababli bu yerda ham (boundary-aniqlashdagi kabi) MARKAZ (centroid)
        asosida: har bir oynaning MARKAZGA nisbatan o'xshashligi, ENG
        KICHIGI qaytariladi — markaz shovqinni o'rtachalagani uchun ancha
        barqaror baho beradi, lekin haqiqiy aralash (ikki ovozli) qismda
        baribir sezilarli past chiqadi (markaz ikki ovoz orasida "osilib"
        qoladi, HECH birига yaqin bo'lmaydi)."""
        embs = [embeddings[i] for i in idxs if embeddings[i] is not None]
        if len(embs) <= 1:
            return 1.0
        centroid = sum(embs)
        centroid = centroid / np.linalg.norm(centroid)
        return min(_cosine_sim(centroid, e) for e in embs)

    # SUBSEG_MIN_S'dan QISQA guruhlarni eng YAQIN qo'shnisiga qo'shish
    # (oldingisi bo'lsa — unga, aks holda keyingisiga).
    merged_any = True
    while merged_any:
        merged_any = False
        for i, idxs in enumerate(groups):
            g_start, g_end = group_bounds(idxs)
            if (g_end - g_start) >= SUBSEG_MIN_S or len(groups) == 1:
                continue
            if i > 0:
                # Oldingisiga qo'shamiz, joriy (endi ortiqcha) guruhni o'chiramiz.
                groups[i - 1] = groups[i - 1] + idxs
                del groups[i]
            else:
                # Birinchi guruh qisqa — keyingisiga qo'shamiz (uning
                # boshiga), joriy (endi ortiqcha, index 0) guruhni o'chiramiz.
                groups[i + 1] = idxs + groups[i + 1]
                del groups[i]
            merged_any = True
            break

    subsegments = []
    for idxs in groups:
        g_start, g_end = group_bounds(idxs)
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
