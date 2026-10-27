import argparse
import glob
import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import soundfile as sf  # noqa: E402

from app.config import DEBUG_AUDIO_DIR, MIN_TURN_SECONDS  # noqa: E402
from app.models.speaker_diart import create_session, identify_speaker  # noqa: E402
from app.models.stt import transcribe_audio  # noqa: E402

try:
    import jiwer
except ImportError:
    jiwer = None


LABELS_FILENAME = "labels.json"
REQUIRED_LABEL_FIELDS = ("speaker", "lang", "text")
EXPECTED_SAMPLE_RATE = 16000


def _load_auto_detected_draft(debug_dir: str, basename: str) -> dict | None:
    sidecar = os.path.join(debug_dir, os.path.splitext(basename)[0] + ".json")
    if not os.path.exists(sidecar):
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_labels_template(wav_paths: list[str], debug_dir: str) -> None:
    template = {}
    prefilled = 0
    for wav_path in wav_paths:
        basename = os.path.basename(wav_path)
        draft = _load_auto_detected_draft(debug_dir, basename)
        if draft is not None:
            template[basename] = {
                "speaker": draft.get("speaker"),
                "lang": draft.get("lang"),
                "text": draft.get("text"),
            }
            prefilled += 1
        else:
            template[basename] = {"speaker": None, "lang": None, "text": None}

    labels_path = os.path.join(debug_dir, LABELS_FILENAME)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"\n[{debug_dir}] {LABELS_FILENAME} topilmadi — {len(template)} fayl uchun shablon yaratildi:")
    print(f"  {labels_path}")
    if prefilled:
        print(
            f"  ({prefilled} tasi SAVE_DEBUG_AUDIO yozib qo'ygan avtomatik .json'lardan OLDINDAN\n"
            f"  to'ldirildi — bular backend O'ZI shu turnda nima deb o'ylagani, HAQIQIY to'g'ri\n"
            f"  javob EMAS. Albatta tinglab tekshiring!)"
        )
    print(
        "  Har bir yozuvni ochib, audio faylni tinglab, haqiqatda KIM, QAYSI TILDA va\n"
        "  NIMA DEGANINI qo'lda to'g'rilang:\n"
        '    "speaker": "A" yoki "B"\n'
        '    "lang":    "en" | "ru" | "ko"\n'
        '    "text":    audio\'da aytilgan ANIQ matn (WER shu bilan solishtiriladi)\n'
        "  To'ldirib bo'lgach, shu skriptni QAYTA ishga tushiring."
    )


def run_session(debug_dir: str):
    wav_paths = sorted(glob.glob(os.path.join(debug_dir, "*.wav")))
    if not wav_paths:
        print(f"[{debug_dir}] .wav fayl topilmadi — o'tkazib yuborildi.")
        return None

    labels_path = os.path.join(debug_dir, LABELS_FILENAME)
    if not os.path.exists(labels_path):
        build_labels_template(wav_paths, debug_dir)
        return "template"

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    unlabeled = [os.path.basename(p) for p in wav_paths if os.path.basename(p) not in labels]
    if unlabeled:
        print(
            f"[{debug_dir}] OGOHLANTIRISH: {len(unlabeled)} fayl labels.json'da yo'q "
            f"(yangi suhbat qo'shilgan bo'lishi mumkin) — baholashda inobatga olinmaydi, "
            f"lekin klaster holatini saqlash uchun pipeline orqali baribir o'tkaziladi."
        )

    speaker_langs: dict[str, str] = {}
    for wav_path in wav_paths:
        label = labels.get(os.path.basename(wav_path))
        if not label:
            continue
        sp, lang = label.get("speaker"), label.get("lang")
        if sp and lang and sp not in speaker_langs:
            speaker_langs[sp] = lang

    if len(speaker_langs) < 2:
        print(
            f"[{debug_dir}] OGOHLANTIRISH: faqat {len(speaker_langs)} spiker uchun til "
            f"aniqlandi (speaker_langs={speaker_langs}) — identify_speaker ikkinchi spiker "
            f"uchun til signalini to'g'ri hisoblay olmasligi mumkin (p_lang ikkisiga ham "
            f"0.5/NEUTRAL bo'lib qoladi). labels.json'da ikkala spiker (A va B) uchun ham "
            f"kamida bitta TO'LIQ yozuv borligiga ishonch hosil qiling."
        )

    session_langs = [speaker_langs.get("A"), speaker_langs.get("B")]

    session = create_session()
    rows = []

    for wav_path in wav_paths:
        basename = os.path.basename(wav_path)
        label = labels.get(basename)
        row = {"file": basename, "label": label}

        try:
            wav, sr = sf.read(wav_path, dtype="float32")
        except Exception as e:
            print(f"[{debug_dir}] {basename}: o'qib bo'lmadi ({e!r}) — o'tkazib yuborildi.")
            continue
        if sr != EXPECTED_SAMPLE_RATE:
            print(
                f"[{debug_dir}] {basename}: sample_rate={sr} (kutilgan {EXPECTED_SAMPLE_RATE}) "
                f"— natija ishonchsiz bo'lishi mumkin."
            )

        duration_s = len(wav) / sr
        if duration_s < MIN_TURN_SECONDS:
            row.update(predicted_speaker=None, skip_reason="too_short", text=None, lang=None)
            rows.append(row)
            continue

        text, detected_lang, lang_prob, unsupported_language, _stt_diag = transcribe_audio(wav_path, None, session_langs)
        text = (text or "").strip()

        if unsupported_language:
            row.update(predicted_speaker=None, skip_reason="unsupported_language", text=text, lang=detected_lang)
            rows.append(row)
            continue

        if not text:
            row.update(predicted_speaker=None, skip_reason="empty_transcription", text=text, lang=detected_lang)
            rows.append(row)
            continue

        speaker, score_debug = identify_speaker(
            session, wav, sr, speaker_langs, detected_lang, lang_prob
        )
        skip_reason = "ambiguous_speaker" if speaker == "ambiguous" else None
        predicted_speaker = None if speaker == "ambiguous" else speaker
        row.update(
            predicted_speaker=predicted_speaker,
            skip_reason=skip_reason,
            text=text,
            lang=detected_lang,
            score_debug=score_debug,
        )
        rows.append(row)

    return rows


def _label_complete(label: dict | None) -> bool:
    return bool(label) and all(label.get(k) not in (None, "") for k in REQUIRED_LABEL_FIELDS)


def _file_wer(reference: str, hypothesis: str) -> float | None:
    if jiwer is None or not reference.strip():
        return None
    try:
        return jiwer.wer(reference, hypothesis) * 100
    except Exception:
        return None


def print_report(session_results: list[tuple[str, list[dict]]]) -> None:
    all_scored = []

    for debug_dir, rows in session_results:
        print(f"\n{'=' * 100}")
        print(f"Sessiya: {debug_dir}  ({len(rows)} fayl)")
        print(f"{'=' * 100}")
        header = f"{'fayl':<45} {'haqiqiy':<8} {'bashorat':<9} {'natija':<8} {'sabab/til':<22} {'WER':>6}"
        print(header)
        print("-" * len(header))

        for row in rows:
            label = row["label"]
            complete = _label_complete(label)
            pred_speaker = row["predicted_speaker"]
            true_speaker = label.get("speaker") if label else None

            if not complete:
                verdict = "label yo'q" if not label else "to'liq emas"
                print(f"{row['file']:<45} {'-':<8} {str(pred_speaker):<9} {verdict:<8}")
                continue

            row["_scored"] = True
            all_scored.append(row)

            match = pred_speaker == true_speaker
            verdict = "OK" if match else "XATO"
            reason = row["skip_reason"] or row.get("lang") or "-"
            wer = _file_wer(label["text"], row["text"] or "")
            wer_str = f"{wer:.0f}%" if wer is not None else "-"
            print(
                f"{row['file']:<45} {true_speaker:<8} {str(pred_speaker):<9} {verdict:<8} "
                f"{str(reason):<22} {wer_str:>6}"
            )

    print(f"\n{'=' * 100}")
    print("UMUMIY XULOSA")
    print(f"{'=' * 100}")

    if not all_scored:
        print(
            "Hali hech qanday fayl TO'LIQ label'langan emas (speaker+lang+text hammasi "
            "to'ldirilgan bo'lishi kerak) — labels.json'ni to'ldirib qayta ishga tushiring."
        )
        return

    total = len(all_scored)
    skipped = [r for r in all_scored if r["predicted_speaker"] is None]
    survived = [r for r in all_scored if r["predicted_speaker"] is not None]
    skip_rate = len(skipped) / total * 100

    print(f"Jami label'langan turnlar:        {total}")
    print(f"Tashlab yuborilgan turnlar:        {len(skipped)}/{total} = {skip_rate:.1f}%")

    if survived:
        speaker_correct = sum(1 for r in survived if r["predicted_speaker"] == r["label"]["speaker"])
        speaker_acc = speaker_correct / len(survived) * 100
        print(
            f"Spiker aniqligi (faqat o'tgan turnlar): {speaker_correct}/{len(survived)} = {speaker_acc:.1f}%"
        )
    else:
        print("Spiker aniqligi: hisoblab bo'lmadi (barcha turnlar tashlab yuborilgan)")

    if jiwer is None:
        print("WER: jiwer o'rnatilmagan — `pip install jiwer` qilib qayta ishga tushiring.")
    else:
        refs = [r["label"]["text"] for r in all_scored]
        hyps = [r["text"] or "" for r in all_scored]
        try:
            overall_wer = jiwer.wer(refs, hyps) * 100
            print(f"WER (barcha label'langan turnlar bo'yicha): {overall_wer:.1f}%")
            print(
                "  (Eslatma: WER so'zlarni probel bo'yicha ajratadi — koreyscha matn uchun "
                "bu metrika kam ishonchli, faqat taqqoslash uchun ishlatilsin.)"
            )
        except Exception as e:
            print(f"WER hisoblanmadi: {e!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        default=[DEBUG_AUDIO_DIR],
        help=f"Baholanadigan debug-audio papka(lar) (default: {DEBUG_AUDIO_DIR})",
    )
    args = parser.parse_args()

    session_results = []
    template_created = False

    for debug_dir in args.dirs:
        result = run_session(debug_dir)
        if result == "template":
            template_created = True
        elif result is not None:
            session_results.append((debug_dir, result))

    if template_created:
        print(
            "\nKamida bitta papka uchun labels.json ENDIGINA yaratildi — avval uni "
            "to'ldirib (yoki tuzatib), keyin skriptni QAYTA ishga tushiring."
        )

    if session_results:
        print_report(session_results)
    elif not template_created:
        print("Baholash uchun hech qanday .wav fayl topilmadi.")


if __name__ == "__main__":
    main()
