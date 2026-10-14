"""Qisqa, qo'lda ishga tushiriladigan tekshiruv — _is_phantom_phrase() haqiqatan
ham fuzzy-ratio (substring EMAS) bilan ishlayotganini tasdiqlaydi.

pytest emas (loyihada hali test infratuzilmasi yo'q) — backend venv'ida
(torch/faster-whisper o'rnatilgan joyda) ishga tushiriladi:

    python backend/tools/test_phantom_phrase.py

Haqiqiy app.models.stt._is_phantom_phrase()ni import qiladi (nusxa emas) —
shunda mantiq ikki joyda yozilib, vaqt o'tishi bilan bir-biridan chetlashib
qolmaydi.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.stt import _is_phantom_phrase  # noqa: E402

# (text, expected_is_phantom) — DIALOG_PLAN.md: real testda topilgan
# false-positive'lar ("субтитры" so'zini tabiiy ishlatgan haqiqiy gaplar)
# + klassik Whisper subtitle-boilerplate gallyutsinatsiyalari.
CASES = [
    ("Для вас есть субтитры, как обычно, здесь...", False),
    ("Где можно включить субтитры", False),
    ("И еще есть субтитры, кажется, здесь или здесь. Я все время забываю, где кнопка.", False),
    ("На YouTube, где можно включить субтитры. [шум в зале]", False),
    ("Субтитры создавал DimaTorzok", True),
    ("субтитры создавал dimatorzok", True),
    ("Редактор субтитров А.Семкин", True),
    ("Корректор А.Egorova", True),
    ("Thank you for watching!", True),
    ("Hello, how are you today?", False),
    ("", False),
]


def main() -> None:
    failures = 0
    for text, expected in CASES:
        actual = _is_phantom_phrase(text)
        ok = actual == expected
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] phantom={actual} (expected={expected})  text={text!r}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} o'tdi.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
