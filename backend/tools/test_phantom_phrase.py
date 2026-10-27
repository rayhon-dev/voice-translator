import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.stt import _is_phantom_phrase  # noqa: E402

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
