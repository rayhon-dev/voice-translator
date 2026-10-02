import { useState, useRef, useEffect } from "react";

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "ru", label: "Русский" },
  { code: "ko", label: "한국어" },
];

export default function LanguageSelector({ selectedLang, onSelect, excludeLangs = [] }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    function handleClickOutside(e) {
      if (ref.current && !ref.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const current = LANGUAGES.find((l) => l.code === selectedLang);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/60 backdrop-blur text-gray-700 text-sm hover:bg-white/80 transition-colors"
      >
        {current?.label}
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="absolute top-full mt-2 left-0 bg-white/90 backdrop-blur rounded-2xl shadow-lg overflow-hidden min-w-[140px] z-20">
          {LANGUAGES.map((lang) => {
            // Suhbatdosh (masalan A) allaqachon shu tilni tanlagan bo'lsa —
            // Google Translate'ning "conversation mode"idagi kabi, ikkinchi
            // spiker bir xil tilni tanlay olmasligi uchun band qilib
            // (disabled) ko'rsatiladi, ro'yxatdan butunlay olib
            // tashlanmaydi — shunda foydalanuvchi "nega faqat 2 ta variant
            // bor" deb chalkashmaydi, balki NIMA UCHUN band ekanini ko'radi.
            const isExcluded = excludeLangs.includes(lang.code);
            return (
              <button
                key={lang.code}
                disabled={isExcluded}
                onClick={() => {
                  if (isExcluded) return;
                  onSelect(lang.code);
                  setOpen(false);
                }}
                className={`flex items-center justify-between gap-2 w-full text-left px-4 py-2 text-sm transition-colors ${
                  isExcluded
                    ? "text-gray-300 cursor-not-allowed"
                    : selectedLang === lang.code
                    ? "bg-purple-100 text-purple-700 hover:bg-purple-100"
                    : "text-gray-700 hover:bg-purple-100"
                }`}
              >
                {lang.label}
                {isExcluded && <span className="text-xs text-gray-300">band</span>}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}