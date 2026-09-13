import { useState, useRef, useEffect } from "react";

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "uz", label: "O'zbek" },
  { code: "ko", label: "한국어" },
  { code: "ru", label: "Русский" },
];

export default function LanguageSelector({ selectedLang, onSelect }) {
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
          {LANGUAGES.map((lang) => (
            <button
              key={lang.code}
              onClick={() => {
                onSelect(lang.code);
                setOpen(false);
              }}
              className={`block w-full text-left px-4 py-2 text-sm hover:bg-purple-100 transition-colors ${
                selectedLang === lang.code ? "bg-purple-100 text-purple-700" : "text-gray-700"
              }`}
            >
              {lang.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}