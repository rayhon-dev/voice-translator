const LANGUAGES = [
  { code: "uz", label: "O'zbek" },
  { code: "ko", label: "한국어" },
  { code: "ru", label: "Русский" },
];

export default function LanguageSelector({ selectedLang, onSelect }) {
  return (
    <div className="flex gap-2">
      {LANGUAGES.map((lang) => (
        <button
          key={lang.code}
          onClick={() => onSelect(lang.code)}
          className={`px-4 py-2 rounded-full text-sm transition-colors
            ${
              selectedLang === lang.code
                ? "bg-purple-400 text-white"
                : "bg-white/60 text-gray-700"
            }`}
        >
          {lang.label}
        </button>
      ))}
    </div>
  );
}
