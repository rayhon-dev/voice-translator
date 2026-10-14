import { useEffect, useRef, useState } from "react";

// Spiker endi A/B emas, cheklanmagan sonli butun son ID (speaker_diart.py'dagi
// onlayn klasterlash) — menyu/yorliq uchun kerakli ma'lumot (label/rang)
// DialogSessionView.jsx'dan tayyor holda keladi (qarang SPEAKER_PALETTE).
export default function DialogBubble({
  originalText,
  translatedText,
  onPlayAudio,
  isPlaying,
  speaker,
  speakerLabel,
  badgeClass,
  uncertain,
  knownSpeakers, // [{ id, label, dotClass }]
  audioError,
  onSelectSpeaker, // (targetSpeaker: number | "new") => void
  onDelete,
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="flex justify-start w-full">
      <div className="max-w-xl rounded-3xl p-6 bg-white/70 backdrop-blur">
        <div className="relative flex items-center gap-2 mb-2" ref={menuRef}>
          <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${badgeClass}`}>
            {speakerLabel}
          </span>
          {/* Spiker aniqlash ishonchi past bo'lsa (noaniq o'xshashlik band,
              juda qisqa turn, yoki hali tasdiqlanmagan yangi spiker
              nomzodi) — foydalanuvchiga buni ko'rsatish kerak, aks holda
              noto'g'ri spikerga ishonib qolishi mumkin. */}
          {uncertain ? (
            <span
              title="Spiker ishonchi past — kerak bo'lsa qo'lda to'g'rilang."
              className="text-xs text-amber-500 font-semibold"
            >
              ?
            </span>
          ) : null}

          <button
            onClick={() => setMenuOpen((o) => !o)}
            title="Spikerni almashtirish"
            className="text-xs text-gray-400 hover:text-purple-500 transition-colors"
          >
            ⇄
          </button>
          <button
            onClick={onDelete}
            title="Bu gapni o'chirish"
            className="text-xs text-gray-400 hover:text-red-500 transition-colors"
          >
            🗑
          </button>

          {menuOpen && (
            <div className="absolute top-full left-0 mt-1 bg-white/95 backdrop-blur rounded-2xl shadow-lg overflow-hidden min-w-[160px] z-20">
              {knownSpeakers
                .filter((s) => s.id !== speaker)
                .map((s) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      onSelectSpeaker(s.id);
                      setMenuOpen(false);
                    }}
                    className="flex items-center gap-2 w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-purple-100 transition-colors"
                  >
                    <span className={`w-2.5 h-2.5 rounded-full ${s.dotClass}`} />
                    {s.label}
                  </button>
                ))}
              <button
                onClick={() => {
                  onSelectSpeaker("new");
                  setMenuOpen(false);
                }}
                className="w-full text-left px-4 py-2 text-sm text-gray-500 hover:bg-purple-100 transition-colors border-t border-gray-100"
              >
                + Yangi spiker
              </button>
            </div>
          )}
        </div>
        <p className="text-lg text-gray-700 mb-3">{originalText}</p>
        <div className="flex items-start gap-3 pt-3 border-t border-gray-200/50">
          <button
            onClick={onPlayAudio}
            className="shrink-0 w-9 h-9 rounded-full bg-purple-400 text-white
              flex items-center justify-center hover:scale-105 transition-transform"
          >
            {isPlaying ? (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" className="w-4 h-4">
                <rect x="6" y="4" width="4" height="16" rx="1" />
                <rect x="14" y="4" width="4" height="16" rx="1" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" className="w-4 h-4">
                <path d="M6 4l12 8-12 8V4z" />
              </svg>
            )}
          </button>
          <p className="text-lg text-gray-800 font-medium">{translatedText}</p>
        </div>
        {audioError ? (
          <p className="text-xs text-red-500 mt-1">
            Audio ijro etilmadi — qayta urinib ko'ring.
          </p>
        ) : null}
      </div>
    </div>
  );
}
