import { useEffect, useRef, useState } from "react";

// Spiker endi A/B emas, cheklanmagan sonli butun son ID (speaker_diart.py'dagi
// onlayn klasterlash) — menyu/yorliq/ism uchun kerakli ma'lumot (label/rang/
// ism) DialogSessionView.jsx'dan tayyor holda keladi. "Men" tushunchasi
// YO'Q — joylashuv FAQAT spiker ID'ning juft-toqligiga bog'liq (`isRight`).
export default function DialogBubble({
  originalText,
  translatedText,
  sourceLang,
  onPlayAudio,
  isPlaying,
  speakerName,
  isRight,
  avatarLetter,
  avatarClass,
  nameColorClass,
  borderColorClass, // "border-X-400" — chap chizig'i (4px) spiker rangida
  lightBgClass, // "bg-X-50" — bubble foni spiker rangida (juda ochiq)
  grouped,
  uncertain,
  knownSpeakers, // [{ id, name, dotClass }] — joriy bubble spikeri ISTISNO qilingan
  audioError,
  onSelectSpeaker, // (targetSpeaker: number | "new") => void
  onRename, // (newName: string | null) => void — null = default ismga qaytarish
  onDelete,
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [isEditingName, setIsEditingName] = useState(false);
  const [draftName, setDraftName] = useState(speakerName);
  const menuRef = useRef(null);
  const nameInputRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (isEditingName) {
      nameInputRef.current?.focus();
      nameInputRef.current?.select();
    }
  }, [isEditingName]);

  const startEditingName = () => {
    setDraftName(speakerName);
    setIsEditingName(true);
  };

  const commitNameEdit = () => {
    const trimmed = draftName.trim();
    // Bo'sh ism saqlanmaydi — default ("N-spiker") qaytadi.
    onRename(trimmed ? trimmed : null);
    setIsEditingName(false);
  };

  const cancelNameEdit = () => {
    setIsEditingName(false);
  };

  const avatar = (
    <div
      className={`w-8 h-8 shrink-0 self-end rounded-full ${avatarClass} flex items-center justify-center text-white text-sm font-semibold uppercase`}
    >
      {avatarLetter}
    </div>
  );

  return (
    <div
      className={`flex items-end gap-2 w-full ${isRight ? "justify-end" : "justify-start"} ${grouped ? "mt-1" : "mt-4"}`}
    >
      {!isRight ? avatar : null}

      <div
        className={`max-w-[80%] sm:max-w-xl rounded-3xl px-4 py-3 sm:p-5 backdrop-blur border-l-4 ${borderColorClass} ${lightBgClass} text-gray-700`}
      >
        <div className="relative flex items-center justify-between gap-2 mb-2" ref={menuRef}>
          <div className="flex items-center gap-1 min-w-0">
            {isEditingName ? (
              <input
                ref={nameInputRef}
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                onBlur={commitNameEdit}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitNameEdit();
                  else if (e.key === "Escape") cancelNameEdit();
                }}
                className={`text-xs font-semibold bg-transparent border-b outline-none ${nameColorClass}`}
              />
            ) : (
              <button
                onClick={startEditingName}
                className={`flex items-center gap-1 text-xs font-semibold hover:underline truncate ${nameColorClass}`}
                title="Ismni tahrirlash"
              >
                <span className="truncate">{speakerName}</span>
                <span className="text-[10px] opacity-60 shrink-0">✏️</span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {uncertain ? (
              <span
                title="Spiker ishonchi past — kerak bo'lsa qo'lda to'g'rilang."
                className="text-xs font-semibold text-amber-500"
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
          </div>

          {menuOpen && (
            <div className="absolute top-full left-0 mt-1 bg-white/95 backdrop-blur rounded-2xl shadow-lg overflow-hidden min-w-[180px] z-20 text-gray-700">
              {knownSpeakers.map((s) => (
                <button
                  key={s.id}
                  onClick={() => {
                    onSelectSpeaker(s.id);
                    setMenuOpen(false);
                  }}
                  className="flex items-center gap-2 w-full text-left px-4 py-2 text-sm hover:bg-purple-100 transition-colors"
                >
                  <span className={`w-2.5 h-2.5 rounded-full ${s.dotClass}`} />
                  {s.name}
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

        <div className="flex items-start gap-2 mb-3">
          <p className="text-base sm:text-lg">{originalText}</p>
          {sourceLang ? (
            <span className="shrink-0 mt-1 text-[10px] font-bold uppercase rounded px-1 border border-gray-300 text-gray-400">
              {sourceLang}
            </span>
          ) : null}
        </div>

        <div className="flex items-start gap-3 pt-3 border-t border-gray-200/50">
          <button
            onClick={onPlayAudio}
            className="shrink-0 w-9 h-9 rounded-full text-white flex items-center justify-center hover:scale-105 transition-transform bg-purple-400"
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
          <p className="text-base sm:text-lg font-medium text-gray-800">{translatedText}</p>
        </div>
        {audioError ? (
          <p className="text-xs mt-1 text-red-500">Audio ijro etilmadi — qayta urinib ko'ring.</p>
        ) : null}
      </div>

      {isRight ? avatar : null}
    </div>
  );
}
