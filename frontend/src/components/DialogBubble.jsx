export default function DialogBubble({
  originalText,
  translatedText,
  onPlayAudio,
  isPlaying,
  speaker,
  conflict,
  audioError,
  onCorrectSpeaker,
}) {
  const isRight = speaker === "B";

  return (
    <div className={`flex ${isRight ? "justify-end" : "justify-start"} w-full`}>
      <div
        className={`max-w-xl rounded-3xl p-6 ${
          isRight ? "bg-purple-100/70" : "bg-white/70"
        } backdrop-blur`}
      >
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs text-gray-400">{speaker}</span>
          {/* Real testda topilgan 3-muammo: til va ovoz zid bo'lganda
              backend natijani hali ham til bo'yicha ko'rsatadi, lekin
              `conflict` bayrog'i bilan — foydalanuvchiga buni ANIQ
              ko'rsatish kerak, aks holda noto'g'ri spikerga ishonib
              qolishi mumkin. */}
          {conflict ? (
            <span
              title="Til va ovoz signallari bir-biriga zid chiqdi — spiker noaniq bo'lishi mumkin."
              className="text-xs text-amber-500"
            >
              ⚠ spiker noaniq
            </span>
          ) : null}
          {/* DIALOG_PLAN.md 4-bosqich: spiker noto'g'ri aniqlangan bo'lsa,
              foydalanuvchi shu yerdan to'g'rilaydi — backend qaytadan
              tarjima qiladi va ovoz markazini yangilaydi. */}
          <button
            onClick={onCorrectSpeaker}
            title="Spiker noto'g'ri aniqlandimi? Bu yerni bosib almashtiring."
            className="text-xs text-gray-400 hover:text-purple-500 transition-colors"
          >
            ⇄ spikerni almashtirish
          </button>
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
