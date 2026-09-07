export default function DialogBubble({ originalText, translatedText, onPlayAudio, isPlaying, speaker }) {
  const isRight = speaker === "B";

  return (
    <div className={`flex ${isRight ? "justify-end" : "justify-start"} w-full`}>
      <div
        className={`max-w-xl rounded-3xl p-6 ${
          isRight ? "bg-purple-100/70" : "bg-white/70"
        } backdrop-blur`}
      >
        <span className="text-xs text-gray-400 block mb-2">{speaker}</span>
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
      </div>
    </div>
  );
}
