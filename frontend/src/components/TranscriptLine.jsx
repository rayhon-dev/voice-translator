export default function TranscriptLine({ originalText, translatedText, onPlayAudio, isPlaying, speaker }) {
  return (
    <div className="grid grid-cols-2 gap-6 w-full">
      <div className="bg-white/70 backdrop-blur rounded-3xl p-8 min-h-[160px] flex flex-col justify-center">
        {speaker && (
          <span className="text-sm text-gray-400 block mb-2">{speaker}</span>
        )}
        <p className="text-2xl text-gray-800 leading-relaxed">{originalText}</p>
      </div>

      <div className="bg-purple-100/70 backdrop-blur rounded-3xl p-8 min-h-[160px] flex items-start gap-4">
        <button
          onClick={onPlayAudio}
          className="shrink-0 w-12 h-12 rounded-full bg-purple-400 text-white
            flex items-center justify-center hover:scale-105 transition-transform mt-1"
        >
          {isPlaying ? (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" className="w-5 h-5">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" className="w-6 h-6">
              <path d="M6 4l12 8-12 8V4z" />
            </svg>
          )}
        </button>
        <p className="text-2xl text-gray-800 leading-relaxed">{translatedText}</p>
      </div>
    </div>
  );
}