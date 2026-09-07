export default function MicButton({ isRecording, onClick }) {
  const bars = Array.from({ length: 24 });

  return (
    <div className="relative w-64 h-32 flex items-center justify-center">
      <div className="absolute inset-0 flex items-center justify-between px-2">
        {bars.map((_, i) => (
          <span
            key={i}
            className={`w-1 rounded-full bg-purple-300/70 ${
              isRecording ? "animate-wave-bar" : ""
            }`}
            style={{
              height: isRecording ? undefined : "6px",
              animationDelay: `${i * 0.07}s`,
            }}
          />
        ))}
      </div>

      <button
        onClick={onClick}
        className={`relative z-10 w-20 h-20 rounded-full flex items-center justify-center
          bg-gradient-to-br from-purple-300 to-pink-200 shadow-lg
          transition-transform active:scale-95
          ${isRecording ? "scale-105" : ""}`}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="white"
          className="w-7 h-7"
        >
          <path d="M12 14a3 3 0 003-3V5a3 3 0 10-6 0v6a3 3 0 003 3z" />
          <path d="M19 11a1 1 0 10-2 0 5 5 0 01-10 0 1 1 0 10-2 0 7 7 0 006 6.93V21a1 1 0 102 0v-3.07A7 7 0 0019 11z" />
        </svg>
      </button>
    </div>
  );
}