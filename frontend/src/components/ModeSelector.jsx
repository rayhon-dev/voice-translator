export default function ModeSelector({ onSelectMode }) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-10 px-6">
      <div className="text-center">
        <h1 className="text-5xl font-serif text-gray-800">
          Voice <span className="italic">Translator</span>
        </h1>
        <p className="mt-3 text-gray-500 text-sm">
          Ingizcha gapiring, istalgan tilga tarjima qiling
        </p>
      </div>

      <div className="flex gap-4">
        <button
          onClick={() => onSelectMode("solo")}
          className="px-8 py-4 rounded-full bg-white/70 backdrop-blur shadow-md
            text-gray-800 font-medium hover:scale-105 transition-transform"
        >
          Solo
        </button>
        <button
          onClick={() => onSelectMode("dialog")}
          className="px-8 py-4 rounded-full bg-purple-400 shadow-md
            text-white font-medium hover:scale-105 transition-transform"
        >
          Dialog
        </button>
      </div>
    </div>
  );
}
