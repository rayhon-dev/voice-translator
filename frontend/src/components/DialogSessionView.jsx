import { useState } from "react";
import { useDialogSession } from "../hooks/useDialogSession";
import { usePlayAudio } from "../hooks/usePlayAudio";
import { speakText } from "../services/api";
import LanguageSelector from "./LanguageSelector";
import DialogBubble from "./DialogBubble";

const SPEAKER_ORDINAL = { A: "1-kishi", B: "2-kishi" };

export default function DialogSessionView({ onBack }) {
  const {
    phase,
    currentSpeaker,
    messages,
    error,
    notice,
    isSpeaking,
    startDialog,
    selectLanguage,
    stopDialog,
    retryEnrollment,
  } = useDialogSession();

  const [audioCache, setAudioCache] = useState({});
  const { playingId, togglePlay } = usePlayAudio();

  const handlePlayAudio = async (index, text, lang) => {
    try {
      let blob = audioCache[index];
      if (!blob) {
        blob = await speakText(text, lang);
        setAudioCache((prev) => ({ ...prev, [index]: blob }));
      }
      togglePlay(index, blob);
    } catch (err) {
      console.error("[DialogSessionView] failed to play audio", err);
    }
  };

  const speakerLabel = SPEAKER_ORDINAL[currentSpeaker] || "Foydalanuvchi";

  if (phase === "idle") {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-6 px-6">
        <button
          onClick={onBack}
          className="absolute top-8 left-10 text-gray-500 text-sm"
        >
          ← Orqaga
        </button>
        <p className="text-gray-600 text-lg">Ikki kishilik suhbat rejimi</p>
        <button
          onClick={startDialog}
          className="px-8 py-4 rounded-full bg-purple-400 shadow-md text-white font-medium hover:scale-105 transition-transform"
        >
          Suhbatni boshlash
        </button>
      </div>
    );
  }

  if (phase === "await_language") {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-6 px-6">
        <p className="text-gray-700 text-lg">{speakerLabel}, tilingizni tanlang</p>
        <LanguageSelector selectedLang="en" onSelect={selectLanguage} />
      </div>
    );
  }

  if (phase === "recording_enroll") {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-6 px-6">
        <p className="text-gray-700 text-lg animate-pulse">
          {speakerLabel}, gapiring...
        </p>
      </div>
    );
  }

  if (phase === "enroll_failed") {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-6 px-6">
        <p className="text-red-500 text-lg text-center max-w-md">
          {error || "Enrollment muvaffaqiyatsiz tugadi."}
        </p>
        <button
          onClick={retryEnrollment}
          className="px-8 py-4 rounded-full bg-purple-400 shadow-md text-white font-medium hover:scale-105 transition-transform"
        >
          Qayta urinish
        </button>
      </div>
    );
  }

  if (phase === "enroll_wait") {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-6 px-6">
        <p className="text-gray-500 text-lg animate-pulse">Ishlanmoqda...</p>
      </div>
    );
  }

  if (phase === "error") {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-6 px-6">
        <button
          onClick={onBack}
          className="absolute top-8 left-10 text-gray-500 text-sm"
        >
          ← Orqaga
        </button>
        <p className="text-red-500 text-lg text-center max-w-md">
          {error || "Xatolik yuz berdi."}
        </p>
        <button
          onClick={startDialog}
          className="px-8 py-4 rounded-full bg-purple-400 shadow-md text-white font-medium hover:scale-105 transition-transform"
        >
          Qayta urinish
        </button>
      </div>
    );
  }

  if (phase === "ended") {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-6 px-6">
        <button
          onClick={onBack}
          className="absolute top-8 left-10 text-gray-500 text-sm"
        >
          ← Orqaga
        </button>
        <p className="text-gray-600 text-lg">Suhbat tugadi</p>
        <button
          onClick={startDialog}
          className="px-8 py-4 rounded-full bg-purple-400 shadow-md text-white font-medium hover:scale-105 transition-transform"
        >
          Yangi suhbat boshlash
        </button>
      </div>
    );
  }

  // phase === "listening"
  return (
    <div className="min-h-screen flex flex-col items-center px-10 py-8 gap-8">
      <div className="w-full flex justify-between items-center">
        <button onClick={onBack} className="text-gray-500 text-sm">
          ← Orqaga
        </button>
        <button
          onClick={stopDialog}
          className="px-5 py-2 rounded-full bg-white/60 backdrop-blur text-gray-700 text-sm hover:bg-white/80 transition-colors"
        >
          Suhbatni tugatish
        </button>
      </div>

      {notice ? (
        <p className="text-amber-600 text-sm">{notice}</p>
      ) : isSpeaking ? (
        <p className="text-purple-500 text-sm animate-pulse">🎙️ Gapirmoqda...</p>
      ) : (
        <p className="text-gray-400 text-sm">Tinglanmoqda — bemalol gaplashavering</p>
      )}

      <div className="w-full max-w-6xl flex flex-col gap-6 flex-1">
        {messages.map((msg, index) => (
          <DialogBubble
            key={index}
            originalText={msg.originalText}
            translatedText={msg.translation}
            speaker={msg.speaker}
            isPlaying={playingId === index}
            onPlayAudio={() => handlePlayAudio(index, msg.translation, msg.targetLang)}
          />
        ))}
      </div>
    </div>
  );
}
