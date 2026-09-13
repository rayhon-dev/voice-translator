import { useState } from "react";
import MicButton from "./MicButton";
import LanguageSelector from "./LanguageSelector";
import TranscriptLine from "./TranscriptLine";
import DialogSessionView from "./DialogSessionView";
import { usePlayAudio } from "../hooks/usePlayAudio";
import { useLiveTranscription } from "../hooks/useLiveTranscription";
import { translateText, speakText } from "../services/api";

export default function ConversationView({ mode, onBack }) {
  // `mode` is fixed for the lifetime of a mounted instance — App.jsx only
  // renders ConversationView after a mode is picked and unmounts it on
  // "back", so it never changes across re-renders of the same instance.
  // This early return (before any hooks below) is therefore safe, even
  // though the lint rule can't verify that invariant statically.
  /* eslint-disable react-hooks/rules-of-hooks */
  if (mode === "dialog") {
    return <DialogSessionView onBack={onBack} />;
  }

  const [sourceLang, setSourceLang] = useState("en");
  const [targetLang, setTargetLang] = useState("uz");
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [audioCache, setAudioCache] = useState({});

  const { liveText, start, stop } = useLiveTranscription();
  const { playingId, togglePlay } = usePlayAudio();
  /* eslint-enable react-hooks/rules-of-hooks */

  const handleMicClick = async () => {
    if (!isRecording) {
      try {
        await start(sourceLang);
        setIsRecording(true);
      } catch (err) {
        console.error("[ConversationView] failed to start recording", err);
        setIsRecording(false);
      }
      return;
    }

    setIsRecording(false);
    setIsProcessing(true);

    const flowStart = performance.now();
    const ms = (from) => `${(performance.now() - from).toFixed(0)}ms`;

    try {
      const stopStart = performance.now();
      const { text } = await stop();
      console.log(`[latency] stop() round trip: ${ms(stopStart)}`);
      const originalText = (text || "").trim();

      if (!originalText) {
        console.warn(
          "[ConversationView] transcription was empty — skipping translation"
        );
        return;
      }

      let translation;
      try {
        const trStart = performance.now();
        translation = await translateText(originalText, targetLang, sourceLang);
        console.log(`[latency] translateText(): ${ms(trStart)}`);
      } catch (err) {
        console.error("[ConversationView] translateText failed", err);
        return;
      }

      setMessages((prev) => [...prev, { originalText, translation, speaker: null }]);
      console.log(
        `[latency] TOTAL stop -> message on screen: ${ms(flowStart)}`
      );
    } catch (err) {
      console.error(
        "[ConversationView] unexpected error while processing recording",
        err
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePlayAudio = async (index, text) => {
    let blob = audioCache[index];
    if (!blob) {
      blob = await speakText(text, targetLang);
      setAudioCache((prev) => ({ ...prev, [index]: blob }));
    }
    togglePlay(index, blob);
  };

  return (
    <div className="min-h-screen flex flex-col items-center px-10 py-8 gap-8">
      <div className="w-full flex justify-between items-center">
        <button onClick={onBack} className="text-gray-500 text-sm">
          ← Orqaga
        </button>

        <div className="flex items-center gap-3">
          <LanguageSelector selectedLang={sourceLang} onSelect={setSourceLang} />

          <button
            onClick={() => {
              setSourceLang(targetLang);
              setTargetLang(sourceLang);
            }}
            className="w-8 h-8 rounded-full bg-white/60 backdrop-blur flex items-center justify-center hover:bg-white/80 transition-colors"
            aria-label="Tillarni almashtirish"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4 text-gray-600">
              <path strokeLinecap="round" strokeLinejoin="round" d="M7 16V4m0 0L3 8m4-4l4 4m6 4v12m0 0l4-4m-4 4l-4-4" />
            </svg>
          </button>

          <LanguageSelector selectedLang={targetLang} onSelect={setTargetLang} />
        </div>
      </div>

      <div className="w-full max-w-6xl flex flex-col gap-6 flex-1">
        {messages.map((msg, index) => (
          <TranscriptLine
            key={index}
            originalText={msg.originalText}
            translatedText={msg.translation}
            speaker={msg.speaker}
            isPlaying={playingId === index}
            onPlayAudio={() => handlePlayAudio(index, msg.translation)}
          />
        ))}

        {isRecording && (
          <div className="grid grid-cols-2 gap-6 w-full">
            <div className="bg-white/50 backdrop-blur rounded-3xl p-8 min-h-[160px] flex items-center">
              <p className="text-2xl text-gray-700">
                {liveText || "Tinglanmoqda..."}
              </p>
            </div>
            <div className="bg-purple-100/30 backdrop-blur rounded-3xl p-8 min-h-[160px] flex items-center justify-center">
              <span className="text-gray-400 text-sm">To'xtatganingizda tarjima chiqadi</span>
            </div>
          </div>
        )}

        {isProcessing && (
          <div className="grid grid-cols-2 gap-6 w-full animate-pulse">
            <div className="bg-white/50 backdrop-blur rounded-3xl p-8 min-h-[160px] flex items-center justify-center">
              <span className="text-gray-400">Yakunlanmoqda...</span>
            </div>
            <div className="bg-purple-100/50 backdrop-blur rounded-3xl p-8 min-h-[160px] flex items-center justify-center">
              <span className="text-gray-400">Tarjima qilinmoqda...</span>
            </div>
          </div>
        )}
      </div>

      <MicButton isRecording={isRecording} onClick={handleMicClick} />
    </div>
  );
}