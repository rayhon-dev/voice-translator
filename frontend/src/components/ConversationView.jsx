import { useState } from "react";
import MicButton from "./MicButton";
import LanguageSelector from "./LanguageSelector";
import TranscriptLine from "./TranscriptLine";
import DialogBubble from "./DialogBubble";
import { usePlayAudio } from "../hooks/usePlayAudio";
import { useLiveTranscription } from "../hooks/useLiveTranscription";
import { translateText, speakText, identifySpeaker, resetDialog } from "../services/api";

export default function ConversationView({ mode, onBack }) {
  const [targetLang, setTargetLang] = useState("uz");
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [audioCache, setAudioCache] = useState({});

  const { liveText, start, stop } = useLiveTranscription();
  const { playingId, togglePlay } = usePlayAudio();

  const handleMicClick = async () => {
    if (!isRecording) {
      try {
        if (mode === "dialog" && messages.length === 0) {
          await resetDialog();
        }
        await start();
        setIsRecording(true);
      } catch (err) {
        console.error("[ConversationView] failed to start recording", err);
        setIsRecording(false);
      }
      return;
    }

    setIsRecording(false);
    setIsProcessing(true);

    // End-to-end latency measurement for the whole "stop -> result on screen"
    // path. Read these together with the backend [/ws/transcribe], [MT] and
    // [SPK] log lines to locate the bottleneck.
    const flowStart = performance.now();
    const ms = (from) => `${(performance.now() - from).toFixed(0)}ms`;

    try {
      // stop() always resolves; `text` comes from a ref that is in sync with
      // the latest WebSocket message, so it is never stale.
      const stopStart = performance.now();
      const { blob: audioBlob, text } = await stop();
      console.log(`[latency] stop() round trip: ${ms(stopStart)}`);
      const originalText = (text || "").trim();

      // Nothing was transcribed (e.g. WS never connected, or silence) — don't
      // call the translation endpoint with an empty string.
      if (!originalText) {
        console.warn(
          "[ConversationView] transcription was empty — skipping translation"
        );
        return;
      }

      let speaker = null;
      if (mode === "dialog") {
        try {
          const spkStart = performance.now();
          speaker = await identifySpeaker(audioBlob);
          console.log(`[latency] identifySpeaker(): ${ms(spkStart)}`);
        } catch (err) {
          console.error("[ConversationView] identifySpeaker failed", err);
          // Non-fatal: keep going with speaker = null.
        }
      }

      let translation;
      try {
        const trStart = performance.now();
        translation = await translateText(originalText, targetLang);
        console.log(`[latency] translateText(): ${ms(trStart)}`);
      } catch (err) {
        console.error("[ConversationView] translateText failed", err);
        return; // finally still clears the processing state
      }

      setMessages((prev) => [...prev, { originalText, translation, speaker }]);
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
        <LanguageSelector selectedLang={targetLang} onSelect={setTargetLang} />
      </div>

      <div className="w-full max-w-6xl flex flex-col gap-6 flex-1">
        {messages.map((msg, index) =>
          mode === "dialog" ? (
            <DialogBubble
              key={index}
              originalText={msg.originalText}
              translatedText={msg.translation}
              speaker={msg.speaker}
              isPlaying={playingId === index}
              onPlayAudio={() => handlePlayAudio(index, msg.translation)}
            />
          ) : (
            <TranscriptLine
              key={index}
              originalText={msg.originalText}
              translatedText={msg.translation}
              speaker={msg.speaker}
              isPlaying={playingId === index}
              onPlayAudio={() => handlePlayAudio(index, msg.translation)}
            />
          )
        )}

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