import { useEffect, useMemo, useState } from "react";
import { useDialogSession } from "../hooks/useDialogSession";
import { usePlayAudio } from "../hooks/usePlayAudio";
import { speakText } from "../services/api";
import DialogBubble from "./DialogBubble";
import AudioLevelBars from "./AudioLevelBars";

// Spiker endi A/B (ikki kishi) emas, cheklanmagan sonli butun son ID
// (speaker_diart.py'dagi onlayn ovoz-klasterlash) — har biriga shu 8
// rangli palitradan (id-1) % 8 bo'yicha barqaror rang tayinlanadi, shunda
// 9-spiker bo'lsa ham rang takrorlanib, hech kim rangsiz qolmaydi.
const SPEAKER_PALETTE = [
  { badge: "bg-purple-100 text-purple-700", dot: "bg-purple-400" },
  { badge: "bg-blue-100 text-blue-700", dot: "bg-blue-400" },
  { badge: "bg-green-100 text-green-700", dot: "bg-green-400" },
  { badge: "bg-amber-100 text-amber-700", dot: "bg-amber-400" },
  { badge: "bg-pink-100 text-pink-700", dot: "bg-pink-400" },
  { badge: "bg-cyan-100 text-cyan-700", dot: "bg-cyan-400" },
  { badge: "bg-orange-100 text-orange-700", dot: "bg-orange-400" },
  { badge: "bg-teal-100 text-teal-700", dot: "bg-teal-400" },
];

function speakerPalette(speakerId) {
  const idx = (Math.max(1, Number(speakerId) || 1) - 1) % SPEAKER_PALETTE.length;
  return SPEAKER_PALETTE[idx];
}

function speakerLabel(speakerId) {
  return `${speakerId}-spiker`;
}

export default function DialogSessionView({ onBack }) {
  const {
    phase,
    messages,
    error,
    notice,
    audioLevel,
    isProcessing,
    startDialog,
    stopDialog,
    correctSpeaker,
    deleteTurn,
    pauseMic,
    resumeMic,
  } = useDialogSession();

  const [audioCache, setAudioCache] = useState({});
  // XATO TUZATISHI: ilgari TTS so'rovi muvaffaqiyatsiz bo'lganda (masalan
  // "en" modeli hali GPU'ga yuklanmagan paytda birinchi so'rov bilan to'qnashib
  // qolsa) handlePlayAudio buni faqat console.error qilardi — foydalanuvchi
  // hech narsa eshitmay, SABABSIZ qolardi. Endi shu seq uchun ko'rinadigan
  // xato belgisi chiqariladi.
  const [audioErrors, setAudioErrors] = useState({});
  const { playingId, togglePlay } = usePlayAudio();

  // "spikerni almashtirish" menyusi uchun — hozirgacha ko'ringan BARCHA
  // (noyob) spiker ID'lari, raqam bo'yicha tartiblangan. Alohida server
  // holati emas — messages'ning o'zidan hisoblab chiqiladi.
  const knownSpeakers = useMemo(() => {
    const ids = [...new Set(messages.map((m) => m.speaker))].sort((a, b) => a - b);
    return ids.map((id) => ({
      id,
      label: speakerLabel(id),
      dotClass: speakerPalette(id).dot,
    }));
  }, [messages]);

  // DIALOG_PLAN.md 3-bosqich (C5): TTS (tarjima ovozi) ijro etilayotganda
  // mikrofon VAD'ini to'xtatamiz — aks holda karnaydan chiqayotgan ovoz
  // mikrofon orqali qayta eshitilib, YANGI turn sifatida yuborilib ketishi
  // mumkin edi. `playingId` — usePlayAudio() dagi reaktiv holat: biror
  // bubble ijro etilayotganda shu bubble'ning `seq`'iga teng, aks holda
  // `null`. shuning uchun null<->non-null o'tishi aynan "TTS boshlandi"/
  // "TTS tugadi" voqeasiga to'g'ri keladi.
  useEffect(() => {
    if (playingId !== null) {
      pauseMic("tts_playback");
    } else {
      resumeMic("tts_playback_ended");
    }
  }, [playingId, pauseMic, resumeMic]);

  const handlePlayAudio = async (seq, text, lang) => {
    try {
      setAudioErrors((prev) => ({ ...prev, [seq]: false }));
      let blob = audioCache[seq];
      if (!blob) {
        blob = await speakText(text, lang);
        setAudioCache((prev) => ({ ...prev, [seq]: blob }));
      }
      togglePlay(seq, blob);
    } catch (err) {
      console.error("[DialogSessionView] failed to play audio", err);
      setAudioErrors((prev) => ({ ...prev, [seq]: true }));
    }
  };

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
    <div className="min-h-screen flex flex-col items-center">
      <div className="sticky top-0 z-20 w-full bg-[#FAF8FC] border-b border-black/5 px-10 py-6 flex flex-col items-center gap-4">
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
        ) : (
          <>
            <AudioLevelBars level={audioLevel} />
            {/* DIALOG_PLAN.md 4-bosqich: "Tinglanmoqda / Qayta ishlanmoqda…"
                ko'rsatkichi — STT/tarjima bir necha soniya cho'zilishi
                mumkin, shu payt hech narsa ko'rinmasa foydalanuvchi
                "ishlamayapti" deb o'ylab qolishi mumkin edi. */}
            <p className="text-xs text-gray-400">
              {isProcessing ? "Qayta ishlanmoqda…" : "Tinglanmoqda…"}
            </p>
          </>
        )}
      </div>

      <div className="w-full max-w-6xl flex flex-col gap-6 flex-1 px-10 py-8">
        {messages.map((msg) => (
          <DialogBubble
            key={msg.seq}
            originalText={msg.originalText}
            translatedText={msg.translation}
            speaker={msg.speaker}
            speakerLabel={speakerLabel(msg.speaker)}
            badgeClass={speakerPalette(msg.speaker).badge}
            uncertain={msg.uncertain}
            knownSpeakers={knownSpeakers}
            isPlaying={playingId === msg.seq}
            audioError={audioErrors[msg.seq]}
            onPlayAudio={() => handlePlayAudio(msg.seq, msg.translation, msg.targetLang)}
            onSelectSpeaker={(targetSpeaker) => correctSpeaker(msg.seq, targetSpeaker)}
            onDelete={() => deleteTurn(msg.seq)}
          />
        ))}
      </div>
    </div>
  );
}
