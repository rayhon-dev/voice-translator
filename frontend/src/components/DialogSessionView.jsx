import { useEffect, useState } from "react";
import { useDialogSession } from "../hooks/useDialogSession";
import { usePlayAudio } from "../hooks/usePlayAudio";
import { speakText } from "../services/api";
import LanguageSelector from "./LanguageSelector";
import DialogBubble from "./DialogBubble";
import AudioLevelBars from "./AudioLevelBars";

const SPEAKER_ORDINAL = { A: "1-kishi", B: "2-kishi" };

export default function DialogSessionView({ onBack }) {
  const {
    phase,
    currentSpeaker,
    speakerLangs,
    messages,
    error,
    notice,
    audioLevel,
    isProcessing,
    startDialog,
    selectLanguage,
    stopDialog,
    correctSpeaker,
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
    // B tanlayotganda A allaqachon tanlagan til band qilinadi (Google
    // Translate'ning "conversation mode"idagi kabi) — ikkala spiker bir
    // xil tilni tanlay olmasligi uchun, chunki bunday holatda tarjima
    // qilinadigan narsa qolmas edi.
    const otherSpeaker = currentSpeaker === "B" ? "A" : "B";
    const otherLang = speakerLangs[otherSpeaker];
    const excludeLangs = otherLang ? [otherLang] : [];
    const defaultLang = otherLang === "en" ? "ru" : "en";
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-6 px-6">
        <p className="text-gray-700 text-lg">{speakerLabel}, tilingizni tanlang</p>
        <LanguageSelector
          selectedLang={speakerLangs[currentSpeaker] || defaultLang}
          onSelect={selectLanguage}
          excludeLangs={excludeLangs}
        />
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
            isPlaying={playingId === msg.seq}
            conflict={msg.conflict}
            audioError={audioErrors[msg.seq]}
            onPlayAudio={() => handlePlayAudio(msg.seq, msg.translation, msg.targetLang)}
            onCorrectSpeaker={() => correctSpeaker(msg.seq)}
          />
        ))}
      </div>
    </div>
  );
}
