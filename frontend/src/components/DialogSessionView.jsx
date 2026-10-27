import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDialogSession } from "../hooks/useDialogSession";
import { usePlayAudio } from "../hooks/usePlayAudio";
import { speakText } from "../services/api";
import DialogBubble from "./DialogBubble";
import AudioLevelBars from "./AudioLevelBars";

const SPEAKER_PALETTE = [
  { avatar: "bg-red-400", name: "text-red-600", dot: "bg-red-400", border: "border-red-400", light: "bg-red-50" },
  { avatar: "bg-orange-400", name: "text-orange-600", dot: "bg-orange-400", border: "border-orange-400", light: "bg-orange-50" },
  { avatar: "bg-amber-400", name: "text-amber-600", dot: "bg-amber-400", border: "border-amber-400", light: "bg-amber-50" },
  { avatar: "bg-green-400", name: "text-green-600", dot: "bg-green-400", border: "border-green-400", light: "bg-green-50" },
  { avatar: "bg-teal-400", name: "text-teal-600", dot: "bg-teal-400", border: "border-teal-400", light: "bg-teal-50" },
  { avatar: "bg-cyan-400", name: "text-cyan-600", dot: "bg-cyan-400", border: "border-cyan-400", light: "bg-cyan-50" },
  { avatar: "bg-blue-400", name: "text-blue-600", dot: "bg-blue-400", border: "border-blue-400", light: "bg-blue-50" },
  { avatar: "bg-indigo-400", name: "text-indigo-600", dot: "bg-indigo-400", border: "border-indigo-400", light: "bg-indigo-50" },
  { avatar: "bg-violet-400", name: "text-violet-600", dot: "bg-violet-400", border: "border-violet-400", light: "bg-violet-50" },
  { avatar: "bg-fuchsia-400", name: "text-fuchsia-600", dot: "bg-fuchsia-400", border: "border-fuchsia-400", light: "bg-fuchsia-50" },
  { avatar: "bg-pink-400", name: "text-pink-600", dot: "bg-pink-400", border: "border-pink-400", light: "bg-pink-50" },
  { avatar: "bg-rose-400", name: "text-rose-600", dot: "bg-rose-400", border: "border-rose-400", light: "bg-rose-50" },
];

function defaultSpeakerName(speakerId) {
  return `${speakerId}-spiker`;
}

const UNKNOWN_SPEAKER_LABEL = "Aniqlanmoqda…";
const NEUTRAL_PALETTE = {
  avatar: "bg-gray-300",
  name: "text-gray-400",
  dot: "bg-gray-300",
  border: "border-gray-300",
  light: "bg-gray-50",
};

const AUTO_SCROLL_THRESHOLD_PX = 150;
const AUTO_SCROLL_RESUME_MS = 5000;

export default function DialogSessionView({ onBack }) {
  const {
    phase,
    messages,
    speakerEvents,
    speakerColorOrder,
    error,
    notice,
    audioLevel,
    isProcessing,
    startDialog,
    stopDialog,
    correctSpeaker,
    deleteTurn,
    renameSpeaker,
    mergeSpeakers,
    pauseMic,
    resumeMic,
  } = useDialogSession();

  const [audioCache, setAudioCache] = useState({});
  const [audioErrors, setAudioErrors] = useState({});
  const { playingId, togglePlay } = usePlayAudio();

  const [speakerNames, setSpeakerNames] = useState({});

  const colorForSpeaker = useCallback(
    (speakerId) =>
      speakerId === null
        ? NEUTRAL_PALETTE
        : SPEAKER_PALETTE[(speakerColorOrder[speakerId] ?? 0) % SPEAKER_PALETTE.length],
    [speakerColorOrder]
  );

  const knownSpeakers = useMemo(() => {
    const ids = [...new Set(messages.map((m) => m.speaker))]
      .filter((id) => id !== null)
      .sort((a, b) => a - b);
    return ids.map((id) => ({
      id,
      name: speakerNames[id] || defaultSpeakerName(id),
      dotClass: colorForSpeaker(id).dot,
    }));
  }, [messages, speakerNames, colorForSpeaker]);

  const applyRename = useCallback(
    (speakerId, newName) => {
      setSpeakerNames((prev) => {
        const next = { ...prev };
        if (newName) {
          next[speakerId] = newName;
        } else {
          delete next[speakerId];
        }
        return next;
      });
      renameSpeaker(speakerId, newName || defaultSpeakerName(speakerId));
    },
    [renameSpeaker]
  );

  const [pendingRename, setPendingRename] = useState(null); // {speakerId, newName, conflictId, conflictName} | null

  const handleRename = useCallback(
    (speakerId, newName) => {
      if (newName) {
        const normalized = newName.trim().toLowerCase();
        const conflict = knownSpeakers.find(
          (s) =>
            s.id !== speakerId &&
            (speakerNames[s.id] || defaultSpeakerName(s.id)).trim().toLowerCase() === normalized
        );
        if (conflict) {
          setPendingRename({
            speakerId,
            newName,
            conflictId: conflict.id,
            conflictName: conflict.name,
          });
          return;
        }
      }
      applyRename(speakerId, newName);
    },
    [knownSpeakers, speakerNames, applyRename]
  );

  const confirmMergeRename = useCallback(() => {
    if (!pendingRename) return;
    mergeSpeakers(pendingRename.speakerId, pendingRename.conflictId);
    setPendingRename(null);
  }, [pendingRename, mergeSpeakers]);

  const cancelMergeRename = useCallback(() => {
    setPendingRename(null);
  }, []);

  const processedEventCountRef = useRef(0);
  useEffect(() => {
    if (processedEventCountRef.current >= speakerEvents.length) return;
    for (let i = processedEventCountRef.current; i < speakerEvents.length; i++) {
      const ev = speakerEvents[i];
      if (ev.type === "removed") {
        setSpeakerNames((prev) => {
          if (!(ev.speaker in prev)) return prev;
          const next = { ...prev };
          delete next[ev.speaker];
          return next;
        });
      } else if (ev.type === "merged") {
        setSpeakerNames((prev) => {
          if (!(ev.from in prev)) return prev;
          const next = { ...prev };
          if (!(ev.into in prev)) {
            next[ev.into] = next[ev.from];
          }
          delete next[ev.from];
          return next;
        });
      }
    }
    processedEventCountRef.current = speakerEvents.length;
  }, [speakerEvents]);

  const scrollContainerRef = useRef(null);
  const autoFollowRef = useRef(true);
  const [showJumpToBottom, setShowJumpToBottom] = useState(false);
  const programmaticScrollRef = useRef(false);
  const programmaticScrollTimeoutRef = useRef(null);
  const resumeTimeoutRef = useRef(null);

  const scrollToBottom = useCallback((behavior = "smooth") => {
    const el = scrollContainerRef.current;
    if (!el) return;
    programmaticScrollRef.current = true;
    el.scrollTo({ top: el.scrollHeight, behavior });
    if (programmaticScrollTimeoutRef.current) {
      clearTimeout(programmaticScrollTimeoutRef.current);
    }
    programmaticScrollTimeoutRef.current = setTimeout(() => {
      programmaticScrollRef.current = false;
    }, 600);
  }, []);

  const resumeAutoFollow = useCallback(() => {
    autoFollowRef.current = true;
    setShowJumpToBottom(false);
    if (resumeTimeoutRef.current) {
      clearTimeout(resumeTimeoutRef.current);
      resumeTimeoutRef.current = null;
    }
    scrollToBottom();
  }, [scrollToBottom]);

  const pauseAutoFollow = useCallback(() => {
    autoFollowRef.current = false;
    setShowJumpToBottom(true);
    if (resumeTimeoutRef.current) {
      clearTimeout(resumeTimeoutRef.current);
    }
    resumeTimeoutRef.current = setTimeout(() => {
      resumeTimeoutRef.current = null;
      resumeAutoFollow();
    }, AUTO_SCROLL_RESUME_MS);
  }, [resumeAutoFollow]);

  const handleUserScroll = useCallback(() => {
    if (programmaticScrollRef.current) return;
    const el = scrollContainerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom > AUTO_SCROLL_THRESHOLD_PX) {
      if (autoFollowRef.current) {
        pauseAutoFollow();
      }
    } else if (!autoFollowRef.current) {
      // Foydalanuvchi qo'lda pastga qaytdi.
      resumeAutoFollow();
    }
  }, [pauseAutoFollow, resumeAutoFollow]);

  useEffect(() => {
    if (!autoFollowRef.current) return;
    scrollToBottom();
  }, [messages, isProcessing, scrollToBottom]);

  useEffect(() => {
    return () => {
      if (programmaticScrollTimeoutRef.current) clearTimeout(programmaticScrollTimeoutRef.current);
      if (resumeTimeoutRef.current) clearTimeout(resumeTimeoutRef.current);
    };
  }, []);

  const jumpToBottom = useCallback(() => {
    resumeAutoFollow();
  }, [resumeAutoFollow]);

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
    <div className="h-screen flex flex-col">
      <div className="shrink-0 w-full bg-[#FAF8FC] border-b border-black/5 px-4 sm:px-10 py-4 sm:py-6 flex flex-col items-center gap-3 sm:gap-4">
        <div className="w-full flex justify-between items-center">
          <button onClick={onBack} className="text-gray-500 text-sm">
            ← Orqaga
          </button>
          <button
            onClick={stopDialog}
            className="px-4 sm:px-5 py-2 rounded-full bg-white/60 backdrop-blur text-gray-700 text-sm hover:bg-white/80 transition-colors"
          >
            Suhbatni tugatish
          </button>
        </div>

        {notice ? (
          <p className="text-amber-600 text-sm">{notice}</p>
        ) : (
          <>
            <AudioLevelBars level={audioLevel} />
            <p className="text-xs text-gray-400">
              {isProcessing ? "Qayta ishlanmoqda…" : "Tinglanmoqda…"}
            </p>
          </>
        )}
      </div>

      <div className="relative flex-1 min-h-0">
        <div
          ref={scrollContainerRef}
          onScroll={handleUserScroll}
          className="h-full overflow-y-auto"
        >
          <div className="w-full max-w-3xl mx-auto flex flex-col px-3 sm:px-10 py-4 sm:py-8">
            {messages.map((msg, idx) => {
              if (msg.placeholder) {
                return (
                  <div key={`ph-${msg.clientSeq}`} className="flex justify-start w-full mt-4">
                    <div className="bg-white/70 backdrop-blur rounded-3xl px-5 py-4 text-gray-400 text-sm">
                      …
                    </div>
                  </div>
                );
              }

              const prev = messages[idx - 1];
              const grouped = Boolean(prev && !prev.placeholder && prev.speaker === msg.speaker);
              const palette = colorForSpeaker(msg.speaker);
              const isUnknown = msg.speaker === null;
              const resolvedName = isUnknown
                ? UNKNOWN_SPEAKER_LABEL
                : speakerNames[msg.speaker] || defaultSpeakerName(msg.speaker);
              const isRight = isUnknown ? false : Number(msg.speaker) % 2 === 0;
              return (
                <DialogBubble
                  key={msg.seq}
                  originalText={msg.originalText}
                  translatedText={msg.translation}
                  sourceLang={msg.sourceLang ? msg.sourceLang.toUpperCase() : null}
                  speaker={msg.speaker}
                  speakerName={resolvedName}
                  isRight={isRight}
                  avatarLetter={
                    isUnknown ? "?" : speakerNames[msg.speaker] ? speakerNames[msg.speaker][0] : String(msg.speaker)
                  }
                  avatarClass={palette.avatar}
                  nameColorClass={palette.name}
                  borderColorClass={palette.border}
                  lightBgClass={palette.light}
                  grouped={grouped}
                  uncertain={msg.uncertain}
                  mtProvider={msg.mtProvider}
                  translationError={msg.translationError}
                  knownSpeakers={knownSpeakers.filter((s) => s.id !== msg.speaker)}
                  isPlaying={playingId === msg.seq}
                  audioError={audioErrors[msg.seq]}
                  onPlayAudio={() => handlePlayAudio(msg.seq, msg.translation, msg.targetLang)}
                  onSelectSpeaker={(targetSpeaker) => correctSpeaker(msg.seq, targetSpeaker)}
                  onRename={(newName) => {
                    if (isUnknown) return;
                    handleRename(msg.speaker, newName);
                  }}
                  onDelete={() => deleteTurn(msg.seq)}
                />
              );
            })}
          </div>
        </div>

        {showJumpToBottom ? (
          <button
            onClick={jumpToBottom}
            className="absolute bottom-4 left-1/2 -translate-x-1/2 px-4 py-2 rounded-full bg-purple-400 text-white text-sm shadow-lg hover:scale-105 transition-transform"
          >
            ↓ Yangi xabarlar
          </button>
        ) : null}
      </div>

      {pendingRename ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-6">
          <div className="max-w-sm w-full rounded-3xl bg-white p-6 shadow-xl text-center">
            <p className="text-gray-700 text-base mb-5">
              «{pendingRename.conflictName}» nomli spiker allaqachon bor. Ularni
              bitta odam sifatida birlashtiraymi?
            </p>
            <div className="flex justify-center gap-3">
              <button
                onClick={confirmMergeRename}
                className="px-6 py-2 rounded-full bg-purple-400 text-white text-sm font-medium hover:scale-105 transition-transform"
              >
                Ha
              </button>
              <button
                onClick={cancelMergeRename}
                className="px-6 py-2 rounded-full bg-gray-100 text-gray-600 text-sm font-medium hover:bg-gray-200 transition-colors"
              >
                Yo'q
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
