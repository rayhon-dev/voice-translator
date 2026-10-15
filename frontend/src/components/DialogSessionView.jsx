import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDialogSession } from "../hooks/useDialogSession";
import { usePlayAudio } from "../hooks/usePlayAudio";
import { speakText } from "../services/api";
import DialogBubble from "./DialogBubble";
import AudioLevelBars from "./AudioLevelBars";

// Spiker endi A/B (ikki kishi) emas, cheklanmagan sonli butun son ID
// (speaker_diart.py'dagi onlayn ovoz-klasterlash). ESKI yondashuv — rang
// (id-1) % N bo'yicha, ya'ni SPIKER RAQAMIGA bog'langan edi: birlashtirish
// (speakers_merged) yoki o'chirish (speaker_removed) natijasida bir xil
// raqam boshqa odamga "tegib" qolsa, uning rangi ham noto'g'ri o'zgarib
// ketardi. ENDI rang spiker OB'EKTIGA (uning paydo bo'lish TARTIBIGA)
// bog'lanadi — qarang pastdagi `speakerColorIds`/`colorForSpeaker`:
// har bir YANGI spiker birinchi marta ko'ringanda navbatdagi rang oladi
// va shu rang (birlashtirish paytida "into" rangi saqlanib qolgan holda)
// sessiya davomida O'ZGARMAYDI. 12 ta rang — bir-biridan aniq farq
// qiladigan tus (hue) bo'yicha tanlangan.
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

// Pastdan shuncha pikseldan UZOQROQ bo'lsa, foydalanuvchi "o'zi yuqoriga
// skroll qilgan" deb hisoblanadi — avtomatik skroll to'xtaydi va "Yangi
// xabarlar" tugmasi chiqadi.
const AUTO_SCROLL_THRESHOLD_PX = 150;

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
  // XATO TUZATISHI: ilgari TTS so'rovi muvaffaqiyatsiz bo'lganda (masalan
  // "en" modeli hali GPU'ga yuklanmagan paytda birinchi so'rov bilan to'qnashib
  // qolsa) handlePlayAudio buni faqat console.error qilardi — foydalanuvchi
  // hech narsa eshitmay, SABABSIZ qolardi. Endi shu seq uchun ko'rinadigan
  // xato belgisi chiqariladi.
  const [audioErrors, setAudioErrors] = useState({});
  const { playingId, togglePlay } = usePlayAudio();

  // --- Guruh-chat qulayliklari (frontend-ONLY holat, backend'ga ta'sir
  // qilmaydi) ---
  // Spiker ismlari: {speakerId: customName}. Kalit yo'q bo'lsa default
  // ("N-spiker") ishlatiladi.
  const [speakerNames, setSpeakerNames] = useState({});

  // Rang HAR BIR spiker OB'EKTIGA uning paydo bo'lish TARTIBIDA
  // tayinlanadi va bir marta tayinlangandan keyin HECH QACHON o'zgarmaydi
  // — tayinlovning o'zi (`speakerColorOrder`: {speakerId: order}) endi
  // useDialogSession.js'da, FAQAT yangi spiker ID birinchi "turn" xabarida
  // ko'ringanda, bitta joyda yuritiladi (qarang undagi izoh). Bu yerda
  // FAQAT shu order'ni palitra indeksiga aylantiramiz — o'chirish
  // (delete_turn) yoki birlashtirish (speakers_merged) HECH QANDAY qayta
  // hisoblashga olib kelmaydi, demak qolgan spikerlarning rangi siljimaydi.
  const colorForSpeaker = useCallback(
    (speakerId) => SPEAKER_PALETTE[(speakerColorOrder[speakerId] ?? 0) % SPEAKER_PALETTE.length],
    [speakerColorOrder]
  );

  const knownSpeakers = useMemo(() => {
    const ids = [...new Set(messages.map((m) => m.speaker))].sort((a, b) => a - b);
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
      // Backend'ga FAQAT diagnostika/log maqsadida yuboriladi — klasterlash
      // mantig'iga ta'sir qilmaydi (qarang dialog.py, rename_speaker action).
      renameSpeaker(speakerId, newName || defaultSpeakerName(speakerId));
    },
    [renameSpeaker]
  );

  // Bir xil ism = bitta odam: foydalanuvchi ismni boshqa MAVJUD spikerning
  // (bo'sh joylarsiz, katta-kichik harfga e'tibor bermasdan solishtirilgan)
  // ismi bilan bir xil qilib tahrirlasa — ikki alohida spiker bitta odam
  // bo'lib qolishiga yo'l qo'ymaslik uchun avval tasdiqlash so'raladi
  // (pastdagi `pendingRename` modal). Tasdiqlanmaguncha ism O'ZGARMAYDI
  // (applyRename chaqirilmaydi, shuning uchun bubble eski ismini
  // ko'rsatishda davom etadi).
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

  // Backend'dan kelgan "speakers_merged"/"speaker_removed" voqealarini
  // (useDialogSession.js'dagi speakerEvents, append-only ro'yxat) kuzatib,
  // LOKAL speakerNames xaritasini mos ravishda yangilaydi — bu, server
  // tomonidan allaqachon SODIR BO'LGAN tashqi hodisaga REAKSIYA (klassik,
  // to'g'ri effekt holati — render vaqtida hisoblab bo'lmaydigan narsa).
  // `processedEventCountRef` — har safar FAQAT yangi (hali ishlanmagan)
  // voqealarni qayta ishlash uchun.
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
          // "from"ning ismi FAQAT "into" hali default nomda bo'lsa
          // (ya'ni xaritada o'z yozuvi yo'q bo'lsa) meros oladi.
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

  // --- Avtomatik skroll (3-band) ---
  const scrollContainerRef = useRef(null);
  // Ref, state EMAS — har scroll-hodisasida o'zgaradi, re-render shart
  // emas (faqat keyingi auto-skroll qarori uchun o'qiladi).
  const autoFollowRef = useRef(true);
  const [showJumpToBottom, setShowJumpToBottom] = useState(false);

  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const nearBottom = distanceFromBottom <= AUTO_SCROLL_THRESHOLD_PX;
    autoFollowRef.current = nearBottom;
    setShowJumpToBottom(!nearBottom);
  }, []);

  // Yangi xabar (yoki "Yozmoqda…" ko'rsatkichi) qo'shilganda — FAQAT
  // foydalanuvchi hozir pastda (auto-follow yoqilgan) bo'lsa — pastga
  // silliq skroll qilinadi. Mavjud bubble (tuzatish/retro-merge) shunchaki
  // YANGILANSA ham shu effekt qayta ishga tushadi, lekin foydalanuvchi
  // yuqoriga skroll qilgan bo'lsa (autoFollowRef=false) HECH NARSA
  // qilinmaydi — sahifa sakrab ketmaydi.
  useEffect(() => {
    if (!autoFollowRef.current) return;
    const el = scrollContainerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, isProcessing]);

  const jumpToBottom = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    autoFollowRef.current = true;
    setShowJumpToBottom(false);
  }, []);

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

      <div className="relative flex-1 min-h-0">
        <div
          ref={scrollContainerRef}
          onScroll={handleScroll}
          className="h-full overflow-y-auto"
        >
          <div className="w-full max-w-3xl mx-auto flex flex-col px-3 sm:px-10 py-4 sm:py-8">
            {messages.map((msg, idx) => {
              const prev = messages[idx - 1];
              // Guruhlashni o'chirish: ENDI faqat BO'SHLIQ (margin)ga ta'sir
              // qiladi — avatar/ism/tugmalar HAR BIR bubble'da (grouped
              // bo'lsa ham) ko'rsatiladi, qarang DialogBubble.jsx.
              const grouped = Boolean(prev && prev.speaker === msg.speaker);
              const palette = colorForSpeaker(msg.speaker);
              const resolvedName = speakerNames[msg.speaker] || defaultSpeakerName(msg.speaker);
              // Toq raqamli spikerlar (1, 3, 5...) chapda, juft raqamlilar
              // (2, 4...) o'ngda — "Men"/"Bu menman" tushunchasi YO'Q, bu
              // faqat spiker ID'ning juft-toqligiga bog'liq, doimiy
              // joylashuv.
              const isRight = Number(msg.speaker) % 2 === 0;
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
                    speakerNames[msg.speaker] ? speakerNames[msg.speaker][0] : String(msg.speaker)
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
                  onRename={(newName) => handleRename(msg.speaker, newName)}
                  onDelete={() => deleteTurn(msg.seq)}
                />
              );
            })}

            {/* "Yozmoqda…" ko'rsatkichi — backend "processing" statusini
                yuborganda (isProcessing=true) ro'yxat oxirida vaqtinchalik
                chiqadi, natija ("turn"/"turn_skipped") kelganda yo'qoladi. */}
            {isProcessing ? (
              <div className="flex justify-start w-full mt-4">
                <div className="bg-white/70 backdrop-blur rounded-3xl px-5 py-4">
                  <span className="flex gap-1">
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  </span>
                </div>
              </div>
            ) : null}
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

      {/* Bir xil ism = bitta odam: tasdiqlash oynasi — qarang handleRename
          izohi yuqorida. */}
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
