import { useCallback, useEffect, useRef, useState } from "react";

const WS_URL = "ws://localhost:8000/ws/dialog";

const SPEECH_RMS_THRESHOLD = 0.02;
// Tabiiy suhbatda gaplar orasidagi pauza ko'pincha 2 soniyadan qisqaroq
// bo'ladi — shuning uchun bu qiymat ilgari juda katta edi (2000ms) va
// navbatlar bir necha o'nlab soniyaga cho'zilib ketardi (real testda 80+
// soniyagacha kuzatildi), bu esa STT'ning til-aniqlashini butunlay
// chalg'itib yuborardi (u qisqa, alohida gaplar uchun mo'ljallangan).
const SILENCE_DURATION_MS = 800;
// Agar odam 2 soniyalik pauza qilmasdan uzoq gapiraversa ham, navbat
// cheksiz cho'zilib ketmasligi (va STT/til-aniqlash sifati yomonlashmasligi,
// tarjima esa real vaqtga yaqin chiqishi) uchun qattiq yuqori chegara.
const MAX_TURN_DURATION_MS = 12000;
const VAD_CHECK_INTERVAL_MS = 100;
const AUDIO_LEVEL_RMS_MAX = 0.15; // shu RMS qiymatida audioLevel = 1 ga yetadi
const AUDIO_LEVEL_SMOOTHING = 0.4; // 0..1, kattaroq = tezroq (kamroq silliqlash)

const DEBUG_VAD = false;

export function useDialogSession() {
  const [phase, setPhase] = useState("idle"); // idle | await_language | listening | ended | error
  const [currentSpeaker, setCurrentSpeaker] = useState(null); // "A" | "B" | null
  // A va B tanlagan tillar, masalan {A: "en"} — B hali tanlamagan bo'lsa
  // kalit yo'q. LanguageSelector'ga B uchun A ning tilini "band" deb
  // ko'rsatish (bir xil tilni ikkalasi ham tanlay olmasligi) uchun kerak.
  const [speakerLangs, setSpeakerLangs] = useState({});
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null); // vaqtinchalik "eshitilmadi" kabi xabarlar uchun, error state'dan alohida
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);

  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const stoppedRef = useRef(false);
  const noticeTimeoutRef = useRef(null);

  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const vadIntervalRef = useRef(null);
  const turnRecorderRef = useRef(null);

  const cleanupMedia = useCallback(() => {
    if (streamRef.current) {
      try {
        streamRef.current.getTracks().forEach((track) => track.stop());
      } catch (err) {
        console.error("[useDialogSession] error stopping media tracks", err);
      }
      streamRef.current = null;
    }
  }, []);

  const closeSocket = useCallback(() => {
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (err) {
        console.error("[useDialogSession] error closing socket", err);
      }
      wsRef.current = null;
    }
  }, []);

  const stopListenLoop = useCallback(() => {
    if (vadIntervalRef.current) {
      if (DEBUG_VAD) {
        console.log("[VAD-DEBUG] stopListenLoop: clearing existing VAD interval");
      }
      clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = null;
    }

    if (turnRecorderRef.current) {
      try {
        if (turnRecorderRef.current.state !== "inactive") {
          turnRecorderRef.current.stop();
        }
      } catch (err) {
        console.error("[useDialogSession] error stopping turn recorder", err);
      }
      turnRecorderRef.current = null;
    }

    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (err) {
        console.error("[useDialogSession] error closing audio context", err);
      }
      audioContextRef.current = null;
    }

    analyserRef.current = null;
    setAudioLevel(0);
  }, []);

  const startListenLoop = useCallback(() => {
    if (!streamRef.current) return;

    stopListenLoop();

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContextClass();
    const source = audioContext.createMediaStreamSource(streamRef.current);

    // Mikrofon kirish signali haddan tashqari baland bo'lganda clipping
    // (buzilish) yuzaga kelib, diarizatsiya/til aniqlashni chalg'itishi
    // mumkin edi (DIAG_CLIP_PEAK_THRESHOLD logida tasdiqlangan). Shu
    // sababli signal tahlil qilinishi (analyser) VA yozib olinishi
    // (MediaRecorder, destination orqali) dan oldin limiter sifatida
    // DynamicsCompressorNode qo'yiladi — nutq uchun odatiy sozlamalar.
    const compressor = audioContext.createDynamicsCompressor();
    compressor.threshold.value = -24;
    compressor.knee.value = 30;
    compressor.ratio.value = 12;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.25;
    source.connect(compressor);

    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    compressor.connect(analyser);

    // MediaRecorder avval xom mikrofon oqimidan (streamRef.current) to'g'ridan
    // to'g'ri yozib olardi — bu compressor grafigini butunlay chetlab
    // o'tardi. Backend'ga (va diarizatsiyaga) haqiqatan ham kompressiyalangan
    // signal borishi uchun, yozib olish shu destination'dan amalga oshiriladi.
    const compressedDestination = audioContext.createMediaStreamDestination();
    compressor.connect(compressedDestination);
    const recordingStream = compressedDestination.stream;

    audioContextRef.current = audioContext;
    analyserRef.current = analyser;

    const dataArray = new Float32Array(analyser.fftSize);
    let silenceStartedAt = null;
    let smoothedLevel = 0;

    const startTurnRecording = () => {
      if (!streamRef.current || stoppedRef.current) return;

      let recorder;
      try {
        recorder = new MediaRecorder(recordingStream);
      } catch (err) {
        console.error("[useDialogSession] failed to create turn recorder", err);
        setError(err?.message || String(err));
        return;
      }

      const localChunks = [];

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          localChunks.push(event.data);
        }
      };

      recorder.onerror = (event) => {
        console.error("[useDialogSession] turn recorder error", event);
      };

      recorder.onstop = () => {
        const recordedMs = recorder._vadStartedAt
          ? Date.now() - recorder._vadStartedAt
          : null;
        try {
          const blob = new Blob(localChunks, { type: "audio/webm" });
          if (DEBUG_VAD) {
            console.log(
              `[VAD-DEBUG] onstop: recordedMs=${recordedMs} chunks=${localChunks.length} blobSize=${blob.size}`
            );
          }
          if (
            !stoppedRef.current &&
            wsRef.current &&
            wsRef.current.readyState === WebSocket.OPEN &&
            blob.size > 0
          ) {
            wsRef.current.send(blob);
          }
        } catch (err) {
          console.error("[useDialogSession] error sending turn audio", err);
        }
      };

      recorder.start();
      recorder._vadStartedAt = Date.now();
      turnRecorderRef.current = recorder;
      if (DEBUG_VAD) {
        console.log("[VAD-DEBUG] turn recording STARTED", recorder._vadStartedAt);
      }
    };

    const stopTurnRecording = (reason) => {
      const recorder = turnRecorderRef.current;
      turnRecorderRef.current = null;
      const recordedMs = recorder && recorder._vadStartedAt
        ? Date.now() - recorder._vadStartedAt
        : null;
      if (DEBUG_VAD) {
        console.log(
          `[VAD-DEBUG] turn recording STOP requested reason=${reason} recordedMsSoFar=${recordedMs}`
        );
      }
      if (recorder && recorder.state !== "inactive") {
        try {
          recorder.stop();
        } catch (err) {
          console.error("[useDialogSession] error stopping turn recorder", err);
        }
      }
    };

    const tick = () => {
      if (stoppedRef.current || !analyserRef.current) return;

      analyserRef.current.getFloatTimeDomainData(dataArray);
      let sumSquares = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sumSquares += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sumSquares / dataArray.length);
      const speaking = rms >= SPEECH_RMS_THRESHOLD;
      const silenceElapsedMs = silenceStartedAt === null ? 0 : Date.now() - silenceStartedAt;

      const normalizedLevel = Math.min(1, rms / AUDIO_LEVEL_RMS_MAX);
      smoothedLevel += (normalizedLevel - smoothedLevel) * AUDIO_LEVEL_SMOOTHING;
      setAudioLevel(smoothedLevel);

      if (DEBUG_VAD) {
        console.log(
          `[VAD-DEBUG] tick rms=${rms.toFixed(5)} threshold=${SPEECH_RMS_THRESHOLD} speaking=${speaking} recording=${!!turnRecorderRef.current} silenceElapsedMs=${silenceElapsedMs}`
        );
      }

      setIsSpeaking(speaking);

      if (speaking) {
        silenceStartedAt = null;
        if (!turnRecorderRef.current) {
          startTurnRecording();
        }
      } else if (turnRecorderRef.current) {
        if (silenceStartedAt === null) {
          silenceStartedAt = Date.now();
          if (DEBUG_VAD) {
            console.log("[VAD-DEBUG] silence timer STARTED");
          }
        } else if (Date.now() - silenceStartedAt >= SILENCE_DURATION_MS) {
          if (DEBUG_VAD) {
            console.log(
              `[VAD-DEBUG] silence duration reached ${Date.now() - silenceStartedAt}ms >= ${SILENCE_DURATION_MS}ms`
            );
          }
          stopTurnRecording("silence_timeout");
          silenceStartedAt = null;
        }
      }

      if (
        turnRecorderRef.current &&
        turnRecorderRef.current._vadStartedAt &&
        Date.now() - turnRecorderRef.current._vadStartedAt >= MAX_TURN_DURATION_MS
      ) {
        if (DEBUG_VAD) {
          console.log(`[VAD-DEBUG] max turn duration reached (${MAX_TURN_DURATION_MS}ms) — forcing stop`);
        }
        stopTurnRecording("max_duration");
        silenceStartedAt = null;
      }
    };

    vadIntervalRef.current = setInterval(tick, VAD_CHECK_INTERVAL_MS);
  }, [stopListenLoop]);

  const handleServerMessage = useCallback(
    (raw) => {
      let msg;
      try {
        msg = JSON.parse(raw);
      } catch (err) {
        console.error("[useDialogSession] bad JSON from server", err, raw);
        return;
      }

      switch (msg.status) {
        case "awaiting_language":
          setCurrentSpeaker(msg.speaker || null);
          setPhase("await_language");
          break;

        case "listening_started":
          setCurrentSpeaker(null);
          stoppedRef.current = false;
          setPhase("listening");
          startListenLoop();
          break;

        case "turn":
          setNotice(null);
          if (noticeTimeoutRef.current) {
            clearTimeout(noticeTimeoutRef.current);
            noticeTimeoutRef.current = null;
          }
          // Backend turnlarni PARALLEL qayta ishlaydi (STT/tarjima vaqti
          // turn uzunligiga qarab farq qiladi) — shuning uchun natijalar
          // aytilish emas, TUGASH tartibida kelishi mumkin (masalan qisqa
          // ikkinchi gap uzun birinchi gapdan oldin tugashi mumkin). `seq`
          // audio backend'ga KELGAN tartibni bildiradi (dialog.py'da
          // receiver()da beriladi) — shunga qarab to'g'ri joyga
          // sortirovka qilamiz.
          //
          // Qabul qilingan yondashuv: DARHOL ko'rsatish + sort, navbatga
          // qo'yib KUTDIRISH emas. Sabab: bu jonli suhbat tarjimoni —
          // asosiy qiymati tezlik (deyarli real vaqtda tarjima). Ikki turn
          // deyarli bir vaqtda tugab, biri boshqasidan oldinroq kelib
          // qolishi juda kam uchraydigan holat; shunday bo'lganda ro'yxat
          // bir lahzalik "o'z-o'zini to'g'irlashi" (yangi xabar yuqoriroqqa
          // joylashishi) ko'rinishida chiqadi — bu barcha xabarlarni eng
          // sekin tugagan turn tugaguncha umuman ko'rsatmay turishdan
          // ANCHA yaxshiroq (navbatga qo'yish real vaqt tuyg'usini
          // buzadi).
          setMessages((prev) => {
            const next = [
              ...prev,
              {
                seq: msg.seq,
                speaker: msg.speaker,
                originalText: msg.original,
                translation: msg.translation,
                sourceLang: msg.source_lang,
                targetLang: msg.target_lang,
              },
            ];
            next.sort((a, b) => a.seq - b.seq);
            return next;
          });
          break;

        case "turn_skipped":
          console.warn("[useDialogSession] turn skipped", msg.reason);
          setNotice("Ovoz aniq eshitilmadi — yana gapirib ko'ring.");
          if (noticeTimeoutRef.current) {
            clearTimeout(noticeTimeoutRef.current);
          }
          noticeTimeoutRef.current = setTimeout(() => {
            setNotice(null);
            noticeTimeoutRef.current = null;
          }, 3000);
          break;

        default:
          console.warn("[useDialogSession] unknown status", msg);
      }
    },
    [startListenLoop]
  );

  const startDialog = useCallback(async () => {
    if (wsRef.current) {
      return;
    }

    setError(null);
    setMessages([]);
    setCurrentSpeaker(null);
    setSpeakerLangs({});
    stoppedRef.current = false;

    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (err) {
      console.error("[useDialogSession] mic access denied", err);
      setError("Mikrofonga ruxsat berilmadi");
      setPhase("error");
      return;
    }

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onmessage = (event) => handleServerMessage(event.data);

      ws.onerror = (event) => {
        console.error("[useDialogSession] websocket error", event);
        setError("Ulanishda xatolik yuz berdi");
        setPhase("error");
      };

      ws.onclose = (event) => {
        if (!stoppedRef.current) {
          if (!event.wasClean) {
            console.error(
              "[useDialogSession] websocket closed unexpectedly",
              event.code,
              event.reason
            );
            setError("Ulanish kutilmaganda uzildi.");
            setPhase("error");
          } else {
            setPhase("ended");
          }
        }
        stoppedRef.current = true;
        stopListenLoop();
        cleanupMedia();
      };
    } catch (err) {
      console.error("[useDialogSession] failed to open websocket", err);
      setError(err?.message || String(err));
      setPhase("error");
    }
  }, [cleanupMedia, handleServerMessage, stopListenLoop]);

  const selectLanguage = useCallback(
    (lang) => {
      try {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ action: "set_language", lang }));
          if (currentSpeaker) {
            setSpeakerLangs((prev) => ({ ...prev, [currentSpeaker]: lang }));
          }
        } else {
          setError("Ulanish ochiq emas — til yuborilmadi.");
          setPhase("error");
        }
      } catch (err) {
        console.error("[useDialogSession] error sending set_language", err);
        setError(err?.message || String(err));
        setPhase("error");
      }
    },
    [currentSpeaker]
  );

  const stopDialog = useCallback(() => {
    stoppedRef.current = true;

    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "stop_dialog" }));
      }
    } catch (err) {
      console.error("[useDialogSession] error sending stop_dialog", err);
    }

    closeSocket();
    stopListenLoop();
    cleanupMedia();
    setPhase("ended");
    setCurrentSpeaker(null);
    setIsSpeaking(false);
    setAudioLevel(0);
  }, [closeSocket, cleanupMedia, stopListenLoop]);

  // Xavfsizlik to'ri: komponent kutilmaganda unmount bo'lsa ham (masalan
  // foydalanuvchi "to'xtatish"ni bosmasdan orqaga qaytsa), mikrofon va socket
  // doim tozalanadi — sizib chiqishning oldi olinadi.
  useEffect(() => {
    return () => {
      stoppedRef.current = true;
      stopListenLoop();
      cleanupMedia();
      closeSocket();
      if (noticeTimeoutRef.current) {
        clearTimeout(noticeTimeoutRef.current);
        noticeTimeoutRef.current = null;
      }
    };
  }, [cleanupMedia, closeSocket, stopListenLoop]);

  return {
    phase,
    currentSpeaker,
    speakerLangs,
    messages,
    error,
    notice,
    isSpeaking,
    audioLevel,
    startDialog,
    selectLanguage,
    stopDialog,
  };
}
