import { useCallback, useEffect, useRef, useState } from "react";

const WS_URL = "ws://localhost:8000/ws/dialog";

const SPEECH_RMS_THRESHOLD = 0.02;
const SILENCE_DURATION_MS = 2000;
const VAD_CHECK_INTERVAL_MS = 100;

const DEBUG_VAD = false;

// Backend'dagi speaker_id.py MIN_VOICED_SECONDS=1.5dan xavfsiz farqda
// bo'lishi uchun ~4s enrollment yozuvi.
const ENROLL_RECORD_MS = 4000;

export function useDialogSession() {
  const [phase, setPhase] = useState("idle"); // idle | await_language | recording_enroll | enroll_wait | listening | ended | error
  const [currentSpeaker, setCurrentSpeaker] = useState(null); // "A" | "B" | null
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null); // vaqtinchalik "eshitilmadi" kabi xabarlar uchun, error state'dan alohida
  const [isSpeaking, setIsSpeaking] = useState(false);

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
  }, []);

  const startListenLoop = useCallback(() => {
    if (!streamRef.current) return;

    stopListenLoop();

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContextClass();
    const source = audioContext.createMediaStreamSource(streamRef.current);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);

    audioContextRef.current = audioContext;
    analyserRef.current = analyser;

    const dataArray = new Float32Array(analyser.fftSize);
    let silenceStartedAt = null;

    const startTurnRecording = () => {
      if (!streamRef.current || stoppedRef.current) return;

      let recorder;
      try {
        recorder = new MediaRecorder(streamRef.current);
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
    };

    vadIntervalRef.current = setInterval(tick, VAD_CHECK_INTERVAL_MS);
  }, [stopListenLoop]);

  // Enrollment uchun bitta uzluksiz yozuv (~4s), keyin to'xtatib, TO'LIQ
  // blobni bir martada yuboramiz. Bu yerda fragmentatsiya muammosi yo'q,
  // chunki backend butun buferni bir martada (finish_enrollment orqali)
  // dekodlaydi, alohida chunk emas.
  const recordEnrollment = useCallback((speakerLabel) => {
    if (!streamRef.current) {
      setError("Mikrofon oqimi topilmadi.");
      setPhase("error");
      return;
    }

    let recorder;
    try {
      recorder = new MediaRecorder(streamRef.current);
    } catch (err) {
      console.error("[useDialogSession] failed to create enroll recorder", err);
      setError(err?.message || String(err));
      setPhase("error");
      return;
    }

    const localChunks = [];

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        localChunks.push(event.data);
      }
    };

    recorder.onerror = (event) => {
      console.error("[useDialogSession] enroll recorder error", event);
      setError("Ovoz yozishda xatolik yuz berdi.");
      setPhase("error");
    };

    recorder.onstop = () => {
      try {
        const blob = new Blob(localChunks, { type: "audio/webm" });
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(blob);
          wsRef.current.send(JSON.stringify({ action: "finish_enroll" }));
          setPhase("enroll_wait");
        } else {
          setError("Ulanish yopiq — enrollment yuborilmadi.");
          setPhase("error");
        }
      } catch (err) {
        console.error("[useDialogSession] error finishing enrollment", err);
        setError(err?.message || String(err));
        setPhase("error");
      }
    };

    try {
      recorder.start();
      setPhase("recording_enroll");
      setCurrentSpeaker(speakerLabel);
    } catch (err) {
      console.error("[useDialogSession] failed to start enroll recorder", err);
      setError(err?.message || String(err));
      setPhase("error");
      return;
    }

    setTimeout(() => {
      try {
        if (recorder.state !== "inactive") recorder.stop();
      } catch (err) {
        console.error("[useDialogSession] error stopping enroll recorder", err);
      }
    }, ENROLL_RECORD_MS);
  }, []);

  const retryEnrollment = useCallback(() => {
    if (currentSpeaker) {
      recordEnrollment(currentSpeaker);
    }
  }, [currentSpeaker, recordEnrollment]);

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
        case "enroll_prompt":
          setCurrentSpeaker(msg.speaker || null);
          setPhase("await_language");
          break;

        case "awaiting_enroll_audio":
          recordEnrollment(msg.speaker);
          break;

        case "enroll_failed":
          setError(`${msg.speaker || "Foydalanuvchi"} ovozini ro'yxatga olish muvaffaqiyatsiz tugadi. Qaytadan urinib ko'ring.`);
          setCurrentSpeaker(msg.speaker || null);
          setPhase("enroll_failed");
          break;

        case "enrolled":
          // enroll_wait holatida qolamiz — keyingi "enroll_prompt" (B uchun)
          // yoki "listening_started" xabarini kutamiz.
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
          setMessages((prev) => [
            ...prev,
            {
              speaker: msg.speaker,
              originalText: msg.original,
              translation: msg.translation,
              sourceLang: msg.source_lang,
              targetLang: msg.target_lang,
            },
          ]);
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
    [recordEnrollment, startListenLoop]
  );

  const startDialog = useCallback(async () => {
    setError(null);
    setMessages([]);
    setCurrentSpeaker(null);
    stoppedRef.current = false;

    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: true,
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

      // Backend enroll_a fazasida ulanganda hech qanday boshlang'ich xabar
      // yubormaydi, shuning uchun birinchi so'zlovchi (A) uchun til so'rash
      // ekranini frontend o'zi ochadi.
      ws.onopen = () => {
        setCurrentSpeaker("A");
        setPhase("await_language");
      };

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

  const selectLanguage = useCallback((lang) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "set_language", lang }));
      } else {
        setError("Ulanish ochiq emas — til yuborilmadi.");
        setPhase("error");
      }
    } catch (err) {
      console.error("[useDialogSession] error sending set_language", err);
      setError(err?.message || String(err));
      setPhase("error");
    }
  }, []);

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
    messages,
    error,
    notice,
    isSpeaking,
    startDialog,
    selectLanguage,
    stopDialog,
    retryEnrollment,
  };
}
