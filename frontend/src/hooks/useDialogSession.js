import { useCallback, useEffect, useRef, useState } from "react";
import { getSileroModel, SILERO_FRAME_SAMPLES } from "../lib/sileroVad";

const WS_URL = "ws://localhost:8000/ws/dialog";


const TARGET_SAMPLE_RATE = 16000;

const WORKLET_CHUNK_SAMPLES = 2048;
const CHUNK_DURATION_MS = (WORKLET_CHUNK_SAMPLES / TARGET_SAMPLE_RATE) * 1000; // ~128ms

const LIMITER_PRE_GAIN = 0.9;
const SOFT_CLIP_DRIVE = 1.5;

const NOISE_FLOOR_CALIBRATION_MS = 1000;
const NOISE_FLOOR_MULTIPLIER = 3;
const NOISE_FLOOR_EMA_ALPHA = 0.05;

const NOISE_FLOOR_PERCENTILE = 0.2;
const MAX_NOISE_FLOOR = 0.05;
const MIN_NOISE_FLOOR = 0.003;

const VAD_RMS_SMOOTHING_ALPHA = 0.5; // 0..1, kattaroq = tezroq (kamroq silliqlash)

const NEAR_FIELD_RATIO = 0.35; // 0..1 — joriy spiker darajasining shuncha ulushidan past freym "fon" deb hisoblanadi
const NEAR_START_RATIO = 0.35;
const NEAR_START_MIN_TURNS = 2;
const NEAR_START_HISTORY_SIZE = 20;

const SPEECH_LEVEL_EMA_ALPHA = 0.1; // sekin EMA — bir martalik baland fon-impulsi speechLevel'ni darhol ko'tarib yubormasin

const SPEECH_LEVEL_SEED_MS = 300;

const MIN_SPEECH_START_MS = 200;

const VAD_POS_THRESHOLD = 0.5;
const VAD_NEG_THRESHOLD = 0.35;

const SILENCE_DURATION_MS = 1200;

const PRE_ROLL_MS = 600;
const PRE_ROLL_LOOKBACK_MAX_MS = 3000;
const BURST_GAP_MS = 600;
const PRE_ROLL_CHUNK_COUNT = Math.ceil(PRE_ROLL_LOOKBACK_MAX_MS / CHUNK_DURATION_MS);

const CUT_TAIL_KEEP_MS = 150;
const SILENCE_TAIL_KEEP_MS = 500;

const CHUNK_SOFT_CUT_MIN_S = 3.0;
const SOFT_CUT_SILENCE_MS = 350;
const HARD_CUT_MAX_MS = 10000;

const AUDIO_LEVEL_RMS_MAX = 0.15; // shu RMS qiymatida audioLevel = 1 ga yetadi
const AUDIO_LEVEL_SMOOTHING = 0.4; // 0..1, kattaroq = tezroq (kamroq silliqlash)

const DEBUG_VAD = false;

const SKIP_REASON_MESSAGES = {
  too_short: "Juda qisqa gap — tashlab yuborildi.",
  unknown_speaker: "Ovoz aniq eshitilmadi — yana gapirib ko'ring.",
  empty_transcription: "Hech narsa eshitilmadi.",
  processing_error: "Ichki xatolik yuz berdi — qayta urinib ko'ring.",
  translation_error: "Tarjima qilishda xatolik yuz berdi.",
  empty_audio: "Audio bo'sh keldi — tashlab yuborildi.",
  unsupported_language: "Tanlangan tillardan boshqa tilda gapirildi.",
  foreign_subsegment: "Shu qismi boshqa tilda edi — tashlab yuborildi.",
  weak_lang_unknown_speaker: "Ovoz/til aniq bo'lmadi — tashlab yuborildi.",
  far_field_turn: "Fondagi ovoz — tashlab yuborildi.",
};
const SKIP_REASON_DEFAULT = "Ovoz aniq eshitilmadi — yana gapirib ko'ring.";

function _buildSoftClipCurve(samples = 1024) {
  const curve = new Float32Array(samples);
  const normalizer = Math.tanh(SOFT_CLIP_DRIVE);
  for (let i = 0; i < samples; i++) {
    const x = (i / (samples - 1)) * 2 - 1;
    curve[i] = Math.tanh(x * SOFT_CLIP_DRIVE) / normalizer;
  }
  return curve;
}

function _computeRms(float32Chunk) {
  let sumSquares = 0;
  for (let i = 0; i < float32Chunk.length; i++) {
    sumSquares += float32Chunk[i] * float32Chunk[i];
  }
  return Math.sqrt(sumSquares / float32Chunk.length);
}

function _mergeFloat32Chunks(chunks) {
  let totalLength = 0;
  for (const c of chunks) totalLength += c.length;
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const c of chunks) {
    merged.set(c, offset);
    offset += c.length;
  }
  return merged;
}

export function useDialogSession() {
  const [phase, setPhase] = useState("idle"); // idle | listening | ended | error
  const [messages, setMessages] = useState([]);
  const [speakerEvents, setSpeakerEvents] = useState([]);
  const [speakerColorOrder, setSpeakerColorOrder] = useState({});
  const nextSpeakerColorOrderRef = useRef(0);
  const nextClientSeqRef = useRef(1);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null); // vaqtinchalik "eshitilmadi" kabi xabarlar uchun, error state'dan alohida
  const [audioLevel, setAudioLevel] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const stoppedRef = useRef(false);
  const noticeTimeoutRef = useRef(null);

  const audioContextRef = useRef(null);
  const workletNodeRef = useRef(null);

  const smoothedLevelRef = useRef(0);
  const calibrationStartedAtRef = useRef(null);
  const calibrationSamplesRef = useRef([]);
  const noiseFloorRef = useRef(null);
  const smoothedRmsRef = useRef(0);
  const speechLevelRef = useRef(0);
  const acceptedLevelsRef = useRef([]);
  const sentLevelsRef = useRef(new Map());
  const speechLevelSeedSumRef = useRef(0);
  const speechLevelSeedCountRef = useRef(0);
  const pendingSpeechStartAtRef = useRef(null);
  // {chunk, t} — t: chunk kelgan vaqt (Date.now()).
  const preRollChunksRef = useRef([]);
  const speechBurstStartAtRef = useRef(null);
  const lastSpeakingAtRef = useRef(null);

  const sileroModelRef = useRef(null);
  const sileroReadyRef = useRef(false);
  const sileroProbRef = useRef(0);
  const sileroSpeakingRef = useRef(false);
  const sileroLeftoverRef = useRef(new Float32Array(0));
  const sileroChainRef = useRef(Promise.resolve());
  const turnChunksRef = useRef(null); // null = hozir faol turn yo'q
  const turnStartedAtRef = useRef(null);
  const silenceStartedAtRef = useRef(null);
  const micPausedRef = useRef(false);
  const lastDiagLogAtRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    getSileroModel()
      .then((model) => {
        if (cancelled) return;
        sileroModelRef.current = model;
        sileroReadyRef.current = true;
        console.log("[useDialogSession] Silero VAD modeli yuklandi — energiya VAD zaxiraga o'tkazildi");
      })
      .catch((err) => {
        console.warn(
          "[useDialogSession] Silero VAD modelini yuklab bo'lmadi — energiya (RMS) VAD'ga qaytildi",
          err
        );
      });
    return () => {
      cancelled = true;
    };
  }, []);

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

  const resetVadState = useCallback(() => {
    smoothedLevelRef.current = 0;
    calibrationStartedAtRef.current = null;
    calibrationSamplesRef.current = [];
    noiseFloorRef.current = null;
    smoothedRmsRef.current = 0;
    speechLevelRef.current = 0;
    speechLevelSeedSumRef.current = 0;
    speechLevelSeedCountRef.current = 0;
    pendingSpeechStartAtRef.current = null;
    preRollChunksRef.current = [];
    speechBurstStartAtRef.current = null;
    lastSpeakingAtRef.current = null;
    turnChunksRef.current = null;
    turnStartedAtRef.current = null;
    silenceStartedAtRef.current = null;
    micPausedRef.current = false;
    sileroProbRef.current = 0;
    sileroSpeakingRef.current = false;
    sileroLeftoverRef.current = new Float32Array(0);
    if (sileroModelRef.current) {
      try {
        sileroModelRef.current.reset_state();
      } catch (err) {
        console.error("[useDialogSession] Silero reset_state failed", err);
      }
    }
  }, []);

  const stopListenLoop = useCallback(() => {
    if (workletNodeRef.current) {
      try {
        workletNodeRef.current.port.onmessage = null;
        workletNodeRef.current.disconnect();
      } catch (err) {
        console.error("[useDialogSession] error disconnecting worklet node", err);
      }
      workletNodeRef.current = null;
    }

    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (err) {
        console.error("[useDialogSession] error closing audio context", err);
      }
      audioContextRef.current = null;
    }

    resetVadState();
    setAudioLevel(0);
  }, [resetVadState]);

  function _categorizeFinalizeReason(reason) {
    if (reason === "silence_timeout") return "silence";
    return "max_duration";
  }

  const sendTurnAudio = useCallback((float32Array, durationMs, reason) => {
    const bytes = float32Array.byteLength;
    if (
      !stoppedRef.current &&
      wsRef.current &&
      wsRef.current.readyState === WebSocket.OPEN &&
      float32Array.length > 0
    ) {
      try {
        const wireReason = _categorizeFinalizeReason(reason);
        const clientSeq = nextClientSeqRef.current++;
        sentLevelsRef.current.set(clientSeq, speechLevelRef.current);
        wsRef.current.send(
          JSON.stringify({ action: "turn_meta", reason: wireReason, client_seq: clientSeq })
        );
        wsRef.current.send(float32Array.buffer);
        setMessages((prev) => [
          ...prev,
          {
            seq: null,
            clientSeq,
            placeholder: true,
            speaker: null,
            originalText: "",
            translation: "",
            sourceLang: null,
            targetLang: null,
            uncertain: false,
            mtProvider: null,
            translationError: false,
          },
        ]);
        console.log(
          `[VAD] turn SENT to backend durationMs=${durationMs.toFixed(0)} bytes=${bytes} ` +
            `reason=${reason} wireReason=${wireReason} clientSeq=${clientSeq}`
        );
      } catch (err) {
        console.error("[useDialogSession] error sending turn audio", err);
      }
    } else {
      console.warn(
        `[VAD] turn NOT SENT (ws closed or empty) durationMs=${durationMs.toFixed(0)} bytes=${bytes} ` +
          `wsState=${wsRef.current ? wsRef.current.readyState : "null"} stopped=${stoppedRef.current}`
      );
    }
  }, []);

  const finalizeTurn = useCallback(
    (reason) => {
      const entries = turnChunksRef.current;
      const turnStartedAt = turnStartedAtRef.current;
      const silenceStartedAt = silenceStartedAtRef.current;
      turnChunksRef.current = null;
      turnStartedAtRef.current = null;
      silenceStartedAtRef.current = null;
      let kept = entries || [];
      let carry = [];
      if ((reason === "soft_cut_pause" || reason === "silence_timeout") && silenceStartedAt !== null) {
        const splitAt = silenceStartedAt + (reason === "soft_cut_pause" ? CUT_TAIL_KEEP_MS : SILENCE_TAIL_KEEP_MS);
        kept = kept.filter((e) => e.t <= splitAt);
        carry = entries.filter((e) => e.t > splitAt);
      }
      preRollChunksRef.current = carry.slice(-PRE_ROLL_CHUNK_COUNT);
      if (preRollChunksRef.current.length > 0) {
        speechBurstStartAtRef.current = preRollChunksRef.current[0].t - CHUNK_DURATION_MS;
        lastSpeakingAtRef.current = Date.now();
      } else {
        speechBurstStartAtRef.current = null;
        lastSpeakingAtRef.current = null;
      }
      if (kept.length === 0) return;

      const merged = _mergeFloat32Chunks(kept.map((e) => e.chunk));
      const durationMs = (merged.length / TARGET_SAMPLE_RATE) * 1000;
      console.log(
        `[VAD] speech ENDED at=${new Date().toISOString()} reason=${reason} ` +
          `turnDurationMs=${durationMs.toFixed(0)} wallClockMs=${
            turnStartedAt ? Date.now() - turnStartedAt : "?"
          }`
      );
      if (DEBUG_VAD) {
        console.log(
          `[VAD-DEBUG] turn FINALIZED reason=${reason} samples=${merged.length} ` +
            `durationMs=${durationMs.toFixed(0)}`
        );
      }
      sendTurnAudio(merged, durationMs, reason);
    },
    [sendTurnAudio]
  );

  const queueSileroFrame = useCallback((frame) => {
    sileroChainRef.current = sileroChainRef.current
      .then(() => sileroModelRef.current.process(frame))
      .then((probs) => {
        const prob = probs.isSpeech;
        sileroProbRef.current = prob;
        // Gisterezis — qarang VAD_POS_THRESHOLD/VAD_NEG_THRESHOLD izohi.
        if (!sileroSpeakingRef.current && prob >= VAD_POS_THRESHOLD) {
          sileroSpeakingRef.current = true;
        } else if (sileroSpeakingRef.current && prob < VAD_NEG_THRESHOLD) {
          sileroSpeakingRef.current = false;
        }
      })
      .catch((err) => {
        console.error(
          "[useDialogSession] Silero inference xatosi, energiya VAD'ga qaytildi",
          err
        );
        sileroReadyRef.current = false;
      });
  }, []);

  const feedSilero = useCallback(
    (chunk) => {
      if (!sileroReadyRef.current || !sileroModelRef.current) return;
      const leftover = sileroLeftoverRef.current;
      let data = chunk;
      if (leftover.length > 0) {
        data = new Float32Array(leftover.length + chunk.length);
        data.set(leftover, 0);
        data.set(chunk, leftover.length);
      }
      let offset = 0;
      while (data.length - offset >= SILERO_FRAME_SAMPLES) {
        queueSileroFrame(data.slice(offset, offset + SILERO_FRAME_SAMPLES));
        offset += SILERO_FRAME_SAMPLES;
      }
      sileroLeftoverRef.current = data.slice(offset);
    },
    [queueSileroFrame]
  );

  const handleWorkletChunk = useCallback(
    (chunk) => {
      if (stoppedRef.current) return;
      const now = Date.now();
      const rms = _computeRms(chunk);

      const normalizedLevel = Math.min(1, rms / AUDIO_LEVEL_RMS_MAX);
      smoothedLevelRef.current += (normalizedLevel - smoothedLevelRef.current) * AUDIO_LEVEL_SMOOTHING;
      setAudioLevel(smoothedLevelRef.current);

      // --- C5: TTS ijro etilayotganda hech narsa qabul qilinmaydi/yozilmaydi ---
      if (micPausedRef.current) {
        return;
      }

      feedSilero(chunk);

      if (calibrationStartedAtRef.current === null) {
        calibrationStartedAtRef.current = now;
      }
      const inCalibration = now - calibrationStartedAtRef.current < NOISE_FLOOR_CALIBRATION_MS;
      if (inCalibration) {
        calibrationSamplesRef.current.push(rms);
        return;
      }
      if (noiseFloorRef.current === null) {
        const samples = calibrationSamplesRef.current.slice().sort((a, b) => a - b);
        const idx =
          samples.length > 0
            ? Math.min(samples.length - 1, Math.floor(samples.length * NOISE_FLOOR_PERCENTILE))
            : -1;
        const estimate = idx >= 0 ? samples[idx] : MIN_NOISE_FLOOR;
        noiseFloorRef.current = Math.min(MAX_NOISE_FLOOR, Math.max(MIN_NOISE_FLOOR, estimate));
        if (DEBUG_VAD) {
          console.log(
            `[VAD-DEBUG] noise floor calibrated (p${NOISE_FLOOR_PERCENTILE * 100} of ` +
              `${samples.length} samples): ${noiseFloorRef.current.toFixed(5)}`
          );
        }
      }

      smoothedRmsRef.current += VAD_RMS_SMOOTHING_ALPHA * (rms - smoothedRmsRef.current);

      const dynamicThreshold = Math.max(MIN_NOISE_FLOOR, noiseFloorRef.current) * NOISE_FLOOR_MULTIPLIER;
      const vadSource = sileroReadyRef.current ? "silero" : "energy";
      const speaking = sileroReadyRef.current
        ? sileroSpeakingRef.current
        : smoothedRmsRef.current >= dynamicThreshold;
      const hasActiveTurn = turnChunksRef.current !== null;
      let startAllowed = speaking;
      const levels = acceptedLevelsRef.current;
      if (!hasActiveTurn && speaking && levels.length >= NEAR_START_MIN_TURNS) {
        const sorted = [...levels].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        // Silliqlangan RMS — bitta past bo'g'in/undosh freymi turn boshlanishini qayta-qayta bekor qilmasin.
        startAllowed = smoothedRmsRef.current >= NEAR_START_RATIO * median;
        if (!startAllowed) {
          console.log(`[VAD] start rejected reason=far_field rms=${rms.toFixed(5)} median=${median.toFixed(5)}`);
        }
      }

      let nearThreshold = dynamicThreshold;
      let nearFieldSpeaking = speaking;
      let silenceReason = speaking ? "above_threshold" : "below_threshold";
      if (hasActiveTurn) {
        const turnAgeMs = now - turnStartedAtRef.current;
        if (turnAgeMs < SPEECH_LEVEL_SEED_MS) {
          speechLevelSeedSumRef.current += smoothedRmsRef.current;
          speechLevelSeedCountRef.current += 1;
          speechLevelRef.current = speechLevelSeedSumRef.current / speechLevelSeedCountRef.current;
          nearThreshold = Math.max(dynamicThreshold, speechLevelRef.current * NEAR_FIELD_RATIO);
          nearFieldSpeaking = speaking;
          silenceReason = speaking ? "seeding_speech" : "seeding_silence";
        } else {
          nearThreshold = Math.max(dynamicThreshold, speechLevelRef.current * NEAR_FIELD_RATIO);
          const rmsNearOk = smoothedRmsRef.current >= nearThreshold;
          nearFieldSpeaking = speaking && rmsNearOk;
          silenceReason = !speaking ? "vad_silence" : rmsNearOk ? "near_field_speech" : "far_field_background";
          if (speaking) {
            speechLevelRef.current += SPEECH_LEVEL_EMA_ALPHA * (smoothedRmsRef.current - speechLevelRef.current);
          }
        }
      }

      // Diagnostika: har 2 soniyada joriy holatni chiqaramiz.
      if (now - lastDiagLogAtRef.current >= 2000) {
        lastDiagLogAtRef.current = now;
        console.log(
          `[VAD] status rms=${rms.toFixed(5)} smoothedRms=${smoothedRmsRef.current.toFixed(5)} ` +
            `noiseFloor=${noiseFloorRef.current.toFixed(5)} threshold=${dynamicThreshold.toFixed(5)} ` +
            `speechLevel=${speechLevelRef.current.toFixed(5)} nearThreshold=${nearThreshold.toFixed(5)} ` +
            `prob=${sileroProbRef.current.toFixed(3)} vadSource=${vadSource} ` +
            `frame=${nearFieldSpeaking ? "speech" : "silence"}(${silenceReason}) ` +
            `turn=${hasActiveTurn ? "recording" : "idle"}`
        );
      }

      if (!speaking && !hasActiveTurn) {
        const updated = noiseFloorRef.current + NOISE_FLOOR_EMA_ALPHA * (smoothedRmsRef.current - noiseFloorRef.current);
        noiseFloorRef.current = Math.min(MAX_NOISE_FLOOR, Math.max(MIN_NOISE_FLOOR, updated));
      }

      if (DEBUG_VAD) {
        console.log(
          `[VAD-DEBUG] rms=${rms.toFixed(5)} smoothedRms=${smoothedRmsRef.current.toFixed(5)} ` +
            `noiseFloor=${noiseFloorRef.current.toFixed(5)} threshold=${dynamicThreshold.toFixed(5)} ` +
            `speechLevel=${speechLevelRef.current.toFixed(5)} nearThreshold=${nearThreshold.toFixed(5)} ` +
            `prob=${sileroProbRef.current.toFixed(3)} vadSource=${vadSource} ` +
            `speaking=${speaking} nearFieldSpeaking=${nearFieldSpeaking} activeTurn=${hasActiveTurn}`
        );
      }

      preRollChunksRef.current.push({ chunk, t: now });
      if (preRollChunksRef.current.length > PRE_ROLL_CHUNK_COUNT) {
        preRollChunksRef.current.shift();
      }
      if (!hasActiveTurn) {
        if (speaking) {
          if (speechBurstStartAtRef.current === null) speechBurstStartAtRef.current = now;
          lastSpeakingAtRef.current = now;
        } else if (lastSpeakingAtRef.current !== null && now - lastSpeakingAtRef.current > BURST_GAP_MS) {
          speechBurstStartAtRef.current = null;
          lastSpeakingAtRef.current = null;
        }
      }

      if (!hasActiveTurn) {
        if (startAllowed) {
          if (pendingSpeechStartAtRef.current === null) {
            pendingSpeechStartAtRef.current = now;
          }
          const pendingElapsedMs = now - pendingSpeechStartAtRef.current;
          if (pendingElapsedMs >= MIN_SPEECH_START_MS) {
            const speechStartAt = Math.min(
              pendingSpeechStartAtRef.current,
              speechBurstStartAtRef.current ?? pendingSpeechStartAtRef.current
            );
            const takeFrom = speechStartAt - PRE_ROLL_MS;
            turnChunksRef.current = preRollChunksRef.current.filter((e) => e.t > takeFrom);
            speechBurstStartAtRef.current = null;
            lastSpeakingAtRef.current = null;
            turnStartedAtRef.current = pendingSpeechStartAtRef.current;
            silenceStartedAtRef.current = null;
            speechLevelRef.current = smoothedRmsRef.current;
            speechLevelSeedSumRef.current = smoothedRmsRef.current;
            speechLevelSeedCountRef.current = 1;
            pendingSpeechStartAtRef.current = null;
            console.log(
              `[VAD] speech STARTED at=${new Date(now).toISOString()} rms=${rms.toFixed(5)} ` +
                `smoothedRms=${smoothedRmsRef.current.toFixed(5)} threshold=${dynamicThreshold.toFixed(5)} ` +
                `prob=${sileroProbRef.current.toFixed(3)} vadSource=${vadSource} ` +
                `confirmedAfterMs=${pendingElapsedMs.toFixed(0)}`
            );
            if (DEBUG_VAD) {
              console.log(`[VAD-DEBUG] turn STARTED (preRollChunks=${preRollChunksRef.current.length})`);
            }
          }
        } else {
          pendingSpeechStartAtRef.current = null;
        }
        return;
      }

      turnChunksRef.current.push({ chunk, t: now });

      if (nearFieldSpeaking) {
        silenceStartedAtRef.current = null;
      } else if (silenceStartedAtRef.current === null) {
        silenceStartedAtRef.current = now;
      }

      const turnElapsedMs = now - turnStartedAtRef.current;
      const silenceElapsedMs = silenceStartedAtRef.current === null ? 0 : now - silenceStartedAtRef.current;

      if (silenceElapsedMs >= SILENCE_DURATION_MS) {
        finalizeTurn("silence_timeout");
      } else if (turnElapsedMs >= CHUNK_SOFT_CUT_MIN_S * 1000 && silenceElapsedMs >= SOFT_CUT_SILENCE_MS) {
        finalizeTurn("soft_cut_pause");
      } else if (turnElapsedMs >= HARD_CUT_MAX_MS) {
        finalizeTurn("hard_cut_max_duration");
      }
    },
    [finalizeTurn, feedSilero]
  );

  const startListenLoop = useCallback(async () => {
    if (!streamRef.current) return;

    stopListenLoop();

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    let audioContext;
    try {
      audioContext = new AudioContextClass({ sampleRate: TARGET_SAMPLE_RATE });
    } catch (err) {
      console.error(
        "[useDialogSession] failed to create AudioContext at 16kHz, falling back to default rate",
        err
      );
      audioContext = new AudioContextClass();
    }

    try {
      await audioContext.resume();
    } catch (err) {
      console.error("[useDialogSession] audioContext.resume() failed", err);
    }
    if (audioContext.state !== "running") {
      console.error(
        `[useDialogSession] AudioContext holati "${audioContext.state}" ("running" kutilgan) — ` +
          "mikrofon tinglanmaydi."
      );
      setError("Audio tizimini ishga tushirishda xatolik yuz berdi — sahifani qayta yuklab ko'ring.");
      setPhase("error");
      try {
        audioContext.close();
      } catch {
        // e'tiborsiz qoldiriladi
      }
      return;
    }

    if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
      console.error(
        `[useDialogSession] AudioContext sampleRate=${audioContext.sampleRate}, kutilgan ` +
          `${TARGET_SAMPLE_RATE} — backend'ga yuborilayotgan audio tezligi NOTO'G'RI bo'lishi mumkin!`
      );
    }

    try {
      await audioContext.audioWorklet.addModule("/pcm-worklet.js");
    } catch (err) {
      console.error("[useDialogSession] failed to load pcm-worklet.js", err);
      setError("Audio modulini yuklashda xatolik yuz berdi.");
      setPhase("error");
      try {
        audioContext.close();
      } catch {
        // e'tiborsiz qoldiriladi
      }
      return;
    }

    if (stoppedRef.current) {
      try {
        audioContext.close();
      } catch {
        // e'tiborsiz qoldiriladi
      }
      return;
    }

    const source = audioContext.createMediaStreamSource(streamRef.current);

    const limiterGain = audioContext.createGain();
    limiterGain.gain.value = LIMITER_PRE_GAIN;
    const limiterShaper = audioContext.createWaveShaper();
    limiterShaper.curve = _buildSoftClipCurve();
    source.connect(limiterGain);
    limiterGain.connect(limiterShaper);

    const workletNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");
    limiterShaper.connect(workletNode);

    const silentSink = audioContext.createGain();
    silentSink.gain.value = 0;
    workletNode.connect(silentSink);
    silentSink.connect(audioContext.destination);

    workletNode.port.onmessage = (event) => handleWorkletChunk(event.data);

    audioContextRef.current = audioContext;
    workletNodeRef.current = workletNode;

    if (DEBUG_VAD) {
      console.log(`[VAD-DEBUG] listen loop STARTED, actual sampleRate=${audioContext.sampleRate}`);
    }
  }, [handleWorkletChunk, stopListenLoop]);

  const pauseMic = useCallback(
    (reason = "unknown") => {
      if (micPausedRef.current) {
        console.log(`[VAD] pauseMic called again while already paused reason=${reason}`);
        return;
      }
      micPausedRef.current = true;
      console.log(`[VAD] pauseMic reason=${reason} at=${new Date().toISOString()}`);
      if (turnChunksRef.current !== null) {
        finalizeTurn("mic_paused_for_tts");
      }
      preRollChunksRef.current = [];
      speechBurstStartAtRef.current = null;
      lastSpeakingAtRef.current = null;
    },
    [finalizeTurn]
  );

  const resumeMic = useCallback((reason = "unknown") => {
    if (!micPausedRef.current) {
      console.log(`[VAD] resumeMic called while not paused reason=${reason}`);
    }
    micPausedRef.current = false;
    sileroLeftoverRef.current = new Float32Array(0);
    sileroProbRef.current = 0;
    sileroSpeakingRef.current = false;
    if (sileroModelRef.current) {
      try {
        sileroModelRef.current.reset_state();
      } catch (err) {
        console.error("[useDialogSession] Silero reset_state failed", err);
      }
    }
    console.log(`[VAD] resumeMic reason=${reason} at=${new Date().toISOString()}`);
  }, []);

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
        case "listening_started":
          stoppedRef.current = false;
          setPhase("listening");
          startListenLoop();
          break;

        case "processing":
          setIsProcessing(true);
          break;

        case "turn":
          for (const clientSeq of msg.client_seqs || []) {
            const level = sentLevelsRef.current.get(clientSeq);
            if (msg.speaker != null && level > 0) {
              acceptedLevelsRef.current = [...acceptedLevelsRef.current, level].slice(-NEAR_START_HISTORY_SIZE);
              sentLevelsRef.current.delete(clientSeq);
            }
          }
          setNotice(null);
          setIsProcessing(false);
          if (noticeTimeoutRef.current) {
            clearTimeout(noticeTimeoutRef.current);
            noticeTimeoutRef.current = null;
          }
          setMessages((prev) => {
            const clientSeqSet = new Set(Array.isArray(msg.client_seqs) ? msg.client_seqs : []);
            const withoutPlaceholders = prev.filter(
              (m) => !(m.placeholder && clientSeqSet.has(m.clientSeq))
            );
            const entry = {
              seq: msg.seq,
              clientSeq: null,
              placeholder: false,
              speaker: msg.speaker,
              originalText: msg.original,
              translation: msg.translation,
              sourceLang: msg.source_lang,
              targetLang: msg.target_lang,
              uncertain: Boolean(msg.uncertain),
              mtProvider: msg.mt_provider ?? null,
              translationError: Boolean(msg.translation_error),
            };
            const existingIdx = withoutPlaceholders.findIndex((m) => m.seq === msg.seq);
            const next =
              existingIdx === -1
                ? [...withoutPlaceholders, entry]
                : withoutPlaceholders.map((m, i) => (i === existingIdx ? entry : m));
            next.sort((a, b) => {
              if (a.placeholder && b.placeholder) return a.clientSeq - b.clientSeq;
              if (a.placeholder) return 1;
              if (b.placeholder) return -1;
              return a.seq - b.seq;
            });
            return next;
          });

          setSpeakerColorOrder((prev) => {
            if (msg.speaker in prev) return prev;
            const order = nextSpeakerColorOrderRef.current;
            nextSpeakerColorOrderRef.current += 1;
            return { ...prev, [msg.speaker]: order };
          });
          break;

        case "turn_deleted":
          setMessages((prev) => prev.filter((m) => m.seq !== msg.seq));
          break;

        case "speakers_merged": {
          const { from, into } = msg;
          setMessages((prev) => prev.map((m) => (m.speaker === from ? { ...m, speaker: into } : m)));
          setSpeakerEvents((prev) => [...prev, { type: "merged", from, into }]);
          break;
        }

        case "speaker_removed":
          setSpeakerEvents((prev) => [...prev, { type: "removed", speaker: msg.speaker }]);
          break;

        case "turn_skipped": {
          for (const clientSeq of msg.client_seqs || []) sentLevelsRef.current.delete(clientSeq);
          console.warn("[useDialogSession] turn skipped", msg.reason);
          setIsProcessing(false);
          const skippedClientSeqSet = new Set(
            Array.isArray(msg.client_seqs) ? msg.client_seqs : []
          );
          setMessages((prev) =>
            prev.filter((m) => !(m.placeholder && skippedClientSeqSet.has(m.clientSeq)))
          );
          setNotice(SKIP_REASON_MESSAGES[msg.reason] || SKIP_REASON_DEFAULT);
          if (noticeTimeoutRef.current) {
            clearTimeout(noticeTimeoutRef.current);
          }
          noticeTimeoutRef.current = setTimeout(() => {
            setNotice(null);
            noticeTimeoutRef.current = null;
          }, 3000);
          break;
        }

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
    setSpeakerColorOrder({});
    nextSpeakerColorOrderRef.current = 0;
    nextClientSeqRef.current = 1;
    acceptedLevelsRef.current = [];
    sentLevelsRef.current.clear();
    stoppedRef.current = false;

    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: false,
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

  const correctSpeaker = useCallback((seq, targetSpeaker) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({ action: "correct_speaker", seq, target_speaker: targetSpeaker })
        );
      }
    } catch (err) {
      console.error("[useDialogSession] error sending correct_speaker", err);
    }
  }, []);

  const deleteTurn = useCallback((seq) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "delete_turn", seq }));
      }
    } catch (err) {
      console.error("[useDialogSession] error sending delete_turn", err);
    }
  }, []);

  const renameSpeaker = useCallback((speaker, name) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "rename_speaker", speaker, name }));
      }
    } catch (err) {
      console.error("[useDialogSession] error sending rename_speaker", err);
    }
  }, []);

  const mergeSpeakers = useCallback((from, into) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "merge_speakers", from, into }));
      }
    } catch (err) {
      console.error("[useDialogSession] error sending merge_speakers", err);
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
    setAudioLevel(0);
  }, [closeSocket, cleanupMedia, stopListenLoop]);

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
    // C5: DialogSessionView.jsx TTS pleybek boshlanishi/tugashiga ulaydi.
    pauseMic,
    resumeMic,
  };
}
