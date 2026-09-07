import { useRef, useState, useCallback } from "react";

const WS_URL = "ws://localhost:8000/ws/transcribe";

// How long to wait after sending the final blob for the backend to deliver
// its last (most accurate) transcription before we close the socket.
const FINAL_RESULT_GRACE_MS = 1200;

export function useLiveTranscription() {
  const [liveText, setLiveText] = useState("");

  // Mirror of the latest transcription text, updated synchronously in the
  // WebSocket onmessage handler. React state (`liveText`) updates async and is
  // captured by stale closures, so the processing flow must read this ref.
  const liveTextRef = useRef("");

  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);

  // Diagnostics: when the most recent WS transcription message arrived.
  const lastMessageAtRef = useRef(0);

  const start = useCallback(async () => {
    setLiveText("");
    liveTextRef.current = "";
    chunksRef.current = [];

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (typeof data.text === "string") {
          liveTextRef.current = data.text; // synchronous — safe from stale closures
          lastMessageAtRef.current = performance.now();
          setLiveText(data.text); // async — for rendering the live caption only
        }
      } catch (err) {
        console.error(
          "[useLiveTranscription] failed to parse WS message",
          err,
          event.data
        );
      }
    };

    ws.onerror = (event) => {
      console.error("[useLiveTranscription] websocket error", event);
    };

    ws.onclose = (event) => {
      if (!event.wasClean) {
        console.error(
          "[useLiveTranscription] websocket closed unexpectedly",
          event.code,
          event.reason
        );
      }
    };

    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size === 0) return;
      chunksRef.current.push(event.data);
      const fullBlob = new Blob(chunksRef.current, { type: "audio/webm" });
      fullBlob
        .arrayBuffer()
        .then((buffer) => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(buffer);
          }
        })
        .catch((err) => {
          console.error("[useLiveTranscription] failed to send audio chunk", err);
        });
    };

    mediaRecorder.start(1000);
  }, []);

  const stop = useCallback(() => {
    // This Promise ALWAYS resolves. Every cleanup step is guarded, and a
    // watchdog covers the case where mediaRecorder.onstop never fires.
    return new Promise((resolve) => {
      let finished = false;
      let watchdog = null;
      const stopCalledAt = performance.now();
      const textBeforeStop = liveTextRef.current;

      const finish = async () => {
        if (finished) return;
        finished = true;
        if (watchdog) clearTimeout(watchdog);

        let blob = null;
        try {
          blob = new Blob(chunksRef.current, { type: "audio/webm" });

          try {
            streamRef.current?.getTracks().forEach((track) => track.stop());
          } catch (err) {
            console.error(
              "[useLiveTranscription] error stopping media tracks",
              err
            );
          }

          try {
            const ws = wsRef.current;
            if (ws && ws.readyState === WebSocket.OPEN) {
              const buffer = await blob.arrayBuffer();
              ws.send(buffer);
            }
          } catch (err) {
            console.error(
              "[useLiveTranscription] error sending final audio blob",
              err
            );
          }

          // Short window for the backend to send its final transcription.
          await new Promise((r) => setTimeout(r, FINAL_RESULT_GRACE_MS));

          try {
            wsRef.current?.close();
          } catch (err) {
            console.error(
              "[useLiveTranscription] error closing websocket",
              err
            );
          }
        } catch (err) {
          console.error(
            "[useLiveTranscription] unexpected error during stop()",
            err
          );
        } finally {
          mediaRecorderRef.current = null;

          const finalText = (liveTextRef.current || "").trim();
          const gotNewFinal =
            lastMessageAtRef.current > stopCalledAt &&
            liveTextRef.current !== textBeforeStop;
          console.log(
            `[useLiveTranscription] stop() finished in ` +
              `${(performance.now() - stopCalledAt).toFixed(0)}ms | ` +
              `grace=${FINAL_RESULT_GRACE_MS}ms | ` +
              `final WS message arrived after stop()? ${gotNewFinal} | ` +
              `text length=${finalText.length}` +
              (gotNewFinal
                ? ""
                : " (=> using last text from DURING recording; backend FINAL pass " +
                  "did not return before the socket closed)")
          );

          resolve({ blob, text: finalText });
        }
      };

      const mediaRecorder = mediaRecorderRef.current;

      // Nothing to wait on (never started, or already stopped) — finish now.
      if (!mediaRecorder || mediaRecorder.state === "inactive") {
        finish();
        return;
      }

      mediaRecorder.onstop = finish;

      // Belt-and-suspenders: if onstop never fires, force cleanup anyway so
      // the caller's `await stop()` can never hang forever.
      watchdog = setTimeout(() => {
        console.error(
          "[useLiveTranscription] mediaRecorder.onstop did not fire within timeout — forcing cleanup"
        );
        finish();
      }, FINAL_RESULT_GRACE_MS + 3000);

      try {
        mediaRecorder.stop();
      } catch (err) {
        console.error(
          "[useLiveTranscription] error calling mediaRecorder.stop()",
          err
        );
        finish(); // onstop won't fire after a throw — run cleanup ourselves
      }
    });
  }, []);

  return { liveText, start, stop };
}
