# Voice Translator

A real-time speech translation app with two modes:

- **Solo mode** — one person speaks, their speech is transcribed live and translated into a target language.
- **Dialog mode** — two people have a conversation; the app automatically detects who is speaking, transcribes their turn, and translates it into the other person's language.

## Tech stack

- **Backend:** FastAPI (Python), served over both REST and WebSocket endpoints.
- **Frontend:** React + Vite, using the browser's `MediaRecorder` and Web Audio API (`AnalyserNode`) for microphone capture and voice-activity detection.
- **Containerization:** Docker + Docker Compose, with NVIDIA GPU passthrough (tested under WSL2).
- **Models:**
  - **Speech-to-text:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`medium` size, float16, CUDA).
  - **Translation:** [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) (NLLB-200).
  - **Text-to-speech:** [`facebook/mms-tts-*`](https://huggingface.co/facebook/mms-tts) (one model per target language).
  - **Speaker identification:** [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) (used only in Dialog mode, to tell the two speakers apart).
- **Supported languages:** English (`en`), Uzbek (`uz`), Korean (`ko`), Russian (`ru`) — for both speech recognition and translation/text-to-speech. Whisper (`faster-whisper`, `medium`) is multilingual and transcribes in whichever of these four languages the speaker selects (it is not English-only); that selected language is then translated into whichever of the four the listener/other speaker uses.

## Requirements

- Docker and Docker Compose (Compose v2 syntax — no `version:` key needed).
- An NVIDIA GPU with `nvidia-container-toolkit` installed (or GPU passthrough enabled under WSL2 on Windows).
- **VRAM:** the app has been developed and tested on an **RTX 4050 with 6GB VRAM**, running Whisper (medium), NLLB-200-distilled-600M, four MMS-TTS models, and Resemblyzer concurrently. Treat ~6GB as the practical minimum; more headroom is safer if you plan to run other GPU workloads at the same time.
- A working microphone in the browser you use to open the app.

## Installation and running

```bash
git clone <this-repo-url>
cd voice-translator
docker compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

**First run note:** the backend downloads and caches all models (Whisper, NLLB, MMS-TTS, Resemblyzer) from Hugging Face on first startup — this can take several minutes and roughly 2.5GB of disk space. Downloaded weights are cached in the `hf-cache` Docker volume, so subsequent restarts only need to *load* the models (not re-download), which still takes on the order of a minute while the GPU warms up. The backend's `/` health check has a 600-second start period specifically to tolerate this on first boot.

## Solo mode

Solo mode is for one person translating their own speech.

1. The frontend opens a WebSocket to **`/ws/transcribe`** (`backend/app/routers/live_transcribe.py`), passing the spoken language as a query parameter.
2. The browser (`frontend/src/hooks/useLiveTranscription.js`) records continuously with a single `MediaRecorder`, and every second sends the backend the full audio recorded so far. The backend re-transcribes that growing clip on each poll (`DIALOG_PROCESS_INTERVAL`, see below) and streams the text back, so a live (partial) transcription is shown while you're still talking.
3. When you stop recording, the complete clip is sent one final time, the backend transcribes it in full, and that result is used as the final transcription before the socket closes.
4. The resulting text is sent to `POST /translate` (`backend/app/routers/translate.py`) to get the translation, and optionally to `POST /speak` (`backend/app/routers/speak.py`) to synthesize and play back spoken audio in the target language.

## Dialog mode

Dialog mode is for a two-person conversation, connecting to **`/ws/dialog`** (`backend/app/routers/dialog.py`).

1. **Enrollment:** each of the two speakers, in turn, picks their spoken language and records a short (~4 second) voice sample. The backend (`backend/app/models/speaker_id.py`) builds a voice embedding (via Resemblyzer) for each speaker from that sample.
2. **Listening:** once both speakers are enrolled, the app moves into continuous listening. The frontend uses the Web Audio API (`AudioContext` + `AnalyserNode`) to measure microphone volume (RMS) roughly 10 times a second — this is a **client-side voice-activity detector (VAD)**. It requires no server round-trip to know whether someone is currently speaking.
3. **Turn detection:** when volume crosses the speech threshold, the frontend starts a single continuous `MediaRecorder` recording for that turn. When volume stays below the threshold for long enough (silence), the recorder is stopped and the complete audio blob for that turn is sent to the backend in one piece — this avoids the audio-quality problems that come from stitching together many small, independently-encoded recordings.
4. **Per-turn processing (backend):** the backend decodes the full turn, checks it isn't too short (likely noise), identifies which of the two enrolled speakers it best matches (Resemblyzer), transcribes it (faster-whisper) in that speaker's language, and translates it (NLLB) into the other speaker's language. The result is pushed back over the WebSocket and rendered as a chat bubble; either side's translated text can also be played back as speech via `POST /speak`.

## Configuration constants

| Constant | File | What it controls |
|---|---|---|
| `SPEECH_RMS_THRESHOLD` | `frontend/src/hooks/useDialogSession.js` | RMS volume level above which the mic is considered "speaking" (client-side VAD, Dialog mode). |
| `SILENCE_DURATION_MS` | `frontend/src/hooks/useDialogSession.js` | How many milliseconds of continuous silence end the current turn and trigger sending it to the backend. |
| `VAD_CHECK_INTERVAL_MS` | `frontend/src/hooks/useDialogSession.js` | How often (ms) the mic's volume is sampled while listening. |
| `DEBUG_VAD` | `frontend/src/hooks/useDialogSession.js` | Set to `true` to log detailed VAD tick/recording events to the browser console (see Debug mode below). |
| `ENROLL_RECORD_MS` | `frontend/src/hooks/useDialogSession.js` | Length (ms) of each speaker's enrollment recording. |
| `NO_SPEECH_PROB_THRESHOLD` | `backend/app/models/stt.py` | Whisper segments with a "no speech" probability above this are treated as noise/silence and dropped. |
| `AVG_LOGPROB_THRESHOLD` | `backend/app/models/stt.py` | Whisper segments with an average log-probability (confidence) below this are dropped. |
| `RMS_SILENCE_THRESHOLD` | `backend/app/models/stt.py` | Below this RMS, an audio clip is treated as silent and transcription is skipped outright. |
| `DEBUG_STT` | `backend/app/models/stt.py` | Set to `True` to print per-segment KEPT/DROPPED confidence details and fallback-transcript usage to the backend logs (see Debug mode below). |
| `MIN_TURN_SECONDS` | `backend/app/config.py` | A recorded Dialog-mode turn shorter than this (seconds) is discarded as noise before transcription is even attempted. |
| `MIN_VOICED_SECONDS` | `backend/app/models/speaker_id.py` | Minimum voiced audio (seconds) required to enroll a speaker or identify one from a turn; shorter clips are treated as unreliable. |
| `UNKNOWN_SPEAKER_THRESHOLD` | `backend/app/config.py` | Minimum Resemblyzer similarity score to attribute a turn to an enrolled speaker; below this, the turn matches neither A nor B and is skipped. |
| `DIALOG_PROCESS_INTERVAL` | `backend/app/config.py` | Poll interval (seconds) used by Solo mode's live-transcription loop (`/ws/transcribe`). |
| `WHISPER_MODEL_SIZE` / `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | `backend/app/config.py` | Which faster-whisper model size, device, and precision to load. |
| `TRANSLATION_MODEL_NAME` | `backend/app/config.py` | Hugging Face model id used for translation (NLLB-200). |
| `TTS_MODEL_NAMES` | `backend/app/config.py` | Hugging Face model id per language used for text-to-speech. |

## Project structure

```
backend/
  app/
    main.py                 # FastAPI app, router registration, model warmup on startup
    config.py                # Shared configuration constants
    models/
      stt.py                 # faster-whisper wrapper (transcription, segment filtering)
      translator.py          # NLLB-200 wrapper (translation)
      tts.py                  # MMS-TTS wrapper (speech synthesis, uz/ko/ru transliteration helpers)
      speaker_id.py           # Resemblyzer wrapper (enrollment + speaker identification)
    routers/
      dialog.py               # WebSocket /ws/dialog — Dialog mode (enrollment + turn-based conversation)
      live_transcribe.py       # WebSocket /ws/transcribe — Solo mode (live transcription)
      translate.py             # POST /translate
      speak.py                  # POST /speak (text-to-speech)
  Dockerfile
  requirements.txt

frontend/
  src/
    App.jsx                    # Mode selection (Solo vs Dialog)
    main.jsx
    components/
      ModeSelector.jsx          # Solo / Dialog picker
      ConversationView.jsx       # Solo mode UI
      DialogSessionView.jsx       # Dialog mode UI
      LanguageSelector.jsx
      MicButton.jsx
      TranscriptLine.jsx           # Solo mode transcript row
      DialogBubble.jsx              # Dialog mode chat bubble
      AnimatedBackground.jsx
    hooks/
      useLiveTranscription.js       # Solo mode: mic capture + /ws/transcribe client
      useDialogSession.js            # Dialog mode: mic capture, client-side VAD, /ws/dialog client
      usePlayAudio.js                  # Play/pause toggling for a synthesized-speech audio blob
    services/
      api.js                          # REST calls: /translate, /speak
  Dockerfile
  package.json

docker-compose.yml
```

## Debug mode

Both the frontend VAD logic and the backend transcription filter support optional, verbose debug logging that is off by default.

- **Frontend (Dialog mode VAD):** in `frontend/src/hooks/useDialogSession.js`, set `DEBUG_VAD = true`. This logs every VAD sampling tick (RMS value, speaking/silent state, whether a turn recording is active) and every turn start/stop event to the browser's developer console, prefixed with `[VAD-DEBUG]`. Rebuild the frontend (`docker compose build frontend && docker compose up -d frontend`) for the change to take effect.
- **Backend (transcription filtering):** in `backend/app/models/stt.py`, set `DEBUG_STT = True`. This prints, for every Whisper segment, whether it was kept or dropped and why (`no_speech_prob` / `avg_logprob` values), plus a note whenever the raw-transcript fallback is used because all segments were filtered out. Logs are prefixed with `[STT][DEBUG]` and visible via `docker logs voice-translator-backend-1`. Rebuild the backend for the change to take effect.

Remember to set both flags back to `false` / `False` before shipping, since they add non-trivial log volume.
