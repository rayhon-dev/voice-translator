# Voice Translator

A real-time speech translation app with two modes:

- **Solo mode** — one person speaks; their speech is transcribed live and translated into a target language.
- **Dialog mode** — a conversation between two or more people. The app detects who is speaking (no voice enrollment needed), transcribes each turn, and translates it into the other language (English ⇄ Russian).

## Tech stack

- **Backend:** FastAPI (Python), REST + WebSocket endpoints.
- **Frontend:** React + Vite. Microphone audio is captured with an `AudioWorklet` as raw 16 kHz PCM; voice activity is detected in the browser with Silero VAD (`@ricky0123/vad-web`), with an energy-based fallback.
- **Containerization:** Docker + Docker Compose with NVIDIA GPU passthrough (tested under WSL2).
- **Models:**
  - **Speech-to-text:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — `large-v3-turbo` (`deepdml/faster-whisper-large-v3-turbo-ct2`, `int8_float16`, CUDA) via `.env`; `medium` if not configured. ElevenLabs Scribe can be used instead (`STT_PROVIDER=elevenlabs`).
  - **Translation:** [DeepL API](https://www.deepl.com/pro-api) as the primary provider, with [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) as an automatic local fallback.
  - **Text-to-speech:** [`facebook/mms-tts-*`](https://huggingface.co/facebook/mms-tts) (one model per language).
  - **Speaker identification:** [`pyannote/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) voice embeddings on GPU, with `speechbrain/spkrec-ecapa-voxceleb` as a fallback.
- **Languages:** Dialog mode works with English (`en`) and Russian (`ru`). Solo mode supports English, Russian and Korean (`ko`).

## Requirements

- Docker and Docker Compose (Compose v2), **or** Python 3.10 + Node.js for running locally.
- An NVIDIA GPU with `nvidia-container-toolkit` (or GPU passthrough under WSL2).
- **VRAM:** developed and tested on an **RTX 4050 (6 GB)** running Whisper, NLLB-600M, MMS-TTS and the speaker-embedding model together. Treat ~6 GB as the practical minimum.
- A DeepL API key (the free tier is enough). Without it, translation falls back to NLLB.
- A working microphone in the browser.

## Configuration (`backend/.env`)

```env
DEEPL_API_KEY=...                 # DeepL translation (primary)
MT_PROVIDER=deepl                 # deepl | nllb
WHISPER_MODEL_SIZE=deepdml/faster-whisper-large-v3-turbo-ct2
WHISPER_COMPUTE_TYPE=int8_float16
STT_PROVIDER=whisper              # whisper | elevenlabs
ELEVENLABS_API_KEY=...            # only for STT_PROVIDER=elevenlabs
HF_TOKEN=...                      # Hugging Face token, if the pyannote model requires it
```

## Installation and running

**With Docker:**

```bash
git clone <this-repo-url>
cd voice-translator
docker compose up --build
```

**Locally (development):**

```bash
cd backend
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn app.main:app --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

**First run note:** the backend downloads and caches all models from Hugging Face on first start, which can take several minutes. Every start then loads the models onto the GPU before accepting requests (about a minute); wait for `All models ready` in the backend log before opening the app.

## Solo mode

1. The frontend opens a WebSocket to **`/ws/transcribe`** (`backend/app/routers/live_transcribe.py`) with the spoken language.
2. `frontend/src/hooks/useLiveTranscription.js` records continuously and periodically sends the audio recorded so far; the backend re-transcribes it and streams back a live partial transcript.
3. When recording stops, the full clip is transcribed one final time.
4. The text is translated with `POST /translate` and can be played back with `POST /speak`.

## Dialog mode

Dialog mode connects to **`/ws/dialog`** (`backend/app/routers/dialog.py`).

1. **Turn detection (browser, `useDialogSession.js`):** Silero VAD decides when someone is speaking. A turn ends after a pause of `SILENCE_DURATION_MS`; long continuous speech is split at the first short pause after `CHUNK_SOFT_CUT_MIN_S`. Each turn starts from the actual beginning of speech (pre-roll / look-back buffer), and a turn is cut where the silence begins, so the first word of the next sentence is never lost. A relative "near-field" check ignores distant background voices.
2. **Segmentation (`segmenter.py`):** a turn that contains two different voices is split by comparing voice embeddings; splits that turn out to be the same known speaker are merged back.
3. **Language detection and transcription (`stt.py`):** Whisper's language-ID is restricted to the two session languages. Leading silence is trimmed before detection, related languages (Ukrainian, Polish, Bulgarian, …) count toward Russian, and ambiguous turns are decoded in both languages so the more confident result wins. Speech in other languages (e.g. Uzbek, Korean) is rejected instead of being mistranslated. Low-confidence hallucinations are filtered out.
4. **Speaker identification (`speaker_diart.py`):** each turn's voice embedding is compared with the known speakers' voice prototypes. A new voice gets its own speaker number once a second turn confirms it (until then it shows as "Aniqlanmoqda…"); short turns fall back to the most recent speaker when the voice clearly matches.
5. **Translation and display:** consecutive turns of the same speaker are merged retroactively, translated (DeepL → NLLB fallback), and shown as chat bubbles. Each bubble can be played back as speech, assigned to another speaker, deleted, and speakers can be renamed.

## Main tuning constants

| Constant | File | What it controls |
|---|---|---|
| `SILENCE_DURATION_MS` | `frontend/src/hooks/useDialogSession.js` | Pause length (ms) that ends a turn. |
| `PRE_ROLL_MS` | `frontend/src/hooks/useDialogSession.js` | Audio kept from before the detected start of speech. |
| `CHUNK_SOFT_CUT_MIN_S` / `SOFT_CUT_SILENCE_MS` / `HARD_CUT_MAX_MS` | `frontend/src/hooks/useDialogSession.js` | How long continuous speech is split into chunks. |
| `VAD_POS_THRESHOLD` | `frontend/src/hooks/useDialogSession.js` | Silero speech-probability threshold. |
| `SESSION_LANGS` | `backend/app/routers/dialog.py` | The two Dialog mode languages. |
| `MIN_PAIR_MASS` / `MIN_PAIR_MASS_SHORT` | `backend/app/models/stt.py` | Minimum language-ID confidence for long / short turns. |
| `LANG_FAMILY` | `backend/app/models/stt.py` | Related languages counted toward a session language. |
| `SPK_ACCEPT` / `SPK_NEW` | `backend/app/models/speaker_diart.py` | Voice-similarity thresholds for matching a known speaker / opening a new one. |
| `SPK_NEW_CONFIRM_TURNS` | `backend/app/models/speaker_diart.py` | Turns needed to confirm a new speaker. |
| `MIN_TURN_SECONDS` | `backend/app/config.py` | Turns shorter than this are discarded as noise. |

Most backend thresholds can also be overridden with environment variables of the same name.

## Project structure

```
backend/
  app/
    main.py                 # FastAPI app, routers, model warmup on startup
    config.py               # Shared configuration
    gpu_lock.py             # Serializes GPU work between models
    models/
      stt.py                # Whisper / ElevenLabs STT, language gating, hallucination filters
      speaker_diart.py      # Voice embeddings and speaker identification
      segmenter.py          # Splits a turn by speaker change
      translator.py         # DeepL + NLLB fallback
      tts.py                # MMS-TTS speech synthesis
    routers/
      dialog.py             # WebSocket /ws/dialog — Dialog mode
      live_transcribe.py    # WebSocket /ws/transcribe — Solo mode
      translate.py          # POST /translate
      speak.py              # POST /speak
  tools/
    test_dialog_noise_regressions.py  # Regression tests (no models needed)
    test_phantom_phrase.py            # Hallucination-phrase filter check
    eval_dialog.py                    # Evaluates speaker/language accuracy on saved debug audio
    mt_bench.py                       # Compares DeepL and NLLB translations
  Dockerfile
  requirements.txt

frontend/
  src/
    App.jsx
    components/
      DialogSessionView.jsx  # Dialog mode UI
      DialogBubble.jsx       # Dialog chat bubble (play, rename, reassign, delete)
      ConversationView.jsx   # Solo mode UI
      ...
    hooks/
      useDialogSession.js    # Dialog mode: audio capture, VAD, turn logic, /ws/dialog client
      useLiveTranscription.js
      usePlayAudio.js
    lib/
      sileroVad.js           # Silero VAD model loader
    services/
      api.js                 # REST calls: /translate, /speak

docker-compose.yml
```

## Testing

```bash
cd backend
venv/bin/python -m unittest tools.test_dialog_noise_regressions
venv/bin/python tools/test_phantom_phrase.py
```

## Debugging

- **Saved audio:** with `SAVE_DEBUG_AUDIO = True` (`backend/app/config.py`), every processed turn is saved to `backend/debug_audio/` as a `.wav` with a `.json` label (speaker, language, text). Turns rejected for language are saved as `*_rejected_lang_*.wav`. The folder is cleared on every backend start.
- **Frontend VAD:** set `DEBUG_VAD = true` in `useDialogSession.js` for detailed VAD logs in the browser console (`[VAD]` lines are always logged).
- **Backend STT:** set `DEBUG_STT = True` in `stt.py` for per-segment keep/drop details.

Turn `SAVE_DEBUG_AUDIO`, `DEBUG_VAD` and `DEBUG_STT` off for production use.
