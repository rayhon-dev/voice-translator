# Voice Translator

Live speech → English transcription (Whisper / faster-whisper), translation
(NLLB-200), speaker identification (Resemblyzer), and text-to-speech, with a
FastAPI backend and a React + Vite frontend.

- `backend/` — FastAPI + uvicorn, GPU models (CUDA 12.1)
- `frontend/` — React 19 + Vite 8, served as static files by nginx

---

## Running with Docker (GPU)

### Prerequisites

- **Docker Desktop** with the **WSL 2 based engine** enabled
  (*Settings → General → Use the WSL 2 based engine*).
- An **NVIDIA GPU + Windows driver** with WSL2 CUDA support (any reasonably
  recent GeForce Game Ready / Studio driver). You do **not** install a driver
  inside WSL — the Windows driver is used through `/dev/dxg`.
- **NVIDIA Container Toolkit** — bundled with Docker Desktop's WSL2 GPU
  integration; nothing extra to install in most setups.
- ~10 GB free disk for images, plus ~3 GB for the model cache volume.

Verify GPU access from a container before starting:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

You should see your RTX 4050 listed. If this fails, fix it before continuing —
the backend needs the GPU.

### Build and run everything

From the repo root:

```bash
docker compose up --build
```

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000  (docs at `/docs`)

Stop with `Ctrl+C`; `docker compose down` removes the containers. The model
cache survives (it's a named volume). To wipe it too: `docker compose down -v`.

### First start is slow — later starts are fast

On the **first** run the backend downloads the Whisper `small` and
NLLB-200-distilled-600M weights from HuggingFace (~2.5 GB) and then runs a
warm-up inference for each model. Expect **several minutes** before
`[warmup] All models ready ...` appears in the logs and the API answers.

Downloads are written to the `hf-cache` Docker volume, so every subsequent
`docker compose up` (even after `--build`) skips the download and is ready in
a few seconds.

### Notes / limitations

- **The frontend talks to `http://localhost:8000` from the browser.** This
  works when the browser runs on the same machine as Docker (the normal
  Windows + WSL2 dev setup). To serve it to other machines, set the API base
  URL in `frontend/src/services/api.js` and the WS URL in
  `frontend/src/hooks/useLiveTranscription.js`, or put a reverse proxy in
  front of both, and rebuild the frontend image.
- **CORS:** the backend allows the origin `http://localhost:5173`
  (`backend/app/main.py`). If you change the frontend's published port, update
  that allow-list and rebuild.
- **6 GB VRAM is tight** with Whisper + NLLB loaded together. If you hit CUDA
  OOM, lower `WHISPER_COMPUTE_TYPE` to `int8_float16` in
  `backend/app/config.py` and rebuild.
- `webrtcvad` is compiled from source inside the backend image (hence
  `build-essential`); `ffmpeg` is required for faster-whisper's audio
  decoding.
- Single uvicorn worker on purpose — one GPU, models loaded once at startup.
