# CUDA 12.1 + cuDNN 8 runtime on Ubuntu 22.04 (system Python is 3.10).
# cuDNN 8 is required by CTranslate2 4.x, the inference backend of
# faster-whisper 1.0.3. PyTorch's cu121 wheels bundle their own CUDA libs.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

# System dependencies:
#   python3 / pip / python3-dev  - runtime + headers for building C extensions
#   build-essential              - compiles webrtcvad (pulled in by resemblyzer)
#   ffmpeg                       - faster-whisper audio decode + librosa fallback
#   libsndfile1                  - soundfile / librosa
#   curl                         - container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        ffmpeg \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch from the CUDA 12.1 index FIRST, and pin numpy, so that when
# transformers / resemblyzer are resolved below they see a cu121 torch build
# already satisfied instead of pulling the default PyPI wheel.
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 \
    && python3 -m pip install numpy==1.26.4

# Dependency layer - cached until requirements.txt changes.
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Application code.
COPY app ./app

EXPOSE 8000

# Single worker: one 6 GB GPU. All three models are loaded + warmed up once at
# startup by the lifespan handler in app/main.py.
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
