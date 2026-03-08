FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Install system deps: ffmpeg CLI (used by transformers ffmpeg_read fallback
# and pyannote's torchaudio), git (pip needs it for some deps).
# We do NOT use torchcodec — it requires ffmpeg shared libs that conflict
# with this base image. transformers auto-falls back to subprocess ffmpeg.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
ENV USER=appuser HOME=/home/appuser
RUN useradd -m -s /bin/bash $USER

WORKDIR /app

# Install Python deps — pyannote-audio 4.0.4 pins torch==2.8.0 which matches our base image.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt && \
    pip uninstall -y torchcodec 2>/dev/null || true
# ^ torchcodec gets pulled as a transitive dep but fails at runtime
#   because it needs ffmpeg shared libs. Removing it forces transformers
#   to fall back to ffmpeg_read (subprocess), which works perfectly.

# Copy application code
COPY main.py config.py diarization_utils.py ./

RUN chown -R $USER:$USER /app

USER $USER

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
