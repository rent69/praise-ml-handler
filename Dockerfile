FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Install ffmpeg shared libraries via conda (the base image is conda-based).
# torchcodec needs libavutil/libavcodec/libavformat .so files at runtime —
# conda's ffmpeg package provides these in the correct paths.
# We also install git (needed by pip for some deps).
RUN conda install -y -c conda-forge ffmpeg && \
    conda clean -afy && \
    apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
ENV USER=appuser HOME=/home/appuser
RUN useradd -m -s /bin/bash $USER

WORKDIR /app

# Install Python deps — pyannote-audio 4.0.4 pins torch==2.8.0 which matches our base image.
# torchcodec==0.7.0 is pinned for torch 2.8 compatibility (see compatibility table).
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY main.py config.py diarization_utils.py ./

RUN chown -R $USER:$USER /app

USER $USER

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
