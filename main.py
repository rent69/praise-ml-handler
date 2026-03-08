"""
praise-ml: Unified ASR + Diarization + Speaker Embedding + Matching endpoint.

FastAPI server for HF Inference Endpoints custom container deployment.
Models loaded from /repository (HF-mounted) or downloaded from Hub.
"""
import logging
import time
import base64
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, List

import httpx
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline, AutoModelForCausalLM

from diarization_utils import (
    diarize_with_embeddings,
    preprocess_inputs,
    post_process_segments_and_transcripts,
    diarize_audio,
)

# ------------------------------------------------------
# Config + Logging
# ------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32 if DEVICE.type == "cpu" else torch.float16

ASR_MODEL = os.environ.get("ASR_MODEL", "distil-whisper/distil-large-v3")
ASSISTANT_MODEL = os.environ.get("ASSISTANT_MODEL", None)
DIARIZATION_MODEL = os.environ.get(
    "DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
)
HF_TOKEN = os.environ.get("HF_TOKEN", None)


# ------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------
class KnownSpeaker(BaseModel):
    slug: str
    name: str
    centroid_b64: str
    samples: Optional[List[dict]] = None


class ProcessRequest(BaseModel):
    # Audio input — provide EITHER inputs (base64) OR audio_url (public URL)
    inputs: Optional[str] = Field(None, description="Base64-encoded audio bytes")
    audio_url: Optional[str] = Field(None, description="Public URL to audio file (downloaded by endpoint)")
    task: str = "transcribe"
    language: Optional[str] = "en"
    batch_size: int = 12
    chunk_length_s: int = 30
    assisted: bool = False
    sampling_rate: int = 16000
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    return_embeddings: bool = False
    known_speakers: Optional[List[dict]] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    asr_model: str
    diarization_model: Optional[str]


# ------------------------------------------------------
# Audio resolution — base64 or URL download
# ------------------------------------------------------
MAX_AUDIO_SIZE = 500 * 1024 * 1024  # 500MB limit for URL downloads
DOWNLOAD_TIMEOUT = 300  # 5 min timeout for large files


def resolve_audio_bytes(request: ProcessRequest) -> bytes:
    """Resolve audio bytes from either base64 input or URL download."""
    if request.inputs and request.audio_url:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'inputs' (base64) or 'audio_url', not both"
        )

    if request.inputs:
        try:
            return base64.b64decode(request.inputs)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {e}")

    if request.audio_url:
        logger.info(f"Downloading audio from URL: {request.audio_url[:100]}...")
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
                resp = client.get(request.audio_url)
                resp.raise_for_status()
                audio_bytes = resp.content
                if len(audio_bytes) > MAX_AUDIO_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Audio file too large: {len(audio_bytes)} bytes (max {MAX_AUDIO_SIZE})"
                    )
                duration = time.perf_counter() - start
                logger.info(f"Downloaded {len(audio_bytes)} bytes in {duration:.1f}s")
                return audio_bytes
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download audio: HTTP {e.response.status_code}"
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=408,
                detail=f"Audio download timed out after {DOWNLOAD_TIMEOUT}s"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Audio download error: {e}"
            )

    raise HTTPException(
        status_code=400,
        detail="Must provide either 'inputs' (base64) or 'audio_url'"
    )


# ------------------------------------------------------
# Model Manager
# ------------------------------------------------------
class ModelManager:
    """Manages ASR + diarization models with GPU memory swapping.
    
    T4 has 15GB VRAM. Whisper large-v3 (~3GB) + pyannote community-1 (~11GB)
    together exceed available memory for inference. We load both but swap
    them on/off GPU sequentially during processing:
    
    1. ASR on GPU, diarization on CPU  → transcribe
    2. ASR to CPU, diarization to GPU  → diarize + embeddings
    """
    def __init__(self):
        self.asr_pipeline = None
        self.assistant_model = None
        self.diarization_pipeline = None

    async def load(self):
        start = time.perf_counter()
        logger.info(f"Loading models (sequential GPU strategy for T4)...")

        # Load ASR to GPU first
        logger.info(f"Loading ASR model: {ASR_MODEL}")
        self.asr_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
        )

        # Load diarization to CPU initially (will move to GPU when needed)
        if DIARIZATION_MODEL:
            logger.info(f"Loading diarization model to CPU: {DIARIZATION_MODEL}")
            self.diarization_pipeline = Pipeline.from_pretrained(
                DIARIZATION_MODEL,
                token=HF_TOKEN,
            )
            # Keep on CPU — will move to GPU after ASR completes
            logger.info("Diarization pipeline loaded on CPU")

        duration = time.perf_counter() - start
        logger.info(f"All models loaded in {duration:.1f}s")
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory after load: {mem:.1f}GB")

    def swap_to_asr(self):
        """Ensure ASR is on GPU, diarization on CPU."""
        if self.diarization_pipeline is not None:
            self.diarization_pipeline.to(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ASR pipeline stays on GPU (loaded there initially)
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"[swap_to_asr] GPU mem: {mem:.1f}GB")

    def swap_to_diarization(self):
        """Move ASR off GPU, diarization onto GPU."""
        # Move ASR model to CPU
        if self.asr_pipeline is not None:
            self.asr_pipeline.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Move diarization to GPU
        if self.diarization_pipeline is not None:
            self.diarization_pipeline.to(DEVICE)
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"[swap_to_diarization] GPU mem: {mem:.1f}GB")

    def restore_asr_to_gpu(self):
        """After processing, restore ASR to GPU for next request."""
        if self.diarization_pipeline is not None:
            self.diarization_pipeline.to(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.asr_pipeline is not None:
            self.asr_pipeline.model.to(DEVICE)
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"[restore_asr] GPU mem: {mem:.1f}GB")

    async def unload(self):
        del self.asr_pipeline
        del self.assistant_model
        del self.diarization_pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


model_manager = ModelManager()


# ------------------------------------------------------
# FastAPI app with lifespan
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.load()
    try:
        yield
    finally:
        await model_manager.unload()


app = FastAPI(title="praise-ml", lifespan=lifespan)


# ------------------------------------------------------
# Routes
# ------------------------------------------------------
@app.get("/health")
def health() -> HealthResponse:
    if model_manager.asr_pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
        asr_model=ASR_MODEL,
        diarization_model=DIARIZATION_MODEL,
    )


@app.post("/")
def process(request: ProcessRequest):
    """Main inference endpoint — ASR + diarization + embeddings + matching.
    
    Accepts audio as either base64 (inputs field) or a public URL (audio_url field).
    The URL path avoids base64 encoding overhead for large files and lets the 
    GPU endpoint pull audio directly from R2/CDN.
    """
    start = time.perf_counter()

    if model_manager.asr_pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Resolve audio bytes from base64 or URL
    audio_bytes = resolve_audio_bytes(request)
    resolve_time = time.perf_counter() - start
    logger.info(f"Audio resolved: {len(audio_bytes)} bytes in {resolve_time:.1f}s")

    # ── Phase 1: ASR (Whisper on GPU, diarization on CPU) ──────────────
    model_manager.swap_to_asr()

    generate_kwargs = {
        "task": request.task,
        "language": request.language,
    }

    try:
        asr_outputs = model_manager.asr_pipeline(
            audio_bytes,
            chunk_length_s=request.chunk_length_s,
            batch_size=request.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )
    except Exception as e:
        logger.error(f"ASR error: {e}")
        model_manager.restore_asr_to_gpu()
        raise HTTPException(status_code=500, detail=f"ASR inference error: {e}")

    asr_time = time.perf_counter() - start - resolve_time
    logger.info(f"ASR complete in {asr_time:.1f}s")

    # ── Phase 2: Diarization (pyannote on GPU, Whisper on CPU) ─────────
    speaker_embeddings = {}
    speaker_matches = {}
    transcript = []

    if model_manager.diarization_pipeline:
        model_manager.swap_to_diarization()
        try:
            transcript, speaker_embeddings, speaker_matches = diarize_with_embeddings(
                model_manager.diarization_pipeline,
                audio_bytes,
                request,
                asr_outputs,
            )
        except Exception as e:
            logger.error(f"Diarization error: {e}")
            model_manager.restore_asr_to_gpu()
            raise HTTPException(
                status_code=500, detail=f"Diarization error: {e}"
            )

    # ── Restore ASR to GPU for next request ────────────────────────────
    model_manager.restore_asr_to_gpu()

    total_time = time.perf_counter() - start
    logger.info(f"Total processing: {total_time:.1f}s — {len(transcript)} segments")

    response = {
        "text": asr_outputs["text"],
        "chunks": asr_outputs["chunks"],
        "speakers": transcript,
        "processing_time_s": round(total_time, 2),
    }
    if speaker_embeddings:
        response["speaker_embeddings"] = speaker_embeddings
    if speaker_matches:
        response["speaker_matches"] = speaker_matches

    # Force GPU memory cleanup after every request to prevent OOM on consecutive chunks
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response



@app.post("/cleanup")
def cleanup():
    """Force GPU memory cleanup between chunks.
    Called by analyze-service between chunked processing passes."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / 1e9
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / 1e9
        logger.info(f"[cleanup] GPU memory: {before:.2f}GB → {after:.2f}GB")
        return {"status": "ok", "gpu_mem_before_gb": round(before, 2), "gpu_mem_after_gb": round(after, 2)}
    return {"status": "ok", "gpu": "not available"}
