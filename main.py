"""
praise-ml: Unified ASR + Diarization + Speaker Embedding + Matching endpoint.

FastAPI server for HF Inference Endpoints custom container deployment.
Models loaded from /repository (HF-mounted) or downloaded from Hub.
"""
import logging
import time
import base64
import os
from contextlib import asynccontextmanager
from typing import Optional, List

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

ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-large-v3")
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
    inputs: str = Field(..., description="Base64-encoded audio bytes")
    task: str = "transcribe"
    language: Optional[str] = "en"
    batch_size: int = 24
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
# Model Manager
# ------------------------------------------------------
class ModelManager:
    def __init__(self):
        self.asr_pipeline = None
        self.assistant_model = None
        self.diarization_pipeline = None

    async def load(self):
        start = time.perf_counter()
        logger.info(f"Loading models on {DEVICE} ({TORCH_DTYPE})...")

        # ASR pipeline
        logger.info(f"Loading ASR model: {ASR_MODEL}")
        self.asr_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
        )

        # Assistant model for speculative decoding (optional)
        if ASSISTANT_MODEL:
            logger.info(f"Loading assistant model: {ASSISTANT_MODEL}")
            self.assistant_model = AutoModelForCausalLM.from_pretrained(
                ASSISTANT_MODEL,
                torch_dtype=TORCH_DTYPE,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            ).to(DEVICE)

        # Diarization pipeline
        if DIARIZATION_MODEL:
            logger.info(f"Loading diarization model: {DIARIZATION_MODEL}")
            self.diarization_pipeline = Pipeline.from_pretrained(
                DIARIZATION_MODEL,
                token=HF_TOKEN,
            )
            self.diarization_pipeline.to(DEVICE)

        duration = time.perf_counter() - start
        logger.info(f"All models loaded in {duration:.1f}s")

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
    """Main inference endpoint — ASR + diarization + embeddings + matching."""
    start = time.perf_counter()

    if model_manager.asr_pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Decode audio
    try:
        audio_bytes = base64.b64decode(request.inputs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {e}")

    # ASR
    generate_kwargs = {
        "task": request.task,
        "language": request.language,
    }
    if request.assisted and model_manager.assistant_model:
        generate_kwargs["assistant_model"] = model_manager.assistant_model

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
        raise HTTPException(status_code=500, detail=f"ASR inference error: {e}")

    # Diarization + embeddings + matching
    speaker_embeddings = {}
    speaker_matches = {}
    transcript = []

    if model_manager.diarization_pipeline:
        try:
            transcript, speaker_embeddings, speaker_matches = diarize_with_embeddings(
                model_manager.diarization_pipeline,
                audio_bytes,
                request,
                asr_outputs,
            )
        except Exception as e:
            logger.error(f"Diarization error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Diarization error: {e}"
            )

    duration = time.perf_counter() - start
    logger.info(f"Processed in {duration:.1f}s — {len(transcript)} segments")

    response = {
        "text": asr_outputs["text"],
        "chunks": asr_outputs["chunks"],
        "speakers": transcript,
    }
    if speaker_embeddings:
        response["speaker_embeddings"] = speaker_embeddings
    if speaker_matches:
        response["speaker_matches"] = speaker_matches

    return response
