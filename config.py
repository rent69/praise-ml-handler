import logging

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional, Literal, List

logger = logging.getLogger(__name__)


class ModelSettings(BaseSettings):
    asr_model: str
    assistant_model: Optional[str] = None
    diarization_model: Optional[str] = None
    hf_token: Optional[str] = None


class KnownSpeaker(BaseModel):
    """A known speaker profile for matching."""
    slug: str
    name: str
    centroid_b64: str  # base64-encoded float32 embedding
    # Optional additional sample embeddings for best-of-N matching
    samples: Optional[List[dict]] = None


class InferenceConfig(BaseModel):
    task: Literal["transcribe", "translate"] = "transcribe"
    batch_size: int = 24
    assisted: bool = False
    chunk_length_s: int = 30
    sampling_rate: int = 16000
    language: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    # praise.global extensions
    return_embeddings: bool = False
    known_speakers: Optional[List[dict]] = None  # List of KnownSpeaker dicts


model_settings = ModelSettings()

logger.info(f"asr model: {model_settings.asr_model}")
logger.info(f"assist model: {model_settings.assistant_model}")
logger.info(f"diar model: {model_settings.diarization_model}")
