import logging

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional, Literal

logger = logging.getLogger(__name__)


class ModelSettings(BaseSettings):
    asr_model: str
    assistant_model: Optional[str]
    diarization_model: Optional[str]
    hf_token: Optional[str]


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


model_settings = ModelSettings()

logger.info(f"asr model: {model_settings.asr_model}")
logger.info(f"assist model: {model_settings.assistant_model}")
logger.info(f"diar model: {model_settings.diarization_model}")