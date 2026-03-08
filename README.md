---
tags:
- endpoints-compatible
---

# praise-ml-handler

Unified ASR + Diarization + Speaker Embedding + Speaker Matching handler for praise.global.

Forked from [sergeipetrov/asrdiarization-handler](https://huggingface.co/sergeipetrov/asrdiarization-handler).

## Extensions over upstream

- **Speaker embedding extraction** — extracts per-speaker embeddings from pyannote's internal wespeaker model as a byproduct of diarization
- **Speaker matching** — matches diarized speakers against known voice profiles using cosine similarity
- **Confidence tiers** — HIGH (≥0.55), MEDIUM (≥0.35), LOW (<0.35) calibrated for pyannote embeddings

## API

Standard Inference Endpoint `POST /` with `inputs` (base64 audio) and `parameters`:

```json
{
  "inputs": "<base64_audio>",
  "parameters": {
    "task": "transcribe",
    "language": "en",
    "batch_size": 24,
    "chunk_length_s": 30,
    "min_speakers": 2,
    "max_speakers": 12,
    "return_embeddings": true,
    "known_speakers": [
      {"slug": "bob-ryan", "name": "Bob Ryan", "centroid_b64": "..."}
    ]
  }
}
```

## Response

```json
{
  "text": "full transcript...",
  "chunks": [...],
  "speakers": [...],
  "speaker_embeddings": {
    "SPEAKER_00": {"embedding_b64": "...", "embedding_dim": 512, "total_seconds": 45.2, "num_segments": 12}
  },
  "speaker_matches": {
    "SPEAKER_00": {"matched_slug": "bob-ryan", "matched_name": "Bob Ryan", "confidence": "HIGH", "score": 0.72}
  }
}
```

## Deployment

Create via HF Inference Endpoints API with env vars:
- `ASR_MODEL=openai/whisper-large-v3`
- `DIARIZATION_MODEL=pyannote/speaker-diarization-3.1`
- `HF_TOKEN=<your_token>`
- `ASSISTANT_MODEL=distil-whisper/distil-large-v3` (optional, for speculative decoding)
