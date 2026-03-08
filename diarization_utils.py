import torch
import numpy as np
import base64
from torchaudio import functional as F
from transformers.pipelines.audio_utils import ffmpeg_read
from starlette.exceptions import HTTPException
import sys

import logging
logger = logging.getLogger(__name__)


def preprocess_inputs(inputs, sampling_rate):
    inputs = ffmpeg_read(inputs, sampling_rate)

    if sampling_rate != 16000:
        inputs = F.resample(
            torch.from_numpy(inputs), sampling_rate, 16000
        ).numpy()

    if len(inputs.shape) != 1:
        logger.error(f"Diarization pipeline expects single channel audio, received {inputs.shape}")
        raise HTTPException(
            status_code=400,
            detail=f"Diarization pipeline expects single channel audio, received {inputs.shape}"
        )

    # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs


def _extract_annotation(diarization_output):
    """
    Extract a pyannote Annotation object from the pipeline output.
    
    pyannote 3.1 returns an Annotation directly (has .itertracks()).
    pyannote 4.0 / community-1 returns a DiarizationOutput object with
    .speaker_diarization and .exclusive_speaker_diarization attributes.
    We prefer exclusive_speaker_diarization when available (better for
    reconciling with ASR timestamps).
    """
    # community-1 / pyannote 4.0 output object
    if hasattr(diarization_output, 'speaker_diarization'):
        # Prefer exclusive_speaker_diarization when available (better for ASR reconciliation)
        excl = getattr(diarization_output, 'exclusive_speaker_diarization', None)
        if excl is not None:
            logger.info("Using exclusive_speaker_diarization from community-1 output")
            return excl, diarization_output
        annotation = diarization_output.speaker_diarization
        logger.info("Using speaker_diarization from community-1 output")
        return annotation, diarization_output
    
    # pyannote 3.1 returns Annotation directly (has .itertracks)
    if hasattr(diarization_output, 'itertracks'):
        logger.info("Using legacy Annotation output (pyannote 3.1 style)")
        return diarization_output, diarization_output
    
    raise ValueError(f"Unexpected diarization output type: {type(diarization_output)}. "
                     f"Attributes: {dir(diarization_output)}")


def diarize_audio(diarizer_inputs, diarization_pipeline, parameters):
    raw_output = diarization_pipeline(
        {"waveform": diarizer_inputs, "sample_rate": parameters.sampling_rate},
        num_speakers=parameters.num_speakers,
        min_speakers=parameters.min_speakers,
        max_speakers=parameters.max_speakers,
    )

    annotation, full_output = _extract_annotation(raw_output)

    segments = []
    for segment, track, label in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "segment": {"start": segment.start, "end": segment.end},
                "track": track,
                "label": label,
            }
        )

    if len(segments) == 0:
        logger.warning("Diarization returned 0 segments")
        return [], full_output

    # Combine consecutive segments from the same speaker
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            new_segments.append(
                {
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = segments[i]

    new_segments.append(
        {
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": cur_segment["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        }
    )

    return new_segments, full_output


def extract_speaker_embeddings(diarization_pipeline, diarizer_inputs, diarization_result, sampling_rate=16000):
    """
    Extract per-speaker embeddings from pyannote's diarization output.
    
    pyannote 4.0 / community-1 DiarizeOutput includes speaker_embeddings
    as a numpy array of shape (num_speakers, embedding_dim), ordered by
    speaker_diarization.labels(). We use these directly — no need to
    probe internal pipeline models.
    
    Falls back to internal model probing for pyannote 3.1 which returns
    a raw Annotation without pre-computed embeddings.
    """
    try:
        annotation, full_output = _extract_annotation(diarization_result)
        
        # Debug: log what attributes the output has
        output_attrs = [a for a in dir(full_output) if not a.startswith('_')]
        logger.info(f"DiarizeOutput attributes: {output_attrs}")
        logger.info(f"DiarizeOutput type: {type(full_output)}")
        raw_emb_attr = getattr(full_output, 'speaker_embeddings', 'MISSING')
        logger.info(f"speaker_embeddings attr: type={type(raw_emb_attr)}, value={raw_emb_attr if not isinstance(raw_emb_attr, np.ndarray) else f'ndarray shape={raw_emb_attr.shape}'}")
        
        # ── Strategy 1: Use DiarizeOutput.speaker_embeddings (pyannote 4.0+) ──
        raw_embeddings = getattr(full_output, 'speaker_embeddings', None)
        if raw_embeddings is not None and isinstance(raw_embeddings, np.ndarray) and raw_embeddings.size > 0:
            # Get the speaker labels in the same order as the embeddings array
            # DiarizeOutput.speaker_embeddings is ordered by speaker_diarization.labels()
            sd_annotation = getattr(full_output, 'speaker_diarization', annotation)
            labels = sd_annotation.labels()
            
            logger.info(f"Using DiarizeOutput.speaker_embeddings: shape={raw_embeddings.shape}, labels={labels}")
            
            # Compute per-speaker durations from annotation
            speaker_durations = {}
            speaker_seg_counts = {}
            for segment, _, label in annotation.itertracks(yield_label=True):
                speaker_durations[label] = speaker_durations.get(label, 0.0) + segment.duration
                speaker_seg_counts[label] = speaker_seg_counts.get(label, 0) + 1
            
            speaker_embeddings = {}
            for i, label in enumerate(labels):
                if i >= raw_embeddings.shape[0]:
                    break
                emb = raw_embeddings[i].astype(np.float32)
                # Normalize
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                centroid_b64 = base64.b64encode(emb.tobytes()).decode("utf-8")
                
                speaker_embeddings[label] = {
                    "embedding_b64": centroid_b64,
                    "embedding_dim": int(emb.shape[0]),
                    "total_seconds": round(speaker_durations.get(label, 0.0), 2),
                    "num_segments": speaker_seg_counts.get(label, 0),
                }
                logger.info(f"Speaker {label}: dim={emb.shape[0]}, {speaker_durations.get(label, 0):.1f}s, {speaker_seg_counts.get(label, 0)} segs")
            
            return speaker_embeddings
        
        # ── Strategy 2: Probe internal embedding model (pyannote 3.1 fallback) ──
        logger.info("DiarizeOutput.speaker_embeddings not available, probing internal model...")
        embedding_model = None
        for attr in ('_embedding', 'embedding'):
            candidate = getattr(diarization_pipeline, attr, None)
            if candidate is not None and hasattr(candidate, 'parameters'):
                embedding_model = candidate
                logger.info(f"Found embedding model via pipeline.{attr}")
                break
        
        if embedding_model is None:
            logger.error(f"Cannot find embedding model. Pipeline attrs: {[a for a in dir(diarization_pipeline) if not a.startswith('__')]}")
            return {}
        
        device = next(embedding_model.parameters()).device
        
        speaker_labels = set()
        for segment, _, label in annotation.itertracks(yield_label=True):
            speaker_labels.add(label)
        
        speaker_embeddings = {}
        waveform = diarizer_inputs  # shape: (1, seq_len)
        
        for speaker in speaker_labels:
            speaker_segments = []
            total_seconds = 0.0
            for segment, _, label in annotation.itertracks(yield_label=True):
                if label == speaker:
                    speaker_segments.append(segment)
                    total_seconds += segment.duration
            
            if total_seconds < 0.5:
                continue
            
            segment_embeddings = []
            for seg in speaker_segments:
                start_sample = int(seg.start * sampling_rate)
                end_sample = int(seg.end * sampling_rate)
                if end_sample > waveform.shape[1]:
                    end_sample = waveform.shape[1]
                if end_sample - start_sample < sampling_rate * 0.3:
                    continue
                chunk = waveform[:, start_sample:end_sample].to(device)
                with torch.no_grad():
                    emb = embedding_model(chunk)
                if emb.dim() > 1:
                    emb = emb.squeeze()
                emb = emb / (torch.norm(emb) + 1e-8)
                segment_embeddings.append(emb.cpu().numpy())
            
            if len(segment_embeddings) == 0:
                continue
            
            centroid = np.mean(segment_embeddings, axis=0).astype(np.float32)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroid_b64 = base64.b64encode(centroid.tobytes()).decode("utf-8")
            
            speaker_embeddings[speaker] = {
                "embedding_b64": centroid_b64,
                "embedding_dim": int(centroid.shape[0]),
                "total_seconds": round(total_seconds, 2),
                "num_segments": len(segment_embeddings),
            }
            logger.info(f"Speaker {speaker}: {total_seconds:.1f}s, {len(segment_embeddings)} segs, dim={centroid.shape[0]}")
        
        return speaker_embeddings
        
    except Exception as e:
        logger.error(f"Error extracting speaker embeddings: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def match_speakers(speaker_embeddings, known_speakers):
    """
    Match diarized speakers against known speaker profiles using cosine similarity.
    
    known_speakers: list of dicts with {slug, name, centroid_b64, samples?}
    speaker_embeddings: dict from extract_speaker_embeddings
    
    Returns dict mapping SPEAKER_XX -> {matched_slug, matched_name, confidence, score}
    """
    if not known_speakers or not speaker_embeddings:
        return {}
    
    # Decode known speaker centroids
    known_profiles = []
    for ks in known_speakers:
        try:
            centroid_bytes = base64.b64decode(ks["centroid_b64"])
            centroid = np.frombuffer(centroid_bytes, dtype=np.float32)
            
            # Also decode sample embeddings if present
            samples = []
            if ks.get("samples"):
                for s in ks["samples"]:
                    if s.get("embedding_b64"):
                        s_bytes = base64.b64decode(s["embedding_b64"])
                        samples.append(np.frombuffer(s_bytes, dtype=np.float32))
            
            known_profiles.append({
                "slug": ks["slug"],
                "name": ks["name"],
                "centroid": centroid,
                "samples": samples,
            })
        except Exception as e:
            logger.warning(f"Could not decode profile for {ks.get('slug', '?')}: {e}")
            continue
    
    if not known_profiles:
        return {}
    
    matches = {}
    
    for spk_label, spk_data in speaker_embeddings.items():
        try:
            query_bytes = base64.b64decode(spk_data["embedding_b64"])
            query = np.frombuffer(query_bytes, dtype=np.float32)
        except Exception:
            continue
        
        best_score = -1.0
        best_profile = None
        
        for profile in known_profiles:
            # Cosine similarity with centroid
            centroid_score = float(np.dot(query, profile["centroid"]) / 
                                   (np.linalg.norm(query) * np.linalg.norm(profile["centroid"]) + 1e-8))
            
            # Best-of-N: also check individual samples
            best_sample_score = centroid_score
            for sample in profile["samples"]:
                s_score = float(np.dot(query, sample) / 
                               (np.linalg.norm(query) * np.linalg.norm(sample) + 1e-8))
                best_sample_score = max(best_sample_score, s_score)
            
            # Final score = max of centroid and best sample
            final_score = max(centroid_score, best_sample_score)
            
            if final_score > best_score:
                best_score = final_score
                best_profile = profile
        
        if best_profile is None:
            continue
        
        # Confidence tiers (calibrated for pyannote wespeaker embeddings)
        if best_score >= 0.55:
            confidence = "HIGH"
        elif best_score >= 0.35:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        matches[spk_label] = {
            "matched_slug": best_profile["slug"],
            "matched_name": best_profile["name"],
            "confidence": confidence,
            "score": round(best_score, 4),
        }
        
        logger.info(f"Speaker {spk_label} -> {best_profile['name']} ({confidence}, {best_score:.4f})")
    
    return matches


def post_process_segments_and_transcripts(new_segments, transcript, group_by_speaker) -> list:
    end_timestamps = np.array(
        [chunk["timestamp"][-1] if chunk["timestamp"][-1] is not None else sys.float_info.max for chunk in transcript])
    segmented_preds = []

    for segment in new_segments:
        end_time = segment["segment"]["end"]
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append(
                {
                    "speaker": segment["speaker"],
                    "text": "".join(
                        [chunk["text"] for chunk in transcript[: upto_idx + 1]]
                    ),
                    "timestamp": (
                        transcript[0]["timestamp"][0],
                        transcript[upto_idx]["timestamp"][1],
                    ),
                }
            )
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

        transcript = transcript[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]

        if len(end_timestamps) == 0:
            break

    return segmented_preds


def diarize(diarization_pipeline, file, parameters, asr_outputs):
    """Original diarize function — backward compatible."""
    _, diarizer_inputs = preprocess_inputs(file, parameters.sampling_rate)

    segments, _ = diarize_audio(
        diarizer_inputs,
        diarization_pipeline,
        parameters
    )

    return post_process_segments_and_transcripts(
        segments, asr_outputs["chunks"], group_by_speaker=False
    )


def diarize_with_embeddings(diarization_pipeline, file, parameters, asr_outputs):
    """
    Extended diarize that also extracts per-speaker embeddings and optionally
    matches against known speaker profiles.
    
    Returns: (transcript, speaker_embeddings, speaker_matches)
    """
    _, diarizer_inputs = preprocess_inputs(file, parameters.sampling_rate)

    segments, diarization_result = diarize_audio(
        diarizer_inputs,
        diarization_pipeline,
        parameters
    )

    transcript = post_process_segments_and_transcripts(
        segments, asr_outputs["chunks"], group_by_speaker=False
    )
    
    # Extract embeddings
    speaker_embeddings = {}
    if parameters.return_embeddings:
        speaker_embeddings = extract_speaker_embeddings(
            diarization_pipeline, diarizer_inputs, diarization_result,
            sampling_rate=parameters.sampling_rate
        )
    
    # Match against known speakers
    speaker_matches = {}
    if parameters.known_speakers and speaker_embeddings:
        speaker_matches = match_speakers(speaker_embeddings, parameters.known_speakers)
    
    return transcript, speaker_embeddings, speaker_matches
