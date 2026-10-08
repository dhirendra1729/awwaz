"""
WebSocket Streaming Audio Ingress Endpoint.
Streams real-time 16kHz PCM audio chunks and yields 2-tier sliding window deepfake inferences.
"""

import json
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..audio.ingress import StreamingIngressPipeline
from ..audio.vad import SileroVADEngine
from ..models.acoustic_net import AWWAZAcousticNet

router = APIRouter()


@router.websocket("/ws/audio-stream")
async def websocket_audio_stream_endpoint(websocket: WebSocket):
    """
    Ingests binary 16kHz 16-bit PCM chunks.
    Yields 2-Tier real-time triage and confirmation events over WebSocket.
    """
    await websocket.accept()

    ingress = StreamingIngressPipeline(
        sample_rate=16000,
        fast_window_sec=1.0,
        confirm_window_sec=4.0,
    )
    vad = SileroVADEngine(sample_rate=16000)
    model = AWWAZAcousticNet(sample_rate=16000)

    try:
        while True:
            # Receive raw binary PCM audio bytes
            data = await websocket.receive_bytes()
            if not data:
                continue

            # Convert bytes to int16 / float32 numpy array
            pcm_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # VAD check
            vad_res = vad.process_frame(pcm_chunk)
            if not vad_res.is_speech and vad_res.energy_rms < 0.005:
                # Discard pure silence frames
                continue

            ingress.ingest_chunk(pcm_chunk)

            # Tier-1 Fast Triage (1.0s window evaluated in <25ms)
            if ingress.is_fast_window_ready():
                fast_frame = ingress.get_fast_window()
                fast_score = model.sincnet_branch.score_fast_triage(fast_frame.audio)
                if fast_score >= 0.70:
                    await websocket.send_text(
                        json.dumps({
                            "event_type": "TIER_1_EARLY_WARNING",
                            "timestamp_ms": round(fast_frame.timestamp_ms, 2),
                            "triage_score": round(fast_score, 4),
                            "alert": "Acoustic anomaly detected in preliminary 1.0s window",
                        })
                    )

            # Tier-2 Calibrated Confirmation (4.0s window evaluated in ~60ms)
            if ingress.is_confirm_window_ready():
                confirm_frame = ingress.get_confirm_window()
                res = model.infer(confirm_frame.audio, duration_sec=confirm_frame.duration_sec)
                await websocket.send_text(
                    json.dumps({
                        "event_type": "TIER_2_CONFIRMED_EVALUATION",
                        "timestamp_ms": round(confirm_frame.timestamp_ms, 2),
                        "calibrated_p_spoof": res.calibrated_p_spoof,
                        "uncertainty_band": res.uncertainty_band,
                        "is_spoof_suspected": res.is_spoof_suspected,
                        "spliced_segments": [
                            {"start": s.start_sec, "end": s.end_sec, "conf": s.confidence}
                            for s in res.spliced_segments
                        ],
                        "inference_latency_ms": res.inference_latency_ms,
                    })
                )

    except WebSocketDisconnect:
        pass
    finally:
        # Privacy Invariant: Purge RAM audio buffers immediately on disconnect
        ingress.purge()
