"""
Privacy Invariant Test: Zero Raw Audio on Disk & SafeEar Decoupling.
Reference: Section 8 & Section 10 of AWWAZ Master Blueprint.
"""

import os
import tempfile
import numpy as np
import pytest

try:
    from awwaz.audio.ingress import AudioRingBuffer, StreamingIngressPipeline
    from awwaz.audio.privacy import SafeEarPrivacyDecoupler
except ImportError:
    from src.awwaz.audio.ingress import AudioRingBuffer, StreamingIngressPipeline
    from src.awwaz.audio.privacy import SafeEarPrivacyDecoupler


def test_ephemeral_ram_buffer_never_writes_files():
    """Verify that buffer ingestion and processing does not create disk artifacts."""
    initial_files = set(os.listdir("."))

    pipeline = StreamingIngressPipeline(sample_rate=16000, fast_window_sec=1.0, confirm_window_sec=4.0)

    # Ingest 10 seconds of simulated audio
    for _ in range(500):
        dummy_chunk = np.random.uniform(-0.5, 0.5, 320).astype(np.float32)
        pipeline.ingest_chunk(dummy_chunk)
        if pipeline.is_fast_window_ready():
            _ = pipeline.get_fast_window()
        if pipeline.is_confirm_window_ready():
            _ = pipeline.get_confirm_window()

    pipeline.purge()

    current_files = set(os.listdir("."))
    # Assert no new temporary audio files were dumped to current directory
    assert initial_files == current_files


def test_circular_ring_buffer_overwrites_old_audio():
    """Ring buffer must cap in-memory capacity and overwrite older frames."""
    sample_rate = 16000
    max_duration_sec = 4.5
    capacity = int(sample_rate * max_duration_sec)  # 72,000 samples

    ring = AudioRingBuffer(sample_rate=sample_rate, max_duration_sec=max_duration_sec)

    # Push 3x capacity
    total_samples = capacity * 3
    audio = np.linspace(1.0, float(total_samples), total_samples, dtype=np.float32)
    ring.push(audio)

    # Buffer must hold only latest capacity samples
    assert ring.available_samples() == capacity
    latest = ring.get_latest(100)
    expected_end = total_samples
    assert np.isclose(latest[-1], expected_end, atol=1.0)


def test_safeear_semantic_stream_is_purged():
    """Verify SafeEar decoupler extracts Zac and purges Zsem semantic text stream."""
    decoupler = SafeEarPrivacyDecoupler(feature_dim=64, discard_semantic_stream=True)
    audio = np.random.uniform(-0.8, 0.8, 16000).astype(np.float32)

    decoupled = decoupler.decouple(audio, sample_rate=16000)

    # Acoustic residual present for spoof detection
    assert decoupled.acoustic_residual_zac is not None
    assert decoupled.acoustic_residual_zac.shape == (64,)

    # Semantic stream strictly purged
    assert decoupled.semantic_content_zsem is None
    assert decoupled.is_semantic_purged is True
    assert decoupled.retention_policy == "RAM_ONLY_EPHEMERAL_DPDP_COMPLIANT"
