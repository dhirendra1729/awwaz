"""
Audio Ingress: RAM-only Circular Ring Buffer and 2-Tier Progressive Sliding Window.

Invariant:
- Ephemeral-by-default: Audio lives strictly in volatile memory.
- Zero raw audio is ever written to disk or persistent storage.
- Max ring buffer size is capped (default 4.5 seconds @ 16 kHz = 72,000 samples).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class WindowTier(str, Enum):
    FAST_TRIAGE = "fast_triage"       # 1.0s window for SincNet fast edge inference (~25ms)
    CONFIRM_DECISION = "confirm_tier"  # 4.0s window for full AWWAZ-AcousticNet (~60ms)


@dataclass(frozen=True)
class WindowFrame:
    tier: WindowTier
    audio: np.ndarray          # 1D float32 array normalized to [-1.0, 1.0]
    sample_rate: int
    duration_sec: float
    timestamp_ms: float
    is_speech: bool


class AudioRingBuffer:
    """
    Fixed-capacity circular buffer operating in volatile RAM.
    Guarantees that audio older than max_duration_sec is automatically overwritten.
    """
    def __init__(self, sample_rate: int = 16000, max_duration_sec: float = 4.5):
        self.sample_rate = sample_rate
        self.capacity = int(sample_rate * max_duration_sec)
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_pos = 0
        self.total_samples_written = 0

    def push(self, chunk: np.ndarray) -> None:
        """
        Append PCM audio samples into the circular buffer.
        Accepts float32 in [-1, 1] or int16 PCM.
        """
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)

        if np.issubdtype(chunk.dtype, np.integer):
            chunk = chunk.astype(np.float32) / 32768.0
        else:
            chunk = chunk.astype(np.float32)

        n = len(chunk)
        if n == 0:
            return

        if n >= self.capacity:
            # Chunk is larger than full buffer: keep only newest samples
            self.buffer[:] = chunk[-self.capacity:]
            self.write_pos = 0
            self.total_samples_written += n
            return

        end_pos = self.write_pos + n
        if end_pos <= self.capacity:
            self.buffer[self.write_pos:end_pos] = chunk
            self.write_pos = end_pos % self.capacity
        else:
            first_part = self.capacity - self.write_pos
            second_part = n - first_part
            self.buffer[self.write_pos:] = chunk[:first_part]
            self.buffer[:second_part] = chunk[first_part:]
            self.write_pos = second_part

        self.total_samples_written += n

    def get_latest(self, num_samples: int) -> np.ndarray:
        """Extract the most recent `num_samples` from the ring buffer."""
        available = min(self.total_samples_written, self.capacity)
        if num_samples > available:
            raise ValueError(f"Requested {num_samples} samples, but only {available} available in buffer.")

        out = np.zeros(num_samples, dtype=np.float32)
        start_idx = (self.write_pos - num_samples) % self.capacity
        if start_idx + num_samples <= self.capacity:
            out[:] = self.buffer[start_idx:start_idx + num_samples]
        else:
            first_len = self.capacity - start_idx
            second_len = num_samples - first_len
            out[:first_len] = self.buffer[start_idx:]
            out[first_len:] = self.buffer[:second_len]
        return out

    def available_samples(self) -> int:
        return min(self.total_samples_written, self.capacity)

    def clear(self) -> None:
        """Explicit RAM purge for privacy compliance."""
        self.buffer.fill(0)
        self.write_pos = 0
        self.total_samples_written = 0


class StreamingIngressPipeline:
    """
    High-level ingestion pipeline providing 2-tier sliding windows
    for real-time streaming audio inference.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        fast_window_sec: float = 1.0,
        confirm_window_sec: float = 4.0,
        max_buffer_sec: float = 4.5,
    ):
        self.sample_rate = sample_rate
        self.fast_samples = int(fast_window_sec * sample_rate)
        self.confirm_samples = int(confirm_window_sec * sample_rate)
        self.ring_buffer = AudioRingBuffer(sample_rate=sample_rate, max_duration_sec=max_buffer_sec)
        self.fast_stride_samples = int(0.25 * sample_rate)
        self.confirm_stride_samples = int(1.0 * sample_rate)
        self.last_fast_eval_pos = 0
        self.last_confirm_eval_pos = 0

    def ingest_chunk(self, chunk: np.ndarray) -> None:
        self.ring_buffer.push(chunk)

    def is_fast_window_ready(self) -> bool:
        written = self.ring_buffer.total_samples_written
        if written < self.fast_samples:
            return False
        return (written - self.last_fast_eval_pos) >= self.fast_stride_samples

    def is_confirm_window_ready(self) -> bool:
        written = self.ring_buffer.total_samples_written
        if written < self.confirm_samples:
            return False
        return (written - self.last_confirm_eval_pos) >= self.confirm_stride_samples

    def get_fast_window(self) -> WindowFrame:
        samples = self.ring_buffer.get_latest(self.fast_samples)
        self.last_fast_eval_pos = self.ring_buffer.total_samples_written
        timestamp_ms = (self.ring_buffer.total_samples_written / self.sample_rate) * 1000.0
        return WindowFrame(
            tier=WindowTier.FAST_TRIAGE,
            audio=samples,
            sample_rate=self.sample_rate,
            duration_sec=self.fast_samples / self.sample_rate,
            timestamp_ms=timestamp_ms,
            is_speech=True,
        )

    def get_confirm_window(self) -> WindowFrame:
        samples = self.ring_buffer.get_latest(self.confirm_samples)
        self.last_confirm_eval_pos = self.ring_buffer.total_samples_written
        timestamp_ms = (self.ring_buffer.total_samples_written / self.sample_rate) * 1000.0
        return WindowFrame(
            tier=WindowTier.CONFIRM_DECISION,
            audio=samples,
            sample_rate=self.sample_rate,
            duration_sec=self.confirm_samples / self.sample_rate,
            timestamp_ms=timestamp_ms,
            is_speech=True,
        )

    def purge(self) -> None:
        """Purge all in-memory buffers immediately upon call termination."""
        self.ring_buffer.clear()
        self.last_fast_eval_pos = 0
        self.last_confirm_eval_pos = 0
