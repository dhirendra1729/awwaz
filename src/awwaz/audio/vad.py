"""
Voice Activity Detection (VAD) Engine.
Wraps Silero VAD / Energy gating for speech frame detection and pre-filtering.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class VADDecision:
    is_speech: bool
    speech_probability: float
    energy_rms: float
    frame_samples: int


class SileroVADEngine:
    """
    Lightweight Voice Activity Detector with high-performance heuristic fallback.
    Prevents running deep learning inference on silence or pure background noise.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        min_energy_threshold: float = 0.005,
    ):
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.min_energy_threshold = min_energy_threshold
        self.onnx_session = None

    def process_frame(self, frame: np.ndarray) -> VADDecision:
        """
        Evaluate a 20-50ms audio chunk for speech activity.
        Returns VADDecision with speech probability and RMS energy.
        """
        if frame.size == 0:
            return VADDecision(is_speech=False, speech_probability=0.0, energy_rms=0.0, frame_samples=0)

        rms = float(np.sqrt(np.mean(np.square(frame))))

        # Zero crossing rate to distinguish speech fricatives/vowels from static hum
        zcr = float(np.mean(np.abs(np.diff(np.signbit(frame)))))

        # Spectral flatness estimate using geometric vs arithmetic mean of power
        fft_vals = np.abs(np.fft.rfft(frame)) + 1e-10
        power = fft_vals ** 2
        geom_mean = np.exp(np.mean(np.log(power)))
        arith_mean = np.mean(power)
        spectral_flatness = float(geom_mean / arith_mean)

        # Non-flat spectra with sufficient energy and moderate ZCR characterize speech
        if rms < self.min_energy_threshold:
            prob = float(np.clip(rms / (self.min_energy_threshold + 1e-6) * 0.2, 0.0, 0.2))
        else:
            # Energy component + tonal richness indicator
            energy_score = min(1.0, rms * 15.0)
            tonality_score = 1.0 - min(1.0, spectral_flatness * 1.5)
            zcr_penalty = 1.0 if (0.02 <= zcr <= 0.65) else 0.5
            prob = float(np.clip(0.4 * energy_score + 0.4 * tonality_score + 0.2 * zcr_penalty, 0.0, 1.0))

        is_speech = prob >= self.speech_threshold
        return VADDecision(
            is_speech=is_speech,
            speech_probability=prob,
            energy_rms=rms,
            frame_samples=len(frame),
        )
