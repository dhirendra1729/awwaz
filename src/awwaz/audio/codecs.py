"""
Telephony Codec Simulator and Channel Laundering Pipeline.
Simulates PSTN G.711 mu-law, GSM AMR-NB, VoLTE AMR-WB, and Opus codecs.
"""

from enum import Enum
from typing import Tuple
import numpy as np


class CodecType(str, Enum):
    G711_ULAW = "g711_ulaw"    # 8 kHz, 64 kbps (PSTN Landline)
    AMR_NB = "amr_nb"          # 8 kHz, 12.2 kbps (GSM 2G/3G Cellular)
    AMR_WB = "amr_wb"          # 16 kHz, 23.85 kbps (VoLTE HD Voice)
    OPUS_LOW = "opus_low"      # 16 kHz, 12 kbps (VoIP / WhatsApp)
    OPUS_STD = "opus_std"      # 16 kHz, 32 kbps (WebRTC High-Fi)


class TelephonyCodecSimulator:
    """
    Simulates real-world lossy telephony degradation and channel laundering.
    Applies bandpass filtering, mu-law / AMR companding quantization, and down/up-sampling.
    """
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def apply_codec(self, audio: np.ndarray, codec: CodecType) -> np.ndarray:
        """Process 1D float32 audio through the specified telephony codec channel."""
        if audio.size == 0:
            return audio

        if codec == CodecType.G711_ULAW:
            return self._simulate_g711_ulaw(audio)
        elif codec == CodecType.AMR_NB:
            return self._simulate_amr_nb(audio)
        elif codec == CodecType.AMR_WB:
            return self._simulate_amr_wb(audio)
        elif codec == CodecType.OPUS_LOW:
            return self._simulate_opus(audio, bitrate_kbps=12.0)
        elif codec == CodecType.OPUS_STD:
            return self._simulate_opus(audio, bitrate_kbps=32.0)
        return audio

    def _simulate_g711_ulaw(self, audio: np.ndarray) -> np.ndarray:
        """G.711 mu-law: 8 kHz downsampling, 300Hz-3400Hz bandpass, 8-bit non-linear quantization."""
        # 1. Bandpass filter via FFT
        filtered = self._fft_bandpass(audio, low_hz=300.0, high_hz=3400.0, sr=self.target_sample_rate)

        # 2. Downsample to 8 kHz and upsample back to 16 kHz
        downsampled = self._resample_simple(filtered, orig_sr=self.target_sample_rate, target_sr=8000)

        # 3. Mu-law companding (mu = 255)
        mu = 255.0
        x = np.clip(downsampled, -1.0, 1.0)
        companded = np.sign(x) * np.log(1.0 + mu * np.abs(x)) / np.log(1.0 + mu)

        # 8-bit uniform quantization
        quantized = np.round((companded + 1.0) * 127.5) / 127.5 - 1.0

        # Expand back
        expanded = np.sign(quantized) * ((1.0 + mu) ** np.abs(quantized) - 1.0) / mu

        # Resample back to original sample rate
        return self._resample_simple(expanded, orig_sr=8000, target_sr=self.target_sample_rate)

    def _simulate_amr_nb(self, audio: np.ndarray) -> np.ndarray:
        """AMR-NB: 8 kHz, 200Hz-3400Hz, aggressive CELP spectral smearing and high-frequency stripping."""
        filtered = self._fft_bandpass(audio, low_hz=200.0, high_hz=3400.0, sr=self.target_sample_rate)
        downsampled = self._resample_simple(filtered, orig_sr=self.target_sample_rate, target_sr=8000)

        # CELP codebook approximation: coarse quantization + light harmonic smearing
        noise = np.random.normal(0, 0.005, size=downsampled.shape).astype(np.float32)
        quantized = np.round(downsampled * 64.0) / 64.0 + noise

        return self._resample_simple(quantized, orig_sr=8000, target_sr=self.target_sample_rate)

    def _simulate_amr_wb(self, audio: np.ndarray) -> np.ndarray:
        """AMR-WB: 16 kHz HD Voice, 50Hz-7000Hz."""
        filtered = self._fft_bandpass(audio, low_hz=50.0, high_hz=7000.0, sr=self.target_sample_rate)
        # Moderate bit-depth quantization (14-bit equivalent)
        return np.round(filtered * 2048.0) / 2048.0

    def _simulate_opus(self, audio: np.ndarray, bitrate_kbps: float) -> np.ndarray:
        """Opus low/standard bitrate simulation with psychoacoustic masking envelope."""
        cutoff = 6000.0 if bitrate_kbps < 16.0 else 8000.0
        filtered = self._fft_bandpass(audio, low_hz=50.0, high_hz=cutoff, sr=self.target_sample_rate)
        quant_levels = 128.0 if bitrate_kbps < 16.0 else 512.0
        return np.round(filtered * quant_levels) / quant_levels

    def _fft_bandpass(self, audio: np.ndarray, low_hz: float, high_hz: float, sr: int) -> np.ndarray:
        n = len(audio)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        fft_data = np.fft.rfft(audio)
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        fft_data[~mask] *= 0.01  # -40dB suppression outside passband
        return np.fft.irfft(fft_data, n=n).astype(np.float32)

    def _resample_simple(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        num_target = int(len(audio) * target_sr / orig_sr)
        if num_target == 0:
            return np.zeros(0, dtype=np.float32)
        x_orig = np.linspace(0, 1, len(audio), endpoint=False)
        x_target = np.linspace(0, 1, num_target, endpoint=False)
        return np.interp(x_target, x_orig, audio).astype(np.float32)
