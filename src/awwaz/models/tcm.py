"""
Temporal-Channel Modeling (TCM) Module.
Reference: Interspeech 2024 - Modeling Temporal and Channel Dynamics for Audio Anti-Spoofing.

Standard self-attention models treat channels uniformly. TCM introduces dedicated
channel-wise excitation alongside temporal attention to capture spectral sub-band
vocoder smearing and vocoder discontinuities.
"""

from typing import Tuple
import numpy as np


class TemporalChannelModelingModule:
    """
    Dual-branch TCM block:
    Branch 1: Channel-Attention (identifies anomalous spectral sub-band distributions)
    Branch 2: Temporal-Attention (captures temporal phase jitter across frames)
    """
    def __init__(self, channels: int = 64, reduction_ratio: int = 4):
        self.channels = channels
        self.reduction_ratio = max(1, reduction_ratio)
        reduced_dim = max(1, channels // self.reduction_ratio)
        
        # Channel attention projection weights
        rng = np.random.RandomState(42)
        self.w_c1 = rng.randn(channels, reduced_dim).astype(np.float32) * 0.1
        self.w_c2 = rng.randn(reduced_dim, channels).astype(np.float32) * 0.1

        # Temporal attention projection weights
        self.w_q = rng.randn(channels, channels).astype(np.float32) * 0.1
        self.w_k = rng.randn(channels, channels).astype(np.float32) * 0.1

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Input: feature map of shape (Time, Channels) or (Channels, Time)
        Returns: refined feature map of same shape with channel and temporal modulation.
        """
        if x.size == 0:
            return x

        # Ensure shape (Time, Channels)
        transpose_needed = False
        if x.shape[0] == self.channels and len(x.shape) == 2 and x.shape[1] != self.channels:
            x = x.T
            transpose_needed = True

        time_steps, c_dim = x.shape
        if c_dim != self.channels:
            # Linear projection to target channel dim if needed
            proj = np.eye(c_dim, self.channels, dtype=np.float32) if c_dim <= self.channels else np.eye(self.channels, c_dim, dtype=np.float32).T
            x = x @ proj
            c_dim = self.channels

        # 1. Channel Attention: Global Average Pooling over time -> MLP -> Sigmoid
        gap = np.mean(x, axis=0)  # Shape (Channels,)
        c_att = np.maximum(0, gap @ self.w_c1)  # ReLU
        c_weights = self._sigmoid(c_att @ self.w_c2)  # Shape (Channels,)
        x_channel = x * c_weights

        # 2. Temporal Attention: Scaled Dot-Product Attention over time steps
        q = x @ self.w_q
        k = x @ self.w_k
        scores = (q @ k.T) / np.sqrt(c_dim)
        t_weights = self._softmax(scores, axis=-1)
        x_temporal = t_weights @ x

        # Residual fusion
        out = x + 0.5 * x_channel + 0.5 * x_temporal

        if transpose_needed:
            return out.T
        return out
