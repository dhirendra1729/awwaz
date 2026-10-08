"""
SpAArSIST Asymmetric Graph Attention Pooling (Interspeech 2026).
Reference: Sparsified Graph Attention Networks for Efficient Audio Anti-Spoofing.

Implements asymmetric graph node pruning (k_inf = 0.5 < k_tr = 0.8) using
parameter-free L2-norm scoring, slashing backend compute MACs by over 20%.
"""

from typing import Tuple
import numpy as np


class AsymmetricGraphPruner:
    """
    Selects top-k most salient graph nodes based on parameter-free L2 energy norm.
    Inference retains top 50% nodes (k_inf = 0.5); training retains 80% (k_tr = 0.8).
    """
    def __init__(self, k_tr: float = 0.80, k_inf: float = 0.50):
        self.k_tr = k_tr
        self.k_inf = k_inf

    def prune(self, node_features: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        node_features shape: (num_nodes, node_dim)
        Returns: pruned node features of shape (k * num_nodes, node_dim)
        """
        if node_features.shape[0] <= 2:
            return node_features

        ratio = self.k_tr if is_training else self.k_inf
        k_count = max(1, int(node_features.shape[0] * ratio))

        # Parameter-free L2 norm node importance scoring
        l2_scores = np.linalg.norm(node_features, axis=1)
        top_indices = np.argsort(l2_scores)[-k_count:]
        top_indices = np.sort(top_indices)  # Maintain temporal order

        return node_features[top_indices]


class SpAArSISTGraphAttention:
    """
    Heterogeneous Spectro-Temporal Graph Attention layer with asymmetric pooling.
    """
    def __init__(self, node_dim: int = 64, k_inf: float = 0.50, k_tr: float = 0.80):
        self.node_dim = node_dim
        self.pruner = AsymmetricGraphPruner(k_tr=k_tr, k_inf=k_inf)
        rng = np.random.RandomState(42)
        self.w_gat = rng.randn(node_dim, node_dim).astype(np.float32) * 0.1

    def pool_graph(self, nodes: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Applies asymmetric pruning followed by attentive graph pooling.
        Returns graph-level summary vector of shape (node_dim,).
        """
        if nodes.size == 0:
            return np.zeros(self.node_dim, dtype=np.float32)

        # 1. Prune redundant nodes
        pruned_nodes = self.pruner.prune(nodes, is_training=is_training)

        # 2. Graph attention message passing
        h = pruned_nodes @ self.w_gat

        # 3. Attentive pooling: calculate node importance weights
        weights = np.exp(np.mean(h, axis=1))
        weights = weights / (np.sum(weights) + 1e-8)

        # 4. Weighted graph sum representation
        graph_repr = np.sum(pruned_nodes * weights[:, np.newaxis], axis=0)
        norm = np.linalg.norm(graph_repr) + 1e-8
        return graph_repr / norm
