"""
Simple ring aggregation for spatially smoothing urban embeddings.

Computes a weighted mean of k-ring neighbourhood embeddings for each hexagon.
No learnable parameters -- purely geometric spatial smoothing.

Ring k=0 is the hexagon itself, k=1 is its immediate neighbours, etc.
Weights are normalized to sum to 1 across all rings.

Weighting schemes:
- exponential: W_k = e^{-k}  (strong centre bias)
- logarithmic: W_k = 1 / log2(k + 2)
- linear: W_k = 1 - k/K
- flat: W_k = 1/K  (uniform across rings)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

WeightingScheme = Literal["exponential", "logarithmic", "linear", "flat"]


def _compute_weights(K: int, scheme: WeightingScheme) -> np.ndarray:
    """Compute normalized ring weights for rings 0..K.

    Returns:
        Array of shape (K+1,) summing to 1.
    """
    k = np.arange(K + 1, dtype=np.float64)

    if scheme == "exponential":
        w = np.exp(-k)
    elif scheme == "logarithmic":
        w = 1.0 / np.log2(k + 2)
    elif scheme == "linear":
        w = 1.0 - k / K if K > 0 else np.ones(1)
    elif scheme == "flat":
        w = np.ones(K + 1)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme!r}")

    w /= w.sum()
    return w


class SimpleRingAggregator:
    """Simple ring aggregation -- no learnable params.

    For each hexagon, computes weighted mean of k-ring neighbourhood embeddings.
    Uses SRAI H3Neighbourhood for k-ring computation.

    Args:
        neighbourhood: SRAI H3Neighbourhood instance (possibly loaded from pickle).
        K: Maximum ring distance (default 3). Ring 0 = self, ring K = K-th ring.
        weighting: One of 'exponential', 'logarithmic', 'linear', 'flat'.
    """

    def __init__(
        self,
        neighbourhood,
        K: int = 3,
        weighting: WeightingScheme = "exponential",
    ):
        self.neighbourhood = neighbourhood
        self.K = K
        self.weighting = weighting
        self.weights = _compute_weights(K, weighting)

    def aggregate(self, embeddings_df: pd.DataFrame) -> pd.DataFrame:
        """Apply ring aggregation to spatially smooth embeddings.

        Args:
            embeddings_df: DataFrame indexed by ``region_id`` (H3 hex IDs)
                with numeric embedding columns.

        Returns:
            DataFrame with same index and columns, but spatially smoothed.
            Each row is a weighted combination of the original hex embedding
            and the mean embeddings of its k-ring neighbours.
        """
        hex_ids = embeddings_df.index.tolist()
        hex_set = set(hex_ids)
        n_hexes = len(hex_ids)
        n_dims = embeddings_df.shape[1]

        # Pre-extract numpy array for fast indexing
        values = embeddings_df.values  # (n_hexes, n_dims)
        hex_to_idx = {h: i for i, h in enumerate(hex_ids)}

        logger.info(
            "Ring aggregation: %d hexagons, %d dims, K=%d, weighting=%s",
            n_hexes, n_dims, self.K, self.weighting,
        )
        logger.info("Ring weights: %s", self.weights)

        result = np.zeros((n_hexes, n_dims), dtype=np.float64)

        for k in range(self.K + 1):
            w_k = self.weights[k]
            if w_k == 0:
                continue

            ring_means = np.zeros((n_hexes, n_dims), dtype=np.float64)

            for i, hex_id in enumerate(tqdm(
                hex_ids,
                desc=f"Ring k={k}",
                disable=n_hexes < 10000,
            )):
                if k == 0:
                    # Ring 0 is the hexagon itself
                    ring_means[i] = values[i]
                else:
                    neighbours = self.neighbourhood.get_neighbours_at_distance(
                        hex_id, k
                    )
                    # Filter to only hexagons present in our embedding set
                    valid = [hex_to_idx[n] for n in neighbours if n in hex_set]
                    if valid:
                        ring_means[i] = values[valid].mean(axis=0)
                    else:
                        # No neighbours at this distance (boundary hex).
                        # Fall back to self-embedding so weight isn't wasted.
                        ring_means[i] = values[i]

            result += w_k * ring_means

        out_df = pd.DataFrame(
            result,
            index=embeddings_df.index,
            columns=embeddings_df.columns,
        )
        out_df.index.name = embeddings_df.index.name
        return out_df
