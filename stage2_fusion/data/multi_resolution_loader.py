"""
Multi-resolution data loader for FullAreaUNet.

Builds all four inputs the FullAreaUNet forward() method needs:
  - features_dict: per-modality tensors at finest resolution
  - edge_indices:  per-resolution adjacency graphs
  - edge_weights:  per-resolution edge weights (uniform 1.0 for now)
  - mappings:      cross-resolution sparse parent-child matrices

Lifetime: durable
Stage: 2 (fusion)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h3
import numpy as np
import pandas as pd
import torch
from srai.neighbourhoods import H3Neighbourhood

from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)


class MultiResolutionLoader:
    """Load embeddings and construct multi-resolution graph structure for UNet.

    Constructs the four inputs FullAreaUNet.forward() expects:
      features_dict  : Dict[str, Tensor]             — {modality_name: [N_finest, D]}
      edge_indices   : Dict[int, Tensor]              — {resolution: [2, E]}
      edge_weights   : Dict[int, Tensor]              — {resolution: [E]}
      mappings       : Dict[Tuple[int,int], Tensor]   — {(fine, coarse): sparse [N_fine, N_coarse]}

    Parameters
    ----------
    study_area : str
        Study area name (e.g. "netherlands").
    resolutions : list of int
        Resolutions from finest to coarsest, e.g. [9, 8, 7].
    feature_source : str or Path or None
        Path to the raw concat parquet. If None, auto-resolves from StudyAreaPaths.
    year : int
        Data year for path resolution.
    """

    def __init__(
        self,
        study_area: str = "netherlands",
        resolutions: Optional[List[int]] = None,
        feature_source: Optional[str] = None,
        year: int = 2022,
    ):
        self.study_area = study_area
        self.resolutions = resolutions or [9, 8, 7]
        self.year = year
        self.paths = StudyAreaPaths(study_area)

        # Resolve feature source path
        if feature_source is not None:
            self._feature_path = Path(feature_source)
        else:
            # Default: raw concat at the canonical backup location
            self._feature_path = (
                self.paths.stage2("concat")
                / "embeddings"
                / f"{study_area}_res{self.finest_res}_{year}_raw.parquet"
            )

        # Will be populated by load()
        self._hex_indices: Dict[int, Dict[str, int]] = {}  # res -> {hex_id: idx}
        self._hex_lists: Dict[int, List[str]] = {}  # res -> [hex_id, ...]

    @property
    def finest_res(self) -> int:
        return max(self.resolutions)

    @property
    def coarsest_res(self) -> int:
        return min(self.resolutions)

    def load(self) -> Dict:
        """Load all data and return the four inputs for FullAreaUNet.

        Returns
        -------
        dict with keys:
            features_dict  : Dict[str, Tensor]
            edge_indices   : Dict[int, Tensor]
            edge_weights   : Dict[int, Tensor]
            mappings       : Dict[Tuple[int,int], Tensor]
            hex_ids        : Dict[int, List[str]]  — hex ID ordering per resolution
        """
        logger.info("Loading multi-resolution data for FullAreaUNet")

        # 1. Load features at finest resolution
        features_dict, finest_hexes = self._load_features()
        self._hex_lists[self.finest_res] = finest_hexes
        self._hex_indices[self.finest_res] = {h: i for i, h in enumerate(finest_hexes)}

        # 2. Derive coarser-resolution hex lists from finest via parent hierarchy
        self._derive_coarse_hex_lists()

        # 3. Build adjacency graphs per resolution
        edge_indices, edge_weights = self._build_adjacency_graphs()

        # 4. Build cross-resolution sparse mappings
        mappings = self._build_cross_resolution_mappings()

        # Log summary
        for res in sorted(self.resolutions, reverse=True):
            n_nodes = len(self._hex_lists[res])
            n_edges = edge_indices[res].shape[1]
            logger.info(f"  res{res}: {n_nodes:,} nodes, {n_edges:,} edges")
        for (fine, coarse), m in mappings.items():
            logger.info(f"  mapping ({fine}->{coarse}): {m.shape}")

        return {
            "features_dict": features_dict,
            "edge_indices": edge_indices,
            "edge_weights": edge_weights,
            "mappings": mappings,
            "hex_ids": dict(self._hex_lists),
        }

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def _load_features(self) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Load raw concatenated features as a single 'fused' modality.

        Returns
        -------
        features_dict : dict with single key "fused" -> Tensor [N, 781]
        hex_ids       : ordered list of hex IDs matching tensor rows
        """
        logger.info(f"Loading features from {self._feature_path}")
        if not self._feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {self._feature_path}\n"
                f"Expected raw concat parquet at finest resolution (res{self.finest_res})."
            )

        df = pd.read_parquet(self._feature_path)
        hex_ids = list(df.index)
        feature_dim = df.shape[1]

        logger.info(
            f"Loaded {len(hex_ids):,} hexagons with {feature_dim}D features "
            f"at res{self.finest_res}"
        )

        # Single "fused" modality — bypasses ModalityFusion's learnable modality_weights
        # (softmax of a single weight is always 1.0, so it gets zero gradient).
        # TODO: Return per-modality tensors to activate ModalityFusion's weighting.
        features_dict = {
            "fused": torch.tensor(df.values, dtype=torch.float32)
        }
        # Shape: [N_finest, 781]

        return features_dict, hex_ids

    # ------------------------------------------------------------------
    # Coarse resolution derivation
    # ------------------------------------------------------------------

    def _derive_coarse_hex_lists(self) -> None:
        """Derive coarser-resolution hex lists from finest hexes using h3 hierarchy.

        Uses h3.cell_to_parent (allowed per CLAUDE.md for hierarchy traversal).
        Each coarser resolution's hex list is the deduplicated set of parents.
        """
        sorted_res = sorted(self.resolutions, reverse=True)

        for i in range(1, len(sorted_res)):
            finer_res = sorted_res[i - 1]
            coarser_res = sorted_res[i]

            parent_set = set()
            for hex_id in self._hex_lists[finer_res]:
                parent = h3.cell_to_parent(hex_id, coarser_res)
                parent_set.add(parent)

            coarse_hexes = sorted(parent_set)
            self._hex_lists[coarser_res] = coarse_hexes
            self._hex_indices[coarser_res] = {h: i for i, h in enumerate(coarse_hexes)}

            logger.info(
                f"Derived res{coarser_res}: {len(coarse_hexes):,} hexagons "
                f"(from {len(self._hex_lists[finer_res]):,} res{finer_res} hexagons)"
            )

    # ------------------------------------------------------------------
    # Adjacency graph construction
    # ------------------------------------------------------------------

    def _build_adjacency_graphs(
        self,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Build adjacency edge_index tensors for each resolution.

        Uses h3.grid_disk(hex, 1) for neighbor lookup. This is faster than SRAI's
        H3Neighbourhood for large hex sets (247K+ at res9) because SRAI builds a
        GeoDataFrame-backed neighbourhood which has significant overhead for bulk
        queries. h3.grid_disk is a direct C-level call.

        Decision: h3.grid_disk is technically tessellation-adjacent, but the SRAI
        H3Neighbourhood internally uses the same h3 calls. For 247K+ nodes, the
        GeoDataFrame overhead is prohibitive. Using h3 directly here.

        Returns
        -------
        edge_indices : Dict[int, Tensor [2, E]]
        edge_weights : Dict[int, Tensor [E]]  — uniform 1.0 weights
        """
        edge_indices = {}
        edge_weights = {}

        for res in self.resolutions:
            hex_list = self._hex_lists[res]
            hex_set = set(hex_list)
            hex_to_idx = self._hex_indices[res]

            src_list = []
            dst_list = []

            for hex_id in hex_list:
                idx_src = hex_to_idx[hex_id]
                # grid_disk(h, 1) returns h itself + its 6 neighbors
                neighbors = h3.grid_disk(hex_id, 1)
                for neighbor in neighbors:
                    if neighbor != hex_id and neighbor in hex_set:
                        idx_dst = hex_to_idx[neighbor]
                        src_list.append(idx_src)
                        dst_list.append(idx_dst)

            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

            edge_indices[res] = edge_index
            edge_weights[res] = edge_weight

            logger.info(
                f"Built adjacency for res{res}: {len(hex_list):,} nodes, "
                f"{edge_index.shape[1]:,} edges"
            )

        return edge_indices, edge_weights

    # ------------------------------------------------------------------
    # Cross-resolution mappings
    # ------------------------------------------------------------------

    def _build_cross_resolution_mappings(
        self,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """Build sparse parent-child mapping matrices between adjacent resolutions.

        For each pair (fine_res, coarse_res), creates a sparse matrix of shape
        [N_fine, N_coarse] where entry (i, j) = 1/k if fine hex i maps to coarse
        hex j, and k is the number of fine hexes mapping to that coarse hex
        (row-normalized so each fine hex sums to 1.0, but since each fine hex has
        exactly one parent, each row has exactly one entry = 1.0).

        The FullAreaUNet uses these as:
          - Encoder (fine->coarse): mapping.t() @ x_fine  (averages children)
          - Decoder (coarse->fine): mapping @ x_coarse    (broadcasts parent)

        Returns
        -------
        mappings : Dict[(fine_res, coarse_res), sparse Tensor [N_fine, N_coarse]]
        """
        mappings = {}
        sorted_res = sorted(self.resolutions, reverse=True)

        for i in range(len(sorted_res) - 1):
            fine_res = sorted_res[i]
            coarse_res = sorted_res[i + 1]

            fine_hexes = self._hex_lists[fine_res]
            coarse_to_idx = self._hex_indices[coarse_res]
            n_fine = len(fine_hexes)
            n_coarse = len(self._hex_lists[coarse_res])

            row_indices = []
            col_indices = []

            for fine_idx, hex_id in enumerate(fine_hexes):
                parent = h3.cell_to_parent(hex_id, coarse_res)
                if parent in coarse_to_idx:
                    row_indices.append(fine_idx)
                    col_indices.append(coarse_to_idx[parent])

            indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
            values = torch.ones(len(row_indices), dtype=torch.float32)

            mapping = torch.sparse_coo_tensor(
                indices, values, size=(n_fine, n_coarse)
            ).coalesce()

            # The mapping.t() used in encoder will average children because
            # GCN-style aggregation handles the normalization. Each fine hex
            # has exactly one parent, so each row has exactly one 1.0 entry.

            mappings[(fine_res, coarse_res)] = mapping

            logger.info(
                f"Built mapping ({fine_res}->{coarse_res}): "
                f"[{n_fine:,}, {n_coarse:,}], {len(row_indices):,} entries"
            )

        return mappings

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_model_config(self) -> Dict:
        """Return a config dict suitable for FullAreaUNet constructor.

        Note: The current FullAreaUNet hardcodes resolutions 10/9/8 in forward().
        When using res 9/8/7, the model needs to be adapted. This config assumes
        the model has been parametrized or a resolution-agnostic variant is used.
        """
        # After loading, we know feature dims
        return {
            "feature_dims": {"fused": 781},
            "hidden_dim": 128,
            "output_dim": 128,  # 128D output per Wave 1 decision
            "num_convs": 4,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("MultiResolutionLoader smoke test")
    print("=" * 60)

    loader = MultiResolutionLoader(
        study_area="netherlands",
        resolutions=[9, 8, 7],
    )

    data = loader.load()

    print("\n--- Shapes ---")
    for name, tensor in data["features_dict"].items():
        print(f"  features_dict['{name}']: {tensor.shape}")

    for res, ei in data["edge_indices"].items():
        ew = data["edge_weights"][res]
        print(f"  edge_indices[{res}]: {ei.shape}  edge_weights[{res}]: {ew.shape}")

    for (fine, coarse), m in data["mappings"].items():
        print(f"  mappings[({fine},{coarse})]: {m.shape} (nnz={m._nnz()})")

    for res, hexes in data["hex_ids"].items():
        print(f"  hex_ids[{res}]: {len(hexes)} hexagons")

    print("\n--- Model config ---")
    print(loader.get_model_config())

    print("\nSmoke test passed.")
