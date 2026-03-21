"""
Multi-resolution data loader for FullAreaUNet.

Builds all four inputs the FullAreaUNet forward() method needs:
  - features_dict: per-modality tensors at finest resolution
  - edge_indices:  per-resolution adjacency graphs
  - edge_weights:  per-resolution edge weights (uniform 1.0 or from accessibility graph)
  - mappings:      cross-resolution sparse parent-child matrices

Lifetime: durable
Stage: 2 (fusion)
"""

import logging
import pickle
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
    year : str
        Data year label for path resolution (e.g. "2022", "20mix").
    accessibility_graph : str or Path or None
        Path to an accessibility-weighted edge Parquet produced by
        ``stage2_fusion.graphs.accessibility_graph.build_accessibility_graph``.
        When provided, the finest resolution uses edges and gravity weights
        from this file instead of a uniform 1-ring lattice.  Coarser
        resolutions still fall back to uniform 1-ring adjacency.
        When None (default), all resolutions use uniform 1-ring with
        edge_weight = 1.0 (backward-compatible).
    """

    def __init__(
        self,
        study_area: str = "netherlands",
        resolutions: Optional[List[int]] = None,
        feature_source: Optional[str] = None,
        year: str = "2022",
        accessibility_graph: Optional[str] = None,
    ):
        self.study_area = study_area
        self.resolutions = resolutions or [9, 8, 7]
        self.year = year
        self.paths = StudyAreaPaths(study_area)

        # Accessibility graph (finest resolution only)
        self._accessibility_graph_path: Optional[Path] = (
            Path(accessibility_graph) if accessibility_graph is not None else None
        )

        # Resolve feature source path
        if feature_source is not None:
            self._feature_path = Path(feature_source)
        else:
            # Default: try _raw.parquet first (legacy), then plain .parquet
            raw_path = (
                self.paths.stage2("concat")
                / "embeddings"
                / f"{study_area}_res{self.finest_res}_{year}_raw.parquet"
            )
            plain_path = (
                self.paths.stage2("concat")
                / "embeddings"
                / f"{study_area}_res{self.finest_res}_{year}.parquet"
            )
            self._feature_path = raw_path if raw_path.exists() else plain_path

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

    def _load_neighbourhood(self, resolution: int) -> H3Neighbourhood:
        """Load a precomputed H3Neighbourhood pickle, falling back to a fresh instance.

        Precomputed pickles are written by scripts/stage2/precompute_neighbourhoods.py
        and stored in StudyAreaPaths.neighbourhood_dir().  A precomputed neighbourhood
        has _available_indices populated (region-filtered), so get_neighbours() returns
        only in-study-area cells without any manual filtering by the caller.

        Falls back to a bare H3Neighbourhood() if no pickle exists (backward-compatible:
        the caller's ``if neighbor in hex_set`` guard still works correctly in that case).

        Parameters
        ----------
        resolution:
            H3 resolution to load the neighbourhood for.

        Returns
        -------
        H3Neighbourhood instance (possibly region-filtered).
        """
        pickle_path = (
            self.paths.neighbourhood_dir()
            / f"{self.study_area}_res{resolution}_neighbourhood.pkl"
        )
        if pickle_path.exists():
            logger.info(f"Loading precomputed neighbourhood for res{resolution}: {pickle_path}")
            with open(pickle_path, "rb") as f:
                neighbourhood = pickle.load(f)
            n_available = (
                len(neighbourhood._available_indices)
                if neighbourhood._available_indices
                else 0
            )
            logger.info(f"  Loaded neighbourhood with {n_available:,} regions")
            return neighbourhood

        logger.warning(
            f"No precomputed neighbourhood pickle found for res{resolution} at "
            f"{pickle_path}.  Falling back to bare H3Neighbourhood() — "
            f"run scripts/stage2/precompute_neighbourhoods.py to cache it."
        )
        return H3Neighbourhood()

    def _build_adjacency_graphs(
        self,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Build adjacency edge_index tensors for each resolution.

        When an accessibility graph Parquet is provided, the **finest**
        resolution uses edges and gravity weights from that file.  Coarser
        resolutions always use the uniform 1-ring lattice (edge_weight=1.0).

        For the uniform path, uses SRAI's H3Neighbourhood.get_neighbours()
        for neighbor lookup per CLAUDE.md policy.  Loads precomputed
        neighbourhood pickles from StudyAreaPaths.neighbourhood_dir() when
        available.

        Returns
        -------
        edge_indices : Dict[int, Tensor [2, E]]
        edge_weights : Dict[int, Tensor [E]]
        """
        edge_indices = {}
        edge_weights = {}

        for res in self.resolutions:
            # Use accessibility graph for the finest resolution when available
            if res == self.finest_res and self._accessibility_graph_path is not None:
                ei, ew = self._load_accessibility_edges(res)
                edge_indices[res] = ei
                edge_weights[res] = ew
            else:
                ei, ew = self._build_uniform_adjacency(res)
                edge_indices[res] = ei
                edge_weights[res] = ew

        return edge_indices, edge_weights

    def _build_uniform_adjacency(
        self, res: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a uniform 1-ring adjacency graph for a single resolution.

        Returns edge_index [2, E] and edge_weight [E] (all 1.0).
        """
        neighbourhood = self._load_neighbourhood(res)

        hex_list = self._hex_lists[res]
        hex_set = set(hex_list)
        hex_to_idx = self._hex_indices[res]

        src_list = []
        dst_list = []

        for hex_id in hex_list:
            idx_src = hex_to_idx[hex_id]
            neighbors = neighbourhood.get_neighbours(hex_id)
            for neighbor in neighbors:
                if neighbor in hex_set:
                    idx_dst = hex_to_idx[neighbor]
                    src_list.append(idx_src)
                    dst_list.append(idx_dst)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

        logger.info(
            f"Built uniform adjacency for res{res}: {len(hex_list):,} nodes, "
            f"{edge_index.shape[1]:,} edges"
        )

        return edge_index, edge_weight

    def _load_accessibility_edges(
        self, res: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load edges and gravity weights from an accessibility Parquet.

        Reads the Parquet produced by
        ``stage2_fusion.graphs.accessibility_graph.build_accessibility_graph``,
        maps hex string IDs to the integer indices already established in
        ``self._hex_indices[res]``, and uses ``gravity_weight`` as the edge
        weight (it already incorporates distance decay and building mass).

        Edges whose origin or destination hex is not in the loader's hex set
        for this resolution are silently dropped (the accessibility graph may
        cover a slightly different footprint).

        Returns
        -------
        edge_index : Tensor [2, E]
        edge_weight : Tensor [E]
        """
        path = self._accessibility_graph_path
        if not path.exists():
            raise FileNotFoundError(
                f"Accessibility graph Parquet not found: {path}"
            )

        logger.info(f"Loading accessibility edges from {path}")
        df = pd.read_parquet(path)

        required_cols = {"origin_hex", "dest_hex", "gravity_weight"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Accessibility Parquet is missing columns: {missing}. "
                f"Expected: {required_cols}"
            )

        hex_to_idx = self._hex_indices[res]

        # Filter to edges where both endpoints are in our hex set
        mask_origin = df["origin_hex"].isin(hex_to_idx)
        mask_dest = df["dest_hex"].isin(hex_to_idx)
        df = df[mask_origin & mask_dest]

        if len(df) == 0:
            logger.warning(
                f"No accessibility edges matched hex set at res{res}. "
                f"Falling back to uniform 1-ring adjacency."
            )
            return self._build_uniform_adjacency(res)

        # Map hex IDs to integer indices
        src_indices = df["origin_hex"].map(hex_to_idx).values
        dst_indices = df["dest_hex"].map(hex_to_idx).values

        edge_index = torch.tensor(
            np.stack([src_indices, dst_indices], axis=0), dtype=torch.long
        )
        edge_weight = torch.tensor(
            df["gravity_weight"].values, dtype=torch.float32
        )

        n_nodes = len(self._hex_lists[res])
        logger.info(
            f"Loaded accessibility graph for res{res}: {n_nodes:,} nodes, "
            f"{edge_index.shape[1]:,} edges "
            f"(gravity_weight range: [{edge_weight.min():.4f}, {edge_weight.max():.4f}])"
        )

        # TODO (Option B): For coarser resolutions, aggregate gravity weights
        # from children's edges instead of falling back to uniform 1-ring.
        # Implementation: for each coarser hex pair, average the gravity
        # weights of edges between their children hexagons.

        return edge_index, edge_weight

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

    # Optional: pass --accessibility-graph path/to/walk_res9.parquet
    import argparse

    parser = argparse.ArgumentParser(description="MultiResolutionLoader smoke test")
    parser.add_argument(
        "--accessibility-graph",
        default=None,
        help="Path to accessibility Parquet for finest resolution edges",
    )
    args = parser.parse_args()

    loader = MultiResolutionLoader(
        study_area="netherlands",
        resolutions=[9, 8, 7],
        accessibility_graph=args.accessibility_graph,
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
