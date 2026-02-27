"""
Hierarchical Cone Inference System
===================================

Applies trained model to ALL possible cones for complete coverage.

Key strategies:
1. Process every cone (no sampling)
2. Handle overlapping regions - hexagons appear in multiple cones
3. Aggregate predictions: average, weighted average, or voting
4. Edge padding: extend cone neighborhoods to ensure proper context

For a hexagon at res10, it might appear in:
- Parent cone at res5 (if within 5-ring neighborhood)
- Multiple ancestor cones through the hierarchy
- We average all predictions for that hexagon
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from utils.paths import StudyAreaPaths

from ..data.hierarchical_cone_masking import (
    HierarchicalConeMaskingSystem,
    ConeBatcher,  # LEGACY - uses all cones in memory
    LazyConeBatcher,  # NEW - loads individual files on-demand
    HierarchicalCone
)
from ..models.cone_batching_unet import ConeBatchingUNet

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for cone-based inference."""

    # Cone parameters (must match training)
    parent_resolution: int = 5
    target_resolution: int = 10
    neighbor_rings: int = 5

    # Inference parameters
    batch_size: int = 64  # Larger batches for inference (no backprop)
    aggregation_method: str = "average"  # "average", "weighted_average", "median"
    edge_padding_rings: int = 1  # Extra rings for boundary context

    # Device
    device: str = "auto"


class HierarchicalConeInference:
    """
    Inference engine for hierarchical cone models.

    Applies model to all cones and aggregates overlapping predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig,
        study_area: str = "netherlands"
    ):
        """
        Initialize inference engine.

        Args:
            model: Trained LatticeUNet model
            config: Inference configuration
            study_area: Name of study area
        """
        self.model = model
        self.config = config
        self.study_area = study_area

        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Initialized HierarchicalConeInference:")
        logger.info(f"  Study Area: {study_area}")
        logger.info(f"  Aggregation: {config.aggregation_method}")
        logger.info(f"  Edge Padding: {config.edge_padding_rings} rings")
        logger.info(f"  Device: {self.device}")

    def load_multi_resolution_data(
        self
    ) -> Tuple[Dict[int, gpd.GeoDataFrame], Dict[int, pd.DataFrame]]:
        """Load regions and embeddings for all resolutions."""
        from ..data.study_area_loader import StudyAreaLoader

        loader = StudyAreaLoader(study_area=self.study_area)

        # Load regions
        regions_by_res = {}
        for res in range(self.config.parent_resolution, self.config.target_resolution + 1):
            try:
                regions_gdf = loader.load_regions(res, with_geometry=True)
                regions_by_res[res] = regions_gdf
                logger.info(f"  Loaded res{res}: {len(regions_gdf)} hexagons")
            except Exception as e:
                logger.warning(f"  Could not load res{res}: {e}")

        # Load embeddings (input features)
        paths = StudyAreaPaths(self.study_area)
        embeddings_by_res = {}
        for res in range(self.config.parent_resolution, self.config.target_resolution + 1):
            embeddings_path = paths.embedding_file("alphaearth", res, year=2022)

            if embeddings_path.exists():
                embeddings_df = pd.read_parquet(embeddings_path)

                # Align indices
                if 'region_id' in embeddings_df.columns:
                    embeddings_df = embeddings_df.set_index('region_id')
                    embeddings_df.index.name = 'region_id'

                # Extract embedding columns
                embedding_cols = [col for col in embeddings_df.columns if col.startswith('A')]
                embeddings_df = embeddings_df[embedding_cols]

                # Filter to common hexagons
                if res in regions_by_res:
                    common_indices = embeddings_df.index.intersection(regions_by_res[res].index)
                    embeddings_df = embeddings_df.loc[common_indices]

                embeddings_by_res[res] = embeddings_df
                logger.info(f"  Loaded res{res} embeddings: {embeddings_df.shape}")

        return regions_by_res, embeddings_by_res

    def build_global_graph(
        self,
        regions_by_resolution: Dict[int, gpd.GeoDataFrame]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Build unified graph across all resolutions."""
        from ..graphs.hexagonal_graph_constructor import HexagonalLatticeConstructor

        # Collect all hexagons
        all_hexagons = set()
        for regions_gdf in regions_by_resolution.values():
            all_hexagons.update(regions_gdf.index)

        global_hex_list = sorted(all_hexagons)
        hex_to_idx = {h: i for i, h in enumerate(global_hex_list)}

        logger.info(f"Global graph nodes: {len(global_hex_list)}")

        # Build edges
        all_edges = []
        all_weights = []

        # Spatial edges within each resolution
        for res, regions_gdf in regions_by_resolution.items():
            constructor = HexagonalLatticeConstructor(
                device=str(self.device),
                neighbor_rings=self.config.neighbor_rings + self.config.edge_padding_rings,
                edge_weight=1.0,
                include_self_loops=False
            )

            edge_features = constructor._construct_hexagonal_lattice(
                regions_gdf, res, mode=f"res{res}"
            )

            # Convert to global indices
            local_hex_list = list(regions_gdf.index)
            local_to_global = {i: hex_to_idx[h] for i, h in enumerate(local_hex_list)}

            for i in range(len(edge_features.edge_index[0])):
                src_global = local_to_global[edge_features.edge_index[0][i]]
                tgt_global = local_to_global[edge_features.edge_index[1][i]]

                all_edges.append([src_global, tgt_global])
                all_weights.append(edge_features.edge_weights[i])

        # Hierarchical edges
        cone_system = HierarchicalConeMaskingSystem(
            parent_resolution=self.config.parent_resolution,
            target_resolution=self.config.target_resolution,
            neighbor_rings=self.config.neighbor_rings
        )

        for parent_res in range(self.config.parent_resolution, self.config.target_resolution):
            child_res = parent_res + 1

            if parent_res in regions_by_resolution and child_res in regions_by_resolution:
                parent_regions = regions_by_resolution[parent_res]

                for parent_hex in parent_regions.index:
                    children = cone_system.get_h3_children(parent_hex, child_res)

                    parent_idx = hex_to_idx.get(parent_hex)
                    if parent_idx is not None:
                        for child_hex in children:
                            child_idx = hex_to_idx.get(child_hex)
                            if child_idx is not None:
                                all_edges.append([parent_idx, child_idx])
                                all_edges.append([child_idx, parent_idx])
                                all_weights.extend([0.5, 0.5])

        # Convert to tensors
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous().to(self.device)
        edge_weights = torch.tensor(all_weights, dtype=torch.float32).to(self.device)

        logger.info(f"Global graph edges: {edge_index.shape[1]}")

        return edge_index, edge_weights, global_hex_list

    def prepare_global_features(
        self,
        embeddings_by_resolution: Dict[int, pd.DataFrame],
        global_hex_list: List[str]
    ) -> torch.Tensor:
        """Prepare feature matrix for all hexagons."""
        embedding_dim = len(list(embeddings_by_resolution.values())[0].columns)
        features = torch.zeros(len(global_hex_list), embedding_dim, dtype=torch.float32)

        hex_to_idx = {h: i for i, h in enumerate(global_hex_list)}

        for res, embeddings_df in embeddings_by_resolution.items():
            for hex_id in embeddings_df.index:
                if hex_id in hex_to_idx:
                    idx = hex_to_idx[hex_id]
                    features[idx] = torch.tensor(
                        embeddings_df.loc[hex_id].values,
                        dtype=torch.float32
                    )

        return features.to(self.device)

    @torch.no_grad()
    def infer_all_cones(
        self,
        cones: List[HierarchicalCone],
        global_features: torch.Tensor,
        global_edge_index: torch.Tensor,
        global_edge_weights: torch.Tensor,
        global_hex_list: List[str]
    ) -> Dict[str, List[Tuple[torch.Tensor, float]]]:
        """
        Run inference on all cones and collect predictions.

        For each hexagon, we collect all predictions from cones it appears in.
        Returns dict mapping hex_id -> [(prediction_tensor, weight), ...]
        """
        logger.info("\n" + "="*60)
        logger.info("Running Inference on All Cones")
        logger.info("="*60)

        # Dictionary to accumulate predictions
        # hex_id -> [(prediction, weight), ...]
        predictions_per_hex = {}

        # TODO: Update to LazyConeBatcher for 92% memory reduction
        # See train_lattice_unet_res10_cones.py::inference_with_weighted_averaging
        # Requires: cache_all_cones() + LazyConeBatcher(parent_hexagons, cache_dir)
        batcher = ConeBatcher(cones, batch_size=self.config.batch_size)  # LEGACY

        for batch_idx, cone_batch in enumerate(tqdm(batcher, desc="Inference")):
            for cone in cone_batch:
                # Get mask for this cone
                mask = cone.get_mask(global_hex_list)

                # Extract cone features
                cone_features = global_features[mask]

                # Mask edges
                cone_edge_mask = mask[global_edge_index[0]] & mask[global_edge_index[1]]
                cone_edge_index = global_edge_index[:, cone_edge_mask]
                cone_edge_weights = global_edge_weights[cone_edge_mask]

                # Remap to local indices
                global_to_local = {
                    global_hex_list[i]: local_i
                    for local_i, i in enumerate(torch.where(mask)[0].tolist())
                }

                local_edges = []
                local_weights = []
                for i in range(cone_edge_index.shape[1]):
                    src_hex = global_hex_list[cone_edge_index[0, i].item()]
                    tgt_hex = global_hex_list[cone_edge_index[1, i].item()]

                    if src_hex in global_to_local and tgt_hex in global_to_local:
                        local_edges.append([
                            global_to_local[src_hex],
                            global_to_local[tgt_hex]
                        ])
                        local_weights.append(cone_edge_weights[i].item())

                if len(local_edges) == 0:
                    continue

                local_edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous().to(self.device)
                local_edge_weights = torch.tensor(local_weights, dtype=torch.float32).to(self.device)

                # Forward pass
                output = self.model(
                    cone_features,
                    local_edge_index,
                    local_edge_weights,
                    batch=None
                )

                # Store predictions for each hexagon in this cone
                local_to_global = {v: k for k, v in global_to_local.items()}

                for local_idx in range(output.shape[0]):
                    hex_id = local_to_global[local_idx]
                    prediction = output[local_idx]

                    # Weight based on hexagon's position in cone
                    # Center of cone gets higher weight than edges
                    weight = self._compute_prediction_weight(hex_id, cone)

                    if hex_id not in predictions_per_hex:
                        predictions_per_hex[hex_id] = []

                    predictions_per_hex[hex_id].append((prediction, weight))

        logger.info(f"\nCollected predictions for {len(predictions_per_hex)} hexagons")

        # Statistics on overlap
        overlaps = [len(preds) for preds in predictions_per_hex.values()]
        logger.info(f"Prediction overlap statistics:")
        logger.info(f"  Min cones per hex: {min(overlaps)}")
        logger.info(f"  Max cones per hex: {max(overlaps)}")
        logger.info(f"  Avg cones per hex: {np.mean(overlaps):.1f}")
        logger.info(f"  Median cones per hex: {np.median(overlaps):.1f}")

        return predictions_per_hex

    def _compute_prediction_weight(
        self,
        hex_id: str,
        cone: HierarchicalCone
    ) -> float:
        """
        Compute weight for a prediction based on hexagon position in cone.

        Hexagons near the center of the cone get higher weight.
        Hexagons at edges (due to padding) get lower weight.
        """
        # If hexagon is the parent, highest weight
        if hex_id == cone.parent_hex:
            return 1.0

        # If in parent neighborhood, high weight
        if hex_id in cone.parent_neighbors:
            return 0.9

        # Otherwise, weight decreases with resolution depth
        # (descendants at finer resolutions get lower weight)
        for res, hex_set in cone.descendants_by_resolution.items():
            if hex_id in hex_set:
                # Weight decreases with depth: 0.8, 0.7, 0.6, ...
                depth = res - cone.parent_resolution
                weight = max(0.5, 1.0 - depth * 0.1)
                return weight

        # Default weight for padding hexagons
        return 0.3

    def aggregate_predictions(
        self,
        predictions_per_hex: Dict[str, List[Tuple[torch.Tensor, float]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate multiple predictions for each hexagon.

        Args:
            predictions_per_hex: Dict mapping hex_id -> [(prediction, weight), ...]

        Returns:
            Dict mapping hex_id -> aggregated_prediction
        """
        logger.info("\nAggregating predictions...")

        aggregated = {}

        for hex_id, predictions in tqdm(predictions_per_hex.items(), desc="Aggregating"):
            if len(predictions) == 1:
                # Single prediction, no aggregation needed
                aggregated[hex_id] = predictions[0][0]
            else:
                # Multiple predictions, aggregate based on method
                if self.config.aggregation_method == "average":
                    # Simple average
                    avg_pred = torch.stack([p for p, _ in predictions]).mean(dim=0)
                    aggregated[hex_id] = avg_pred

                elif self.config.aggregation_method == "weighted_average":
                    # Weighted average
                    total_weight = sum(w for _, w in predictions)
                    weighted_sum = sum(p * w for p, w in predictions)
                    aggregated[hex_id] = weighted_sum / total_weight

                elif self.config.aggregation_method == "median":
                    # Median (more robust to outliers)
                    stacked = torch.stack([p for p, _ in predictions])
                    aggregated[hex_id] = stacked.median(dim=0)[0]

                else:
                    raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")

        logger.info(f"Aggregated predictions for {len(aggregated)} hexagons")

        return aggregated

    def run_inference(
        self,
        output_dir: Optional[Path] = None
    ) -> Dict[int, pd.DataFrame]:
        """
        Run complete inference pipeline.

        Returns:
            Dict mapping resolution -> DataFrame of predictions
        """
        logger.info("\n" + "="*60)
        logger.info("Starting Hierarchical Cone Inference")
        logger.info("="*60)

        # Load data
        logger.info("\nLoading multi-resolution data...")
        regions_by_resolution, embeddings_by_resolution = self.load_multi_resolution_data()

        # Build graph
        logger.info("\nBuilding global graph...")
        global_edge_index, global_edge_weights, global_hex_list = self.build_global_graph(
            regions_by_resolution
        )

        # Prepare features
        logger.info("\nPreparing features...")
        global_features = self.prepare_global_features(embeddings_by_resolution, global_hex_list)

        # Create cones
        logger.info("\nCreating hierarchical cones...")
        cone_system = HierarchicalConeMaskingSystem(
            parent_resolution=self.config.parent_resolution,
            target_resolution=self.config.target_resolution,
            neighbor_rings=self.config.neighbor_rings
        )
        cones = cone_system.create_all_cones(regions_by_resolution)

        # Run inference on all cones
        predictions_per_hex = self.infer_all_cones(
            cones,
            global_features,
            global_edge_index,
            global_edge_weights,
            global_hex_list
        )

        # Aggregate predictions
        aggregated_predictions = self.aggregate_predictions(predictions_per_hex)

        # Organize by resolution and save
        logger.info("\nOrganizing results [old 2024] by resolution...")
        results_by_resolution = {}

        for res in range(self.config.parent_resolution, self.config.target_resolution + 1):
            if res not in regions_by_resolution:
                continue

            regions_gdf = regions_by_resolution[res]

            # Extract predictions for this resolution
            res_predictions = []
            res_hex_ids = []

            for hex_id in regions_gdf.index:
                if hex_id in aggregated_predictions:
                    res_predictions.append(aggregated_predictions[hex_id].cpu().numpy())
                    res_hex_ids.append(hex_id)

            if len(res_predictions) > 0:
                # Create DataFrame
                embedding_dim = len(res_predictions[0])
                embedding_cols = [f"E{i:02d}" for i in range(embedding_dim)]

                predictions_df = pd.DataFrame(
                    res_predictions,
                    index=res_hex_ids,
                    columns=embedding_cols
                )
                predictions_df.index.name = 'region_id'

                results_by_resolution[res] = predictions_df

                logger.info(f"  Res{res}: {predictions_df.shape}")

                # Save if output directory provided
                if output_dir is not None:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Save embeddings
                    output_path = output_dir / f"res{res}_predictions.parquet"
                    predictions_df.to_parquet(output_path)
                    logger.info(f"    Saved: {output_path}")

                    # Save with geometry
                    predictions_gdf = regions_gdf.loc[res_hex_ids].copy()
                    for col in embedding_cols:
                        predictions_gdf[col] = predictions_df[col]

                    output_path_geo = output_dir / f"res{res}_predictions_with_geometry.parquet"
                    predictions_gdf.to_parquet(output_path_geo)
                    logger.info(f"    Saved with geometry: {output_path_geo}")

        logger.info("\n" + "="*60)
        logger.info("Inference Complete!")
        logger.info("="*60)

        return results_by_resolution


def example_usage():
    """Example of how to use the inference system."""

    # Load trained model
    from ..models.cone_batching_unet import ConeBatchingUNet, ConeBatchingUNetConfig

    model_config = ConeBatchingUNetConfig(
        input_dim=64,
        output_dim=64,
        dropout=0.1,
        conv_type="gcn"
    )

    model = ConeBatchingUNet(model_config)

    # Load checkpoint
    paths = StudyAreaPaths("netherlands")
    checkpoint_path = paths.checkpoints("lattice_unet_cones") / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create inference config
    inference_config = InferenceConfig(
        parent_resolution=5,
        target_resolution=10,
        neighbor_rings=5,
        batch_size=64,
        aggregation_method="weighted_average",
        edge_padding_rings=1,
        device="auto"
    )

    # Initialize inference engine
    inference_engine = HierarchicalConeInference(
        model=model,
        config=inference_config,
        study_area="netherlands"
    )

    # Run inference
    output_dir = paths.stage2("lattice_unet_cones") / "inference"
    results = inference_engine.run_inference(output_dir=output_dir)

    logger.info("\nInference results [old 2024]:")
    for res, df in results.items():
        logger.info(f"  Resolution {res}: {df.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()