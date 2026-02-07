#!/usr/bin/env python
"""
Inference Script for Cone-Based LatticeUNet
============================================

Applies trained model to all cones and aggregates predictions with weighted averaging.

Key Strategy:
- Process each cone one-at-a-time (memory efficient)
- Track predictions for each hexagon across multiple cones
- Weight by distance from cone center (center = high weight, periphery = low weight)
- Final embedding = weighted average across all cones containing that hexagon

Usage:
    python scripts/netherlands/infer_cone_alphaearth.py \
        --checkpoint data/study_areas/netherlands/results/cone_alphaearth/checkpoints/best.pth
"""

import sys
from pathlib import Path
import logging
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import h3  # For grid_distance calculation

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from stage2_fusion.data.cone_dataset import ConeDataset
from stage2_fusion.models.lattice_unet import LatticeUNet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConeInferenceAggregator:
    """Weighted aggregation of cone-based predictions."""

    def __init__(
        self,
        checkpoint_path: str,
        study_area: str = "netherlands",
        parent_resolution: int = 5,
        target_resolution: int = 10,
        neighbor_rings: int = 5,
        device: str = "auto",
        output_dir: str = None
    ):
        """
        Initialize inference aggregator.

        Args:
            checkpoint_path: Path to trained model checkpoint
            study_area: Name of study area
            parent_resolution: Coarse resolution (cone roots)
            target_resolution: Fine resolution (observations)
            neighbor_rings: k-hop neighborhood size
            device: Inference device
            output_dir: Output directory for embeddings
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.study_area = study_area
        self.parent_resolution = parent_resolution
        self.target_resolution = target_resolution
        self.neighbor_rings = neighbor_rings

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("=" * 80)
        logger.info("Cone-Based Inference with Weighted Aggregation")
        logger.info("=" * 80)
        logger.info(f"Study Area: {study_area}")
        logger.info(f"Checkpoint: {self.checkpoint_path.name}")
        logger.info(f"Device: {self.device}")

        # Setup output directory
        if output_dir is None:
            output_dir = f"data/study_areas/{study_area}/results/cone_alphaearth/embeddings"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        logger.info("\nLoading ConeDataset...")
        self.dataset = ConeDataset(
            study_area=study_area,
            parent_resolution=parent_resolution,
            target_resolution=target_resolution,
            neighbor_rings=neighbor_rings
        )
        logger.info(f"Dataset: {len(self.dataset)} cones")

        # Load model
        logger.info("\nLoading trained model...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Get model config from checkpoint
        embedding_dim = checkpoint['model_state_dict']['input_proj.weight'].shape[1]
        logger.info(f"Embedding dimension: {embedding_dim}")

        # Recreate model (use same config as training)
        from stage2_fusion.models.lattice_unet import LatticeUNetConfig

        config = LatticeUNetConfig(
            input_dim=embedding_dim,
            hidden_dim=128,  # Should match training
            output_dim=embedding_dim,
            num_layers=4,  # Should match training
            dropout=0.1,
            conv_type="gcn",
            use_batch_norm=False,
            use_graph_norm=True,
            use_skip_connections=True,
            activation="gelu"
        )

        self.model = LatticeUNet(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
        logger.info(f"Best loss: {checkpoint['loss']:.6f}")

        # Initialize aggregation storage
        # Dict[h3_index] -> List[(embedding, weight)]
        self.predictions = defaultdict(list)

    def compute_weight(self, hexagon: str, cone_center: str) -> float:
        """
        Compute weight based on distance from cone center.

        Weight decays linearly with grid distance:
        - center (distance=0) → weight=1.0
        - periphery (distance=neighbor_rings) → weight=0.0

        Args:
            hexagon: H3 index of hexagon
            cone_center: H3 index of cone center

        Returns:
            Weight in [0.0, 1.0]
        """
        # Calculate grid distance at parent resolution
        # Convert both to parent resolution for consistent distance metric
        hex_parent = h3.cell_to_parent(hexagon, self.parent_resolution)

        try:
            distance = h3.grid_distance(hex_parent, cone_center)
        except:
            # If grid_distance fails (rare edge case), use 0 distance
            distance = 0

        # Linear decay
        weight = max(0.0, 1.0 - distance / self.neighbor_rings)

        return weight

    @torch.no_grad()
    def process_cone(self, cone_idx: int):
        """
        Process single cone and store weighted predictions.

        Args:
            cone_idx: Index of cone in dataset
        """
        # Get cone data
        batch = self.dataset[cone_idx]

        # Extract data
        features = batch['features_res10'].to(self.device)
        edge_index = batch['spatial_edges'][self.target_resolution][0].to(self.device)
        edge_weights = batch['spatial_edges'][self.target_resolution][1].to(self.device)
        cone_id = batch['cone_id']  # Center res5 hexagon

        # Get hex ID mapping
        hex_to_local_idx = batch['hex_to_local_idx_by_res'][self.target_resolution]
        local_idx_to_hex = batch['local_idx_to_hex_by_res'][self.target_resolution]

        # Forward pass
        outputs = self.model(
            features,
            edge_index,
            edge_weights,
            batch=None
        )

        # Extract embeddings
        if isinstance(outputs, dict):
            embeddings = outputs['embeddings']
        else:
            embeddings = outputs

        embeddings_np = embeddings.cpu().numpy()

        # Store predictions with weights
        for local_idx, hex_id in local_idx_to_hex.items():
            embedding = embeddings_np[local_idx]

            # Compute weight based on distance from cone center
            weight = self.compute_weight(hex_id, cone_id)

            # Store (embedding, weight) tuple
            self.predictions[hex_id].append((embedding, weight))

    def run_inference(self):
        """Process all cones and collect predictions."""
        logger.info("\n" + "=" * 80)
        logger.info("Processing Cones")
        logger.info("=" * 80)

        for cone_idx in tqdm(range(len(self.dataset)), desc="Inference"):
            self.process_cone(cone_idx)

        logger.info(f"\nCollected predictions for {len(self.predictions):,} unique hexagons")

        # Check coverage
        avg_cones_per_hex = np.mean([len(preds) for preds in self.predictions.values()])
        logger.info(f"Average cones per hexagon: {avg_cones_per_hex:.1f}")

    def aggregate_predictions(self) -> pd.DataFrame:
        """
        Aggregate predictions across cones using weighted averaging.

        Returns:
            DataFrame with columns: h3_index (index), E00, E01, ..., E63
        """
        logger.info("\n" + "=" * 80)
        logger.info("Aggregating Predictions")
        logger.info("=" * 80)

        aggregated_embeddings = {}

        for hex_id, predictions_list in tqdm(self.predictions.items(), desc="Aggregating"):
            # Extract embeddings and weights
            embeddings = np.array([pred[0] for pred in predictions_list])  # [N, D]
            weights = np.array([pred[1] for pred in predictions_list])  # [N]

            # Weighted average
            if weights.sum() > 0:
                weighted_embedding = np.average(embeddings, axis=0, weights=weights)
            else:
                # Fallback: simple average (shouldn't happen)
                weighted_embedding = np.mean(embeddings, axis=0)

            aggregated_embeddings[hex_id] = weighted_embedding

        # Convert to DataFrame
        # Use A00-A63 format to match AlphaEarth convention (for visualization compatibility)
        embedding_dim = len(list(aggregated_embeddings.values())[0])
        embedding_cols = [f"A{i:02d}" for i in range(embedding_dim)]

        embeddings_df = pd.DataFrame.from_dict(
            aggregated_embeddings,
            orient='index',
            columns=embedding_cols
        )
        embeddings_df.index.name = 'h3_index'

        logger.info(f"Aggregated embeddings: {embeddings_df.shape}")

        return embeddings_df

    def save_embeddings(self, embeddings_df: pd.DataFrame):
        """
        Save embeddings to parquet.

        Args:
            embeddings_df: DataFrame with h3_index and embedding columns
        """
        logger.info("\n" + "=" * 80)
        logger.info("Saving Embeddings")
        logger.info("=" * 80)

        # Save without geometry
        output_path = self.output_dir / f"{self.study_area}_res{self.target_resolution}_embeddings.parquet"
        embeddings_df.to_parquet(output_path)
        logger.info(f"Saved: {output_path}")

        # Save with geometry for visualization
        logger.info("Adding geometry for visualization...")

        # Load region geometries
        regions_path = (
            Path(f"data/study_areas/{self.study_area}/regions_gdf") /
            f"{self.study_area}_res{self.target_resolution}.parquet"
        )

        if regions_path.exists():
            regions_gdf = gpd.read_parquet(regions_path)

            # Filter to common hexagons
            common_hexes = embeddings_df.index.intersection(regions_gdf.index)
            logger.info(f"Common hexagons with geometry: {len(common_hexes):,}")

            # Create GeoDataFrame
            embeddings_gdf = regions_gdf.loc[common_hexes].copy()
            for col in embeddings_df.columns:
                embeddings_gdf[col] = embeddings_df.loc[common_hexes, col]

            # Save with geometry
            output_path_geo = self.output_dir / f"{self.study_area}_res{self.target_resolution}_embeddings_with_geometry.parquet"
            embeddings_gdf.to_parquet(output_path_geo)
            logger.info(f"Saved with geometry: {output_path_geo}")
        else:
            logger.warning(f"Regions file not found: {regions_path}")
            logger.warning("Skipping geometry addition")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Cone-Based Inference with Weighted Aggregation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--study-area', type=str, default='netherlands',
                        help='Study area name (default: netherlands)')
    parser.add_argument('--parent-res', type=int, default=5,
                        help='Parent resolution (default: 5)')
    parser.add_argument('--target-res', type=int, default=10,
                        help='Target resolution (default: 10)')
    parser.add_argument('--neighbor-rings', type=int, default=5,
                        help='k-hop neighborhood size (default: 5)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu, default: auto)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto)')

    args = parser.parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Initialize aggregator
    aggregator = ConeInferenceAggregator(
        checkpoint_path=str(checkpoint_path),
        study_area=args.study_area,
        parent_resolution=args.parent_res,
        target_resolution=args.target_res,
        neighbor_rings=args.neighbor_rings,
        device=args.device,
        output_dir=args.output_dir
    )

    # Run inference
    aggregator.run_inference()

    # Aggregate predictions
    embeddings_df = aggregator.aggregate_predictions()

    # Save embeddings
    aggregator.save_embeddings(embeddings_df)

    logger.info("\n" + "=" * 80)
    logger.info("Inference Complete!")
    logger.info(f"Embeddings saved to: {aggregator.output_dir}")
    logger.info("=" * 80)
    logger.info("\nNext Steps:")
    logger.info("1. Visualize clusters:")
    logger.info(f"   python scripts/visualization/visualize_res10_clusters_fast.py \\")
    logger.info(f"     --study-area {args.study_area} \\")
    logger.info(f"     --clusters 8,12,16")


if __name__ == "__main__":
    main()
