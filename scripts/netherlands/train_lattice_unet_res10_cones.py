#!/usr/bin/env python
"""
Train LatticeUNet with Hierarchical Cone Masking - Netherlands Multi-Resolution
================================================================================

Uses hierarchical computational cones for memory-efficient multi-resolution training.
Each cone spans from res5 (with 5-ring neighborhood) down through descendants to res10.

Cones are batched and processed in parallel for efficient GPU utilization.

Usage:
    python scripts/netherlands/train_lattice_unet_res10_cones.py

Outputs saved to: data/study_areas/netherlands/results/lattice_unet_cones/
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project modules directly - bypasses problematic __init__.py imports
from stage2_fusion.models.cone_unet import ConeLatticeUNet, ConeUNetConfig
from stage2_fusion.graphs.hexagonal_graph_constructor import HexagonalLatticeConstructor
from stage2_fusion.data.study_area_loader import StudyAreaLoader
from stage2_fusion.data.hierarchical_cone_masking import (
    HierarchicalConeMaskingSystem,
    LazyConeBatcher,
    HierarchicalCone
)
import h3  # For hierarchical mappings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_cones.log')
    ]
)
logger = logging.getLogger(__name__)


class HierarchicalConeTrainer:
    """Trainer using hierarchical cone masking for multi-resolution learning."""

    def __init__(
        self,
        study_area: str = "netherlands",
        parent_resolution: int = 5,
        target_resolution: int = 10,
        neighbor_rings: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 4,
        learning_rate: float = 0.001,
        epochs: int = 50,
        cone_batch_size: int = 32,
        device: str = "auto"
    ):
        """
        Initialize hierarchical cone trainer.

        Args:
            study_area: Name of study area
            parent_resolution: Coarse resolution for cone roots (5)
            target_resolution: Fine resolution for leaf nodes (10)
            neighbor_rings: Number of rings at parent resolution (5)
            hidden_dim: Hidden dimension size
            num_layers: Number of UNet layers
            learning_rate: Initial learning rate
            epochs: Number of training epochs
            cone_batch_size: Number of cones per batch
            device: Device for training
        """
        self.study_area = study_area
        self.parent_resolution = parent_resolution
        self.target_resolution = target_resolution
        self.neighbor_rings = neighbor_rings
        self.cone_batch_size = cone_batch_size
        self.epochs = epochs

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing Hierarchical Cone Trainer:")
        logger.info(f"  Study Area: {study_area}")
        logger.info(f"  Parent Resolution: {parent_resolution}")
        logger.info(f"  Target Resolution: {target_resolution}")
        logger.info(f"  Neighbor Rings: {neighbor_rings}")
        logger.info(f"  Cone Batch Size: {cone_batch_size}")
        logger.info(f"  Device: {self.device}")

        # Setup paths
        self.output_dir = Path(f"data/study_areas/{study_area}/results/lattice_unet_cones")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "training_logs").mkdir(exist_ok=True)

        # Load multi-resolution data
        logger.info("\nLoading multi-resolution data...")
        self.regions_by_resolution = self._load_all_regions()
        self.embeddings_by_resolution = self._load_all_embeddings()

        # Initialize cone system and build lookup tables (cached)
        logger.info("\nInitializing cone system with lookup tables...")
        self.cone_system = HierarchicalConeMaskingSystem(
            parent_resolution=parent_resolution,
            target_resolution=target_resolution,
            neighbor_rings=neighbor_rings
        )

        # ARCHITECTURE: On-the-Fly Cone Creation
        # - Lookup tables (parent->children) are built ONCE and cached (~30 sec first time)
        # - During training: create cones ON-THE-FLY using simple dict lookups (~1ms per cone)
        # - No need to pre-compute and cache all 408 cones
        # - Each cone is an INDEPENDENT computational shard
        # - U-Net processes ONE CONE at a time (down -> bottleneck -> up)

        # Build/load lookup tables ONCE
        cones_dir = f"data/study_areas/{study_area}/cones"
        os.makedirs(cones_dir, exist_ok=True)

        logger.info("Building/loading parent->children lookup table...")
        lookup_cache_path = f"{cones_dir}/parent_to_children_res{parent_resolution}_to_{target_resolution}.pkl"
        self.cone_system.parent_to_children = self.cone_system._build_parent_lookup(
            self.regions_by_resolution,
            cache_path=lookup_cache_path
        )

        # Get list of parent hexagons and apply H3 spatial sorting
        # Phase 7 Optimization: H3 indices encode spatial proximity (space-filling curve)
        # Sorting groups spatially-adjacent cones together for cache efficiency
        parent_regions = self.regions_by_resolution[parent_resolution]
        self.parent_hexagons = sorted(parent_regions.index)  # H3 spatial sorting
        logger.info(f"Processing {len(self.parent_hexagons)} cones with H3 spatial ordering...")

        # Build and cache all cones (or load from cache if exists)
        # TRUE LAZY LOADING: Individual cone files (cone_{hex}.pkl) for minimal memory usage!
        cones_cache_dir = f"{cones_dir}/cone_cache_res{parent_resolution}_to_{target_resolution}"
        logger.info("\nCaching cones to disk (individual files for true lazy loading)...")
        self.cone_system.cache_all_cones(
            self.regions_by_resolution,
            cache_dir=cones_cache_dir
        )

        # Create lazy batcher (loads individual cone files on-demand)
        # MEMORY SAVINGS: ~60 GB (all cones) -> ~4.5 GB (32 cones per batch only!)
        self.cone_batch_size = cone_batch_size
        self.batcher = LazyConeBatcher(
            parent_hexagons=self.parent_hexagons,
            cache_dir=cones_cache_dir,
            batch_size=cone_batch_size
        )
        logger.info(f"TRUE lazy loading enabled: Only loads 32 cones at a time (~92% memory reduction!)")

        # NOTE: NO GLOBAL GRAPH - Pure cone-based approach!
        # Each cone builds its own spatial edges and hierarchical mappings on-the-fly

        # Initialize model
        logger.info("\nInitializing model...")
        self.model = self._create_model(hidden_dim, num_layers)
        self.model = self.model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs
        )

        # Training state
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        self.best_loss = float('inf')

        # Save config
        self._save_config()

    def _load_all_regions(self) -> Dict[int, gpd.GeoDataFrame]:
        """Load regions for all resolutions."""
        regions_by_res = {}
        loader = StudyAreaLoader(study_area=self.study_area)

        for res in range(self.parent_resolution, self.target_resolution + 1):
            try:
                regions_gdf = loader.load_regions(res, with_geometry=True)
                regions_by_res[res] = regions_gdf
                logger.info(f"  Loaded res{res}: {len(regions_gdf)} hexagons")
            except Exception as e:
                logger.warning(f"  Could not load res{res}: {e}")

        return regions_by_res

    def _load_all_embeddings(self) -> Dict[int, pd.DataFrame]:
        """
        Load embeddings - ONLY res10 needed for U-Net.

        U-Net learns hierarchical representations through its architecture.
        We only need the finest resolution as input - the hierarchy is
        captured in the cone structure (parent-child relationships).
        """
        embeddings_by_res = {}

        # ONLY load target resolution (res10) - U-Net handles hierarchy
        res = self.target_resolution
        embeddings_path = (
            f"data/study_areas/{self.study_area}/embeddings/alphaearth/"
            f"{self.study_area}_res{res}_2022.parquet"
        )

        if Path(embeddings_path).exists():
            embeddings_df = pd.read_parquet(embeddings_path)

            # Align with regions
            if 'h3_index' in embeddings_df.columns:
                embeddings_df = embeddings_df.set_index('h3_index')
                embeddings_df.index.name = 'region_id'

            # Extract embedding columns only
            embedding_cols = [col for col in embeddings_df.columns if col.startswith('A')]
            embeddings_df = embeddings_df[embedding_cols]

            # Filter to common hexagons with regions
            if res in self.regions_by_resolution:
                common_indices = embeddings_df.index.intersection(
                    self.regions_by_resolution[res].index
                )
                embeddings_df = embeddings_df.loc[common_indices]

                embeddings_by_res[res] = embeddings_df
                logger.info(f"  Loaded res{res} embeddings: {embeddings_df.shape}")
            else:
                logger.warning(f"  No embeddings found for res{res}")

        return embeddings_by_res

    def _build_cone_spatial_edges(self, cone: HierarchicalCone) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Build spatial edges PER RESOLUTION for hexagons in THIS cone only.

        Returns:
            Dict[res] -> (edge_index, edge_weight) for hexagons at each resolution in this cone
        """
        spatial_edges = {}

        for res in range(5, 11):
            # Get hexagons at this resolution WITHIN THIS CONE
            if res == 5:
                hexes = {cone.parent_hex} | cone.parent_neighbors  # ~91 hexes
            else:
                hexes = cone.descendants_by_resolution.get(res, set())

            if not hexes:
                continue

            # Get ONLY the regions for this cone's hexagons
            regions_gdf = self.regions_by_resolution[res]
            cone_regions_gdf = regions_gdf.loc[list(hexes)]

            # Build lattice edges for JUST these hexagons
            constructor = HexagonalLatticeConstructor(
                device=str(self.device),
                neighbor_rings=self.neighbor_rings,
                edge_weight=1.0,
                include_self_loops=False
            )

            edge_features = constructor._construct_hexagonal_lattice(
                cone_regions_gdf, res, mode=f"res{res}"
            )

            # These are LOCAL indices within the cone
            spatial_edges[res] = (
                edge_features.edge_index.to(self.device),
                edge_features.edge_weights.to(self.device)
            )

        return spatial_edges

    def _build_cone_hierarchical_mappings(self, cone: HierarchicalCone) -> Dict[int, Tuple[torch.Tensor, int]]:
        """
        Build child->parent mappings for hierarchical aggregation WITHIN THIS CONE.

        Returns:
            Dict[res] -> (child_to_parent_idx, num_parents) for aggregation
        """
        hierarchical_mappings = {}

        for child_res in range(6, 11):
            parent_res = child_res - 1

            # Get parent and child hexagons WITHIN THIS CONE
            if parent_res == 5:
                parent_hexes = sorted({cone.parent_hex} | cone.parent_neighbors)
            else:
                parent_hexes = sorted(cone.descendants_by_resolution.get(parent_res, set()))

            child_hexes = sorted(cone.descendants_by_resolution.get(child_res, set()))

            if not child_hexes or not parent_hexes:
                continue

            # Build mapping: child_idx -> parent_idx (within cone's local indexing)
            parent_hex_to_idx = {h: i for i, h in enumerate(parent_hexes)}

            child_to_parent_idx = []
            for child_hex in child_hexes:
                parent_hex = h3.cell_to_parent(child_hex, parent_res)
                if parent_hex in parent_hex_to_idx:
                    child_to_parent_idx.append(parent_hex_to_idx[parent_hex])
                else:
                    # Orphan - parent outside cone (boundary effect)
                    child_to_parent_idx.append(-1)

            child_to_parent_idx = torch.tensor(child_to_parent_idx, dtype=torch.long)
            valid_mask = child_to_parent_idx >= 0

            hierarchical_mappings[child_res] = (
                child_to_parent_idx[valid_mask].to(self.device),
                len(parent_hexes)
            )

        return hierarchical_mappings

    def _extract_cone_features(self, cone: HierarchicalCone) -> torch.Tensor:
        """Extract res10 features for hexagons in THIS cone only."""
        res10_hexes = sorted(cone.descendants_by_resolution.get(10, set()))

        if not res10_hexes:
            raise ValueError(f"Cone {cone.cone_id} has no res10 hexagons")

        embeddings_df = self.embeddings_by_resolution[10]

        # Extract in consistent order
        features_list = []
        for hex_id in res10_hexes:
            if hex_id in embeddings_df.index:
                features_list.append(embeddings_df.loc[hex_id].values)
            else:
                # Missing data - fill with zeros
                features_list.append(np.zeros(len(embeddings_df.columns)))

        return torch.tensor(features_list, dtype=torch.float32).to(self.device)

    def _create_model(self, hidden_dim: int, num_layers: int) -> ConeLatticeUNet:
        """Create ConeLatticeUNet model for cone-based processing."""
        # Determine embedding dimension (should be same across resolutions)
        embedding_dim = len(list(self.embeddings_by_resolution.values())[0].columns)

        config = ConeUNetConfig(
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            hidden_dims={
                10: 64,   # res10: finest resolution
                9: 128,   # res9
                8: 128,   # res8
                7: 256,   # res7
                6: 256,   # res6
                5: 512    # res5: coarsest (bottleneck)
            },
            lateral_conv_layers=2,  # 2 GCN hops per resolution
            conv_type="gcn",
            use_graph_norm=True,
            use_skip_connections=True,
            activation="gelu",
            dropout=0.1
        )

        model = ConeLatticeUNet(config)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"ConeLatticeUNet created with {num_params:,} parameters")
        logger.info(f"  Per-resolution processing: 2 GCN hops × 6 resolutions")

        return model

    def _save_config(self):
        """Save training configuration."""
        config = {
            'study_area': self.study_area,
            'parent_resolution': self.parent_resolution,
            'target_resolution': self.target_resolution,
            'neighbor_rings': self.neighbor_rings,
            'cone_batch_size': self.cone_batch_size,
            'epochs': self.epochs,
            'device': str(self.device),
            'num_cones': len(self.cones),
            'approach': 'cone-based (no global graph)',
            'model_type': 'ConeLatticeUNet',
            'model_config': {
                'input_dim': self.model.config.input_dim,
                'output_dim': self.model.config.output_dim,
                'hidden_dims': self.model.config.hidden_dims,
                'lateral_conv_layers': self.model.config.lateral_conv_layers,
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(self.output_dir / "training_logs" / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def train_epoch(self) -> float:
        """
        Train one epoch - pure cone-based processing.
        Each cone is completely independent with its own spatial edges and hierarchical mappings.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.batcher)

        for batch_idx, cone_batch in enumerate(self.batcher):
            batch_loss = 0.0

            # Process each cone in batch
            for cone in cone_batch:
                self.optimizer.zero_grad()

                # 1. Build spatial edges for THIS cone (per resolution)
                spatial_edges = self._build_cone_spatial_edges(cone)

                # 2. Build hierarchical mappings for THIS cone
                hierarchical_mappings = self._build_cone_hierarchical_mappings(cone)

                # 3. Extract res10 features for THIS cone
                features_res10 = self._extract_cone_features(cone)

                # 4. Forward pass through ConeLatticeUNet
                output = self.model(
                    features_res10,          # [N_res10_in_cone, feature_dim]
                    spatial_edges,           # Dict[res] -> (edge_index, edge_weight)
                    hierarchical_mappings,   # Dict[res] -> (child->parent_idx, num_parents)
                    batch=None
                )

                # 5. Reconstruction loss
                loss = F.mse_loss(output['reconstruction'], features_res10)

                # 6. Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_loss += loss.item()

            # Average loss for this batch
            avg_batch_loss = batch_loss / len(cone_batch) if len(cone_batch) > 0 else 0.0
            total_loss += avg_batch_loss

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"    Batch {batch_idx+1}/{num_batches} | Loss: {avg_batch_loss:.6f}")

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def validate(self) -> float:
        """Validate model on random sample of cones - pure cone-based."""
        self.model.eval()

        # Sample 10% of cones for validation
        import random
        val_cones = random.sample(self.cones, max(1, len(self.cones) // 10))

        total_loss = 0.0

        for cone in val_cones:
            # Build cone-specific graphs and features
            spatial_edges = self._build_cone_spatial_edges(cone)
            hierarchical_mappings = self._build_cone_hierarchical_mappings(cone)
            features_res10 = self._extract_cone_features(cone)

            # Forward pass
            output = self.model(
                features_res10,
                spatial_edges,
                hierarchical_mappings,
                batch=None
            )

            loss = F.mse_loss(output['reconstruction'], features_res10)
            total_loss += loss.item()

        return total_loss / len(val_cones) if len(val_cones) > 0 else 0.0

    @torch.no_grad()
    def inference_with_weighted_averaging(self) -> pd.DataFrame:
        """
        Inference: Process all cones, aggregate predictions with weighted averaging.

        CONE OVERLAP:
        Each res10 hexagon appears in multiple cones (~10 on average).
        We process each cone independently, then aggregate predictions
        weighted by distance to cone center (closer = higher weight).

        Returns:
            DataFrame with final weighted-averaged predictions per hexagon
        """
        logger.info("\n" + "="*60)
        logger.info("Running Inference with Weighted Averaging")
        logger.info("="*60)
        logger.info("Processing all {} cones (lazy loaded)...".format(len(self.parent_hexagons)))

        self.model.eval()

        # Storage: hex_id -> list of (prediction, weight)
        predictions_by_hex = {}

        # Iterate through lazy batcher (loads cones on-demand)
        total_cones = len(self.parent_hexagons)
        with tqdm(total=total_cones, desc="Inference on cones") as pbar:
            for cone_batch in self.batcher:
                for cone in cone_batch:
                    # Build cone-specific graphs
                    spatial_edges = self._build_cone_spatial_edges(cone)
                    hierarchical_mappings = self._build_cone_hierarchical_mappings(cone)
                    features_res10 = self._extract_cone_features(cone)

                    # Forward pass
                    output = self.model(
                        features_res10,
                        spatial_edges,
                        hierarchical_mappings,
                        batch=None
                    )

                    predictions = output['reconstruction'].cpu().numpy()

                    # Get res10 hexagons for this cone
                    res10_hexes = sorted(cone.descendants_by_resolution.get(10, set()))

                    # Calculate weights based on distance to cone center
                    cone_center_hex = cone.parent_hex

                    for i, hex_id in enumerate(res10_hexes):
                        # Weight = 1 / (1 + distance_to_cone_center)
                        distance = h3.grid_distance(hex_id, cone_center_hex)
                        weight = 1.0 / (1.0 + distance)

                        if hex_id not in predictions_by_hex:
                            predictions_by_hex[hex_id] = []

                        predictions_by_hex[hex_id].append((predictions[i], weight))

                    # Update progress bar
                    pbar.update(1)

        # Aggregate with weighted average
        logger.info("Aggregating predictions with weighted averaging...")
        final_predictions = {}
        total_hexagons = len(predictions_by_hex)
        avg_predictions_per_hex = sum(len(preds) for preds in predictions_by_hex.values()) / total_hexagons

        logger.info(f"  Total hexagons: {total_hexagons}")
        logger.info(f"  Avg predictions per hexagon: {avg_predictions_per_hex:.1f}")

        for hex_id, pred_weight_list in predictions_by_hex.items():
            total_weight = sum(w for _, w in pred_weight_list)
            weighted_sum = sum(p * w for p, w in pred_weight_list) / total_weight
            final_predictions[hex_id] = weighted_sum

        predictions_df = pd.DataFrame.from_dict(final_predictions, orient='index')
        logger.info(f"Final predictions shape: {predictions_df.shape}")

        return predictions_df

    def train(self):
        """Run full training loop - pure cone-based processing."""
        logger.info("\n" + "="*60)
        logger.info("Starting Cone-Based Hierarchical Training (ConeLatticeUNet)")
        logger.info("="*60)
        logger.info("CONE-BASED APPROACH:")
        logger.info("  - No global graph")
        logger.info("  - Each cone builds its own spatial edges + hierarchical mappings")
        logger.info("  - Per-resolution processing: 2 GCN hops × 6 resolutions")
        logger.info("  - Cones overlap -> weighted averaging during inference")
        logger.info("="*60 + "\n")

        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)

            # Log progress
            logger.info(
                f"\nEpoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.6f}"
            )

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                patience_counter = 0
                logger.info(f"  -> New best model! Val loss: {val_loss:.6f}")
            else:
                patience_counter += 1

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Final saves
        logger.info("\nTraining complete!")
        self.save_checkpoint(epoch, is_best=False, name="final")
        self.plot_training_curves()

        # Run inference with weighted averaging
        logger.info("\nRunning final inference with weighted averaging...")
        embeddings_df = self.inference_with_weighted_averaging()

        # Save final embeddings
        output_path = self.output_dir / "embeddings" / "netherlands_res10_urban_embeddings.parquet"
        embeddings_df.to_parquet(output_path)
        logger.info(f"Saved final urban embeddings: {embeddings_df.shape}")

    def save_checkpoint(self, epoch: int, is_best: bool = False, name: Optional[str] = None):
        """Save model checkpoint."""
        if name is None:
            name = "best" if is_best else f"epoch_{epoch+1}"

        checkpoint_path = self.output_dir / "checkpoints" / f"{name}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.best_loss if is_best else self.history['val_loss'][-1],
            'history': self.history
        }, checkpoint_path)

        logger.info(f"  Saved checkpoint: {checkpoint_path}")

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Learning rate
        axes[1].plot(self.history['learning_rates'], linewidth=2, color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "training_curves.png", dpi=150)
        logger.info(f"  Saved training curves")
        plt.close()


def main():
    """Main training function."""
    logger.info("="*60)
    logger.info("Hierarchical Cone Training - Netherlands Multi-Resolution")
    logger.info("="*60)

    # Initialize trainer
    trainer = HierarchicalConeTrainer(
        study_area="netherlands",
        parent_resolution=5,
        target_resolution=10,
        neighbor_rings=5,
        hidden_dim=128,
        num_layers=4,
        learning_rate=0.001,
        epochs=1,  # Test run: 1 epoch (change to 50 for full training)
        cone_batch_size=32,
        device="auto"
    )

    # Train
    trainer.train()

    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info(f"Results saved to: {trainer.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()