#!/usr/bin/env python
"""
⚠️ LEGACY TRAINING SCRIPT ⚠️

Train LatticeUNet on Netherlands AlphaEarth Res10 Embeddings (NON-CONE VERSION)
===============================================================================

This is the LEGACY training script without hierarchical cone optimization.

USE INSTEAD: train_lattice_unet_res10_cones.py
  - Hierarchical cone system (memory-efficient for 6M+ hexagons)
  - Phase 7 geometric optimization (H3 spatial sorting, 5-15% faster)
  - ConeBatcher for parallel processing
  - Actively maintained

This legacy script loads entire graph into memory (impractical for full res10).
Kept for reference and comparison purposes only.

Last Active: Pre-October 2025
Superseded By: train_lattice_unet_res10_cones.py (cone-based training)

---

Original Description:
Trains a Graph Convolutional UNet on hexagonal lattice with 5-ring connectivity.
Uses SRAI for all spatial operations and stores results [old 2024] in data folder.

Usage:
    python scripts/netherlands/train_lattice_unet_res10.py  [LEGACY - use cone version]

Outputs saved to: data/study_areas/netherlands/results [old 2024]/lattice_unet_res10/
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from stage2_fusion.models.lattice_unet import LatticeUNet, LatticeUNetConfig
from stage2_fusion.graphs.hexagonal_graph_constructor import HexagonalLatticeConstructor
from stage2_fusion.data.study_area_loader import StudyAreaLoader
from stage2_fusion.data.hierarchical_cone_masking import (
    HierarchicalConeMaskingSystem,
    ConeBatcher,
    HierarchicalCone
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class LatticeUNetTrainer:
    """Trainer for LatticeUNet on Netherlands AlphaEarth embeddings."""

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
        Initialize trainer.

        Args:
            study_area: Name of study area
            parent_resolution: Coarse resolution for cone roots (5)
            target_resolution: Fine resolution for processing (10 for ~66m hexagons)
            neighbor_rings: Number of hexagonal rings at parent resolution (5 = 90 connections)
            hidden_dim: Hidden dimension size
            num_layers: Number of encoder/decoder layers
            learning_rate: Initial learning rate
            epochs: Number of training epochs
            cone_batch_size: Number of cones to process in parallel
            device: Device for training ('auto', 'cuda', 'cpu')
        """
        self.study_area = study_area
        self.resolution = resolution
        self.neighbor_rings = neighbor_rings
        self.batch_size = batch_size
        self.epochs = epochs

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing LatticeUNet Trainer:")
        logger.info(f"  Study Area: {study_area}")
        logger.info(f"  Resolution: {resolution}")
        logger.info(f"  Neighbor Rings: {neighbor_rings}")
        logger.info(f"  Device: {self.device}")

        # Setup paths
        self.output_dir = Path(f"data/study_areas/{study_area}/results [old 2024]/lattice_unet_res{resolution}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "training_logs").mkdir(exist_ok=True)

        # Initialize data loader
        self.loader = StudyAreaLoader(study_area=study_area)

        # Load data
        logger.info("Loading data...")
        self.embeddings_df, self.regions_gdf = self._load_data()

        # Build graph
        logger.info("Building hexagonal lattice graph...")
        self.edge_index, self.edge_weights = self._build_graph()

        # Initialize model
        logger.info("Initializing model...")
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

    def _load_data(self) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """Load AlphaEarth embeddings and regions."""
        # Load embeddings
        embeddings_path = f"data/study_areas/{self.study_area}/embeddings/alphaearth/{self.study_area}_res{self.resolution}_2022.parquet"
        embeddings_df = pd.read_parquet(embeddings_path)

        logger.info(f"Loaded embeddings: {embeddings_df.shape}")
        logger.info(f"Columns: {list(embeddings_df.columns)[:5]}...")

        # Load regions
        regions_gdf = self.loader.load_regions(self.resolution, with_geometry=True)
        logger.info(f"Loaded regions: {regions_gdf.shape}")
        logger.info(f"Index name: {regions_gdf.index.name}")

        # Align indices: embeddings have 'h3_index' column, regions use 'region_id' index
        if 'h3_index' in embeddings_df.columns:
            embeddings_df = embeddings_df.set_index('h3_index')
            embeddings_df.index.name = 'region_id'

        # Filter to common hexagons
        common_indices = embeddings_df.index.intersection(regions_gdf.index)
        logger.info(f"Common hexagons: {len(common_indices)}")

        embeddings_df = embeddings_df.loc[common_indices]
        regions_gdf = regions_gdf.loc[common_indices]

        # Extract only embedding columns (A00-A63)
        embedding_cols = [col for col in embeddings_df.columns if col.startswith('A')]
        embeddings_df = embeddings_df[embedding_cols]

        logger.info(f"Final embeddings shape: {embeddings_df.shape}")
        logger.info(f"Embedding columns: {embedding_cols[:5]}...{embedding_cols[-2:]}")

        return embeddings_df, regions_gdf

    def _build_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build hexagonal lattice graph with 5 rings."""
        # Initialize graph constructor
        constructor = HexagonalLatticeConstructor(
            device=str(self.device),
            neighbor_rings=self.neighbor_rings,
            edge_weight=1.0,
            include_self_loops=False
        )

        # Build graph for this resolution
        edge_features = constructor._construct_hexagonal_lattice(
            self.regions_gdf,
            self.resolution,
            mode=f"res{self.resolution}"
        )

        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_features.edge_index, dtype=torch.long).to(self.device)
        edge_weights = torch.tensor(edge_features.edge_weights, dtype=torch.float32).to(self.device)

        logger.info(f"Graph constructed:")
        logger.info(f"  Nodes: {len(self.regions_gdf)}")
        logger.info(f"  Edges: {edge_index.shape[1]}")
        logger.info(f"  Avg degree: {edge_index.shape[1] / len(self.regions_gdf):.1f}")

        return edge_index, edge_weights

    def _create_model(self, hidden_dim: int, num_layers: int) -> LatticeUNet:
        """Create LatticeUNet model."""
        config = LatticeUNetConfig(
            input_dim=len(self.embeddings_df.columns),  # 64 for AlphaEarth
            hidden_dim=hidden_dim,
            output_dim=len(self.embeddings_df.columns),  # Reconstruct same dimension
            num_layers=num_layers,
            dropout=0.1,
            conv_type="gcn",
            use_batch_norm=False,
            use_graph_norm=True,
            use_skip_connections=True,
            activation="gelu",
            reconstruction_weight=1.0,
            consistency_weight=0.5
        )

        model = LatticeUNet(config)

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {num_params:,} parameters")

        return model

    def _save_config(self):
        """Save training configuration."""
        config = {
            'study_area': self.study_area,
            'resolution': self.resolution,
            'neighbor_rings': self.neighbor_rings,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device),
            'model_config': {
                'input_dim': len(self.embeddings_df.columns),
                'hidden_dim': self.model.config.hidden_dim,
                'output_dim': self.model.config.output_dim,
                'num_layers': self.model.config.num_layers,
                'conv_type': self.model.config.conv_type
            },
            'data': {
                'num_hexagons': len(self.embeddings_df),
                'num_edges': self.edge_index.shape[1],
                'embedding_dim': len(self.embeddings_df.columns)
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(self.output_dir / "training_logs" / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Convert embeddings to tensor
        features = torch.tensor(
            self.embeddings_df.values,
            dtype=torch.float32
        ).to(self.device)

        # Simple full-batch training (for now - can add spatial batching later)
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(
            features,
            self.edge_index,
            self.edge_weights,
            batch=None
        )

        # Reconstruction loss
        loss = nn.functional.mse_loss(output, features)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update weights
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validate(self) -> float:
        """Validate model."""
        self.model.eval()

        features = torch.tensor(
            self.embeddings_df.values,
            dtype=torch.float32
        ).to(self.device)

        # Forward pass
        output = self.model(
            features,
            self.edge_index,
            self.edge_weights,
            batch=None
        )

        # Validation loss
        loss = nn.functional.mse_loss(output, features)

        return loss.item()

    def train(self):
        """Run full training loop."""
        logger.info("\n" + "="*60)
        logger.info("Starting Training")
        logger.info("="*60)

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
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.6f}"
            )

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                patience_counter = 0
                logger.info(f"  → New best model! Val loss: {val_loss:.6f}")
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
        self.extract_embeddings()

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

    @torch.no_grad()
    def extract_embeddings(self):
        """Extract final embeddings from trained model."""
        logger.info("\nExtracting embeddings...")
        self.model.eval()

        features = torch.tensor(
            self.embeddings_df.values,
            dtype=torch.float32
        ).to(self.device)

        # Forward pass
        output = self.model(
            features,
            self.edge_index,
            self.edge_weights,
            batch=None
        )

        # Convert to numpy
        embeddings_np = output.cpu().numpy()

        # Create DataFrame
        embedding_cols = [f"E{i:02d}" for i in range(embeddings_np.shape[1])]
        embeddings_df = pd.DataFrame(
            embeddings_np,
            index=self.embeddings_df.index,
            columns=embedding_cols
        )

        # Save
        output_path = self.output_dir / "embeddings" / "final_embeddings.parquet"
        embeddings_df.to_parquet(output_path)
        logger.info(f"  Saved embeddings: {output_path}")
        logger.info(f"  Shape: {embeddings_df.shape}")

        # Also save with geometry for visualization
        embeddings_gdf = self.regions_gdf.copy()
        for col in embedding_cols:
            embeddings_gdf[col] = embeddings_df[col]

        output_path_geo = self.output_dir / "embeddings" / "final_embeddings_with_geometry.parquet"
        embeddings_gdf.to_parquet(output_path_geo)
        logger.info(f"  Saved embeddings with geometry: {output_path_geo}")


def main():
    """Main training function."""
    logger.info("="*60)
    logger.info("LatticeUNet Training - Netherlands Res10")
    logger.info("="*60)

    # Initialize trainer
    trainer = LatticeUNetTrainer(
        study_area="netherlands",
        resolution=10,
        neighbor_rings=5,  # 90 connections per hexagon
        hidden_dim=128,
        num_layers=4,
        learning_rate=0.001,
        epochs=50,
        batch_size=5000,
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