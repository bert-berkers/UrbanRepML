#!/usr/bin/env python
"""
Train Cone-Based ConeLatticeUNet on Netherlands AlphaEarth Embeddings
==================================================================

Validates the hierarchical cone approach for single-modality embedding learning.

Uses:
- ConeDataset: Optimized PyTorch dataset with hierarchical validation
- ConeLatticeUNet: Production U-Net model with graph convolutions

Training Strategy:
- Process ONE cone at a time (memory efficient)
- ONE shared model (learned kernels apply to all cones)
- Simple reconstruction: MSE(output, input)

Usage:
    python scripts/netherlands/train_cone_alphaearth.py [--epochs 20] [--batch-size 1]
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from stage2_fusion.data.cone_dataset import ConeDataset, cone_collate_fn
from stage2_fusion.models.cone_unet import ConeConeLatticeUNet, ConeUNetConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConeAlphaEarthTrainer:
    """Trainer for cone-based AlphaEarth embedding learning."""

    def __init__(
        self,
        study_area: str = "netherlands",
        parent_resolution: int = 5,
        target_resolution: int = 10,
        neighbor_rings: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 4,
        learning_rate: float = 0.001,
        epochs: int = 20,
        batch_size: int = 1,
        device: str = "auto",
        output_dir: str = None
    ):
        """
        Initialize trainer.

        Args:
            study_area: Name of study area
            parent_resolution: Coarse resolution (cone roots)
            target_resolution: Fine resolution (observations)
            neighbor_rings: k-hop neighborhood size
            hidden_dim: Hidden layer dimension
            num_layers: Number of U-Net layers
            learning_rate: Initial learning rate
            epochs: Number of training epochs
            batch_size: Cones per batch (typically 1)
            device: Training device
            output_dir: Output directory for results
        """
        self.study_area = study_area
        self.parent_resolution = parent_resolution
        self.target_resolution = target_resolution
        self.neighbor_rings = neighbor_rings
        self.epochs = epochs
        self.batch_size = batch_size

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("=" * 80)
        logger.info("Cone-Based AlphaEarth Training")
        logger.info("=" * 80)
        logger.info(f"Study Area: {study_area}")
        logger.info(f"Resolutions: {parent_resolution} (roots) -> {target_resolution} (leaves)")
        logger.info(f"Neighbor Rings: {neighbor_rings}")
        logger.info(f"Batch Size: {batch_size} cones")
        logger.info(f"Device: {self.device}")
        logger.info(f"VRAM: {'24GB RTX 3090' if torch.cuda.is_available() else 'CPU'}")

        # Setup output directory
        if output_dir is None:
            output_dir = f"data/study_areas/{study_area}/results/cone_alphaearth"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for subdir in ["checkpoints", "embeddings", "plots", "logs"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)

        # Setup logging to file
        file_handler = logging.FileHandler(self.output_dir / "logs" / "training.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)

        # Create dataset
        logger.info("\nInitializing ConeDataset...")
        self.dataset = ConeDataset(
            study_area=study_area,
            parent_resolution=parent_resolution,
            target_resolution=target_resolution,
            neighbor_rings=neighbor_rings
        )

        logger.info(f"Dataset: {len(self.dataset)} cones")

        # Split into train/val (90/10)
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train: {len(self.train_dataset)} cones")
        logger.info(f"Val: {len(self.val_dataset)} cones")

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=cone_collate_fn,
            num_workers=0  # Single worker for stability
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=cone_collate_fn,
            num_workers=0
        )

        # Create model
        logger.info("\nInitializing ConeLatticeUNet...")

        # Get embedding dimension from first batch
        sample_batch = self.dataset[0]
        embedding_dim = sample_batch['features_res10'].shape[1]
        logger.info(f"Embedding dimension: {embedding_dim}")

        config = ConeUNetConfig(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,  # Reconstruct to same dimension
            num_layers=num_layers,
            dropout=0.1,
            conv_type="gcn",
            use_batch_norm=False,
            use_graph_norm=True,
            use_skip_connections=True,
            activation="gelu",
            reconstruction_weight=1.0,
            consistency_weight=0.0  # Not using consistency loss for now
        )

        self.model = ConeLatticeUNet(config).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {num_params:,}")

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
        self._save_config(config)

    def _save_config(self, model_config):
        """Save training configuration."""
        config = {
            'study_area': self.study_area,
            'parent_resolution': self.parent_resolution,
            'target_resolution': self.target_resolution,
            'neighbor_rings': self.neighbor_rings,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'num_cones': len(self.dataset),
            'train_cones': len(self.train_dataset),
            'val_cones': len(self.val_dataset),
            'model_config': {
                'input_dim': model_config.input_dim,
                'hidden_dim': model_config.hidden_dim,
                'output_dim': model_config.output_dim,
                'num_layers': model_config.num_layers,
                'conv_type': model_config.conv_type,
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(self.output_dir / "logs" / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Extract data
            features = batch['features_res10'].to(self.device)
            edge_index = batch['spatial_edges'][self.target_resolution][0].to(self.device)
            edge_weights = batch['spatial_edges'][self.target_resolution][1].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            outputs = self.model(
                features,
                edge_index,
                edge_weights,
                batch=None
            )

            # Get embeddings from output dict
            if isinstance(outputs, dict):
                embeddings = outputs['embeddings']
            else:
                embeddings = outputs

            # Reconstruction loss
            loss = nn.functional.mse_loss(embeddings, features)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{self.epochs} | "
                    f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.6f}"
                )

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # Extract data
            features = batch['features_res10'].to(self.device)
            edge_index = batch['spatial_edges'][self.target_resolution][0].to(self.device)
            edge_weights = batch['spatial_edges'][self.target_resolution][1].to(self.device)

            # Forward pass
            outputs = self.model(
                features,
                edge_index,
                edge_weights,
                batch=None
            )

            # Get embeddings
            if isinstance(outputs, dict):
                embeddings = outputs['embeddings']
            else:
                embeddings = outputs

            # Loss
            loss = nn.functional.mse_loss(embeddings, features)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(self):
        """Run full training loop."""
        logger.info("\n" + "=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)

        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate(epoch)

            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)

            # Log progress
            logger.info(
                f"\nEpoch {epoch+1}/{self.epochs} Summary: "
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
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Final saves
        logger.info("\nTraining complete!")
        self.save_checkpoint(epoch, is_best=False, name="final")
        self.plot_training_curves()

    def save_checkpoint(self, epoch: int, is_best: bool = False, name: str = None):
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

        logger.info(f"  Saved checkpoint: {checkpoint_path.name}")

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
    parser = argparse.ArgumentParser(description='Train Cone-Based AlphaEarth Embeddings')
    parser.add_argument('--study-area', type=str, default='netherlands',
                        help='Study area name (default: netherlands)')
    parser.add_argument('--parent-res', type=int, default=5,
                        help='Parent resolution (default: 5)')
    parser.add_argument('--target-res', type=int, default=10,
                        help='Target resolution (default: 10)')
    parser.add_argument('--neighbor-rings', type=int, default=5,
                        help='k-hop neighborhood size (default: 5)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of U-Net layers (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Cones per batch (default: 1)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu, default: auto)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto)')

    args = parser.parse_args()

    # Initialize trainer
    trainer = ConeAlphaEarthTrainer(
        study_area=args.study_area,
        parent_resolution=args.parent_res,
        target_resolution=args.target_res,
        neighbor_rings=args.neighbor_rings,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir
    )

    # Train
    trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Results saved to: {trainer.output_dir}")
    logger.info(f"Best validation loss: {trainer.best_loss:.6f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
