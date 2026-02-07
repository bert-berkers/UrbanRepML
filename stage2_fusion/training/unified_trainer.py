"""
Unified Training Module with Embedding Extraction
=================================================

Provides unified training pipeline for all UNet variants with
built-in embedding extraction at multiple resolutions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime
import wandb

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Callback for extracting embeddings during training."""

    def __init__(self, resolutions: List[int] = None):
        """
        Initialize embedding extractor.

        Args:
            resolutions: H3 resolutions to extract (default: 5-10)
        """
        self.resolutions = resolutions or list(range(5, 11))
        self.embeddings = {}

    def __call__(self, model: nn.Module, data: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Extract embeddings from model.

        Args:
            model: UNet model
            data: Input data

        Returns:
            Dictionary mapping resolution to embeddings
        """
        model.eval()
        with torch.no_grad():
            # Forward pass
            outputs = model(
                data['features'],
                data['edge_index'],
                data.get('edge_weights'),
                data.get('batch')
            )

            # Extract multi-resolution embeddings
            if hasattr(model, 'extract_multi_resolution_embeddings'):
                self.embeddings = model.extract_multi_resolution_embeddings()
            else:
                # Fallback to single embedding
                self.embeddings = {10: outputs['embeddings']}

        return self.embeddings


class UnifiedTrainer:
    """
    Unified trainer for UNet models with embedding extraction.

    Handles training, validation, and embedding extraction for all UNet variants.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader,  # StudyAreaLoader instance
        config: Optional[Dict] = None,
        device: str = 'auto'
    ):
        """
        Initialize trainer.

        Args:
            model: UNet model to train
            data_loader: StudyAreaLoader instance
            config: Training configuration
            device: Device for training
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config or self._default_config()

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Initialize loss function
        self.criterion = self._create_loss_function()

        # Embedding extractor
        self.embedding_extractor = EmbeddingExtractor()

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        logger.info(f"Initialized UnifiedTrainer on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _default_config(self) -> Dict:
        """Get default training configuration."""
        return {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'early_stopping_patience': 10,
            'gradient_clip': 1.0,
            'validation_split': 0.2,
            'save_frequency': 10,
            'extract_frequency': 10
        }

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )

    def _create_loss_function(self) -> nn.Module:
        """Create loss function."""
        return nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")

        for batch in progress_bar:
            # Move to device
            batch = self._batch_to_device(batch)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                batch['features'],
                batch['edge_index'],
                batch.get('edge_weights'),
                batch.get('batch')
            )

            # Compute loss (self-reconstruction)
            loss = self.criterion(outputs['embeddings'], batch['features'])

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('gradient_clip'):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )

            # Update weights
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches

    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate model.

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = self._batch_to_device(batch)

                outputs = self.model(
                    batch['features'],
                    batch['edge_index'],
                    batch.get('edge_weights'),
                    batch.get('batch')
                )

                loss = self.criterion(outputs['embeddings'], batch['features'])

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_dataloader: DataLoader,
             val_dataloader: Optional[DataLoader] = None):
        """
        Complete training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
        """
        logger.info("Starting training")
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch

            # Training
            train_loss = self.train_epoch(train_dataloader)
            self.training_history['train_loss'].append(train_loss)

            # Validation
            val_loss = train_loss  # Default to train loss
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                self.training_history['val_loss'].append(val_loss)

            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.training_history['learning_rates'].append(current_lr)

            # Logging
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                       f"val_loss={val_loss:.4f}, lr={current_lr:.6f}")

            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best')
            else:
                patience_counter += 1

            if patience_counter >= self.config['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            # Periodic saving
            if epoch % self.config['save_frequency'] == 0:
                self.save_checkpoint(f'epoch_{epoch}')

            # Periodic embedding extraction
            if epoch % self.config['extract_frequency'] == 0:
                self.extract_and_save_embeddings()

        logger.info("Training complete")

    def extract_and_save_embeddings(self):
        """Extract and save embeddings for all resolutions."""
        logger.info("Extracting embeddings for all resolutions")

        # Get full dataset
        full_data = self._prepare_full_data()

        # Extract embeddings
        embeddings = self.embedding_extractor(self.model, full_data)

        # Save each resolution
        for resolution, emb in embeddings.items():
            self.data_loader.save_embeddings(
                emb,
                model_name='lattice_unet',
                resolution=resolution,
                metadata={
                    'epoch': self.current_epoch,
                    'model_class': self.model.__class__.__name__,
                    'device': str(self.device)
                }
            )

        logger.info(f"Saved embeddings for resolutions: {list(embeddings.keys())}")

    def extract_all_resolutions(self):
        """
        Extract embeddings for all resolutions after training.

        This is the main method to call after training completes.
        """
        logger.info("Extracting embeddings for all resolutions (final)")

        # Load best model
        self.load_checkpoint('best')

        # Prepare full dataset
        full_data = self._prepare_full_data()

        # Extract embeddings
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            outputs = self.model(
                full_data['features'],
                full_data['edge_index'],
                full_data.get('edge_weights'),
                full_data.get('batch')
            )

            # Get multi-resolution embeddings
            multi_res_embeddings = self.model.extract_multi_resolution_embeddings()

            # Save each resolution
            for resolution, embeddings in multi_res_embeddings.items():
                logger.info(f"Saving embeddings for resolution {resolution}: shape {embeddings.shape}")

                self.data_loader.save_embeddings(
                    embeddings,
                    model_name='lattice_unet',
                    resolution=resolution,
                    metadata={
                        'final': True,
                        'best_epoch': self.current_epoch,
                        'best_loss': self.best_loss,
                        'model_class': self.model.__class__.__name__
                    }
                )

        logger.info("Embedding extraction complete")

    def _prepare_full_data(self) -> Dict[str, torch.Tensor]:
        """Prepare full dataset for embedding extraction."""
        # This is a placeholder - implement based on your data structure
        # For now, return dummy data
        return {
            'features': torch.randn(1000, self.model.config.input_dim).to(self.device),
            'edge_index': torch.randint(0, 1000, (2, 5000)).to(self.device),
            'edge_weights': torch.ones(5000).to(self.device)
        }

    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config
        }

        self.data_loader.save_model(
            self.model,
            model_name=f'lattice_unet_{name}',
            config=self.config,
            metrics={'best_loss': self.best_loss, 'epoch': self.current_epoch}
        )

    def load_checkpoint(self, name: str):
        """Load training checkpoint."""
        checkpoint = self.data_loader.load_model(f'lattice_unet_{name}')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.current_epoch = checkpoint.get('epoch', 0)

        logger.info(f"Loaded checkpoint: {name}")

    def get_training_summary(self) -> Dict:
        """Get training summary."""
        return {
            'final_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device)
        }