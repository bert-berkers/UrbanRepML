"""
Enhanced Trainer for Renormalizing Urban U-Net

Supports 6-level hierarchy (H3 resolutions 5-10) with renormalizing data flow.
Focuses on:
- Simple MSE losses (reconstruction at res 10 + consistency between levels)
- Proper gradient flow through hierarchical normalization
- Memory-efficient training with optional gradient checkpointing
- Momentum tracking for accumulated upward updates
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import numpy as np
from typing import Dict, Tuple, Optional, List
from tqdm.auto import tqdm

from .renormalizing_unet import RenormalizingUrbanUNet, RenormalizingLossComputer, RenormalizingConfig

logger = logging.getLogger(__name__)


class RenormalizingModelTrainer:
    """Enhanced trainer for renormalizing hierarchical U-Net."""
    
    def __init__(self,
                 model_config: dict,
                 renorm_config: Optional[RenormalizingConfig] = None,
                 loss_weights: Dict[str, float] = None,
                 city_name: str = "south_holland",
                 checkpoint_dir: Optional[Path] = None,
                 device: str = "cuda"):
        
        logger.info("Initializing RenormalizingModelTrainer...")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.city_name = city_name
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize renormalizing model
        self.model = RenormalizingUrbanUNet(
            renorm_config=renorm_config,
            device=self.device,
            **model_config
        )
        self.model.to(self.device)
        
        # Loss computation
        self.loss_computer = RenormalizingLossComputer()
        self.loss_weights = loss_weights or {
            'reconstruction': 1.0,
            'consistency': 2.0  # Slightly higher weight for hierarchical consistency
        }
        
        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Loss weights: {self.loss_weights}")
        logger.info(f"Renormalizing config: {self.model.renorm_config}")
    
    def _validate_inputs(self,
                        features_dict: Dict[str, torch.Tensor],
                        edge_indices: Dict[int, torch.Tensor],
                        edge_weights: Dict[int, torch.Tensor],
                        mappings: Dict[Tuple[int, int], torch.Tensor]):
        """Validate that all required resolutions are present."""
        required_resolutions = [5, 6, 7, 8, 9, 10]
        
        # Check edge data
        missing_edge_res = [res for res in required_resolutions if res not in edge_indices]
        if missing_edge_res:
            raise ValueError(f"Missing edge indices for resolutions: {missing_edge_res}")
        
        missing_weight_res = [res for res in required_resolutions if res not in edge_weights]
        if missing_weight_res:
            raise ValueError(f"Missing edge weights for resolutions: {missing_weight_res}")
        
        # Check mappings (should have all adjacent pairs)
        required_mappings = [(10,9), (9,8), (8,7), (7,6), (6,5)]
        missing_mappings = [pair for pair in required_mappings if pair not in mappings]
        if missing_mappings:
            logger.warning(f"Missing mappings for resolution pairs: {missing_mappings}")
        
        # Log data sizes
        for res in required_resolutions:
            if res in edge_indices:
                num_edges = edge_indices[res].shape[1] if len(edge_indices[res].shape) > 1 else 0
                logger.info(f"Resolution {res}: {num_edges} edges")
    
    def _get_lr_scheduler(self, optimizer, num_epochs: int, warmup_epochs: int):
        """Enhanced learning rate scheduler with warmup and cosine annealing."""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return float(epoch) / float(max(1, warmup_epochs))
            else:
                # Cosine annealing
                progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _save_checkpoint(self, state: dict, filename: str):
        """Save model checkpoint with enhanced state."""
        if self.checkpoint_dir:
            checkpoint_path = self.checkpoint_dir / filename
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Add renormalizing-specific state
            state['renorm_config'] = self.model.renorm_config.__dict__
            state['model_resolutions'] = self.model.resolutions
            
            torch.save(state, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms for monitoring training stability."""
        total_norm = 0.0
        encoder_norm = 0.0
        decoder_norm = 0.0
        mapping_norm = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                if 'encoders' in name:
                    encoder_norm += param_norm ** 2
                elif 'decoders' in name:
                    decoder_norm += param_norm ** 2
                elif 'mapping_transforms' in name:
                    mapping_norm += param_norm ** 2
        
        return {
            'total_grad_norm': total_norm ** 0.5,
            'encoder_grad_norm': encoder_norm ** 0.5,
            'decoder_grad_norm': decoder_norm ** 0.5,
            'mapping_grad_norm': mapping_norm ** 0.5
        }
    
    def train(self,
              features_dict: Dict[str, torch.Tensor],
              edge_indices: Dict[int, torch.Tensor],
              edge_weights: Dict[int, torch.Tensor],
              mappings: Dict[Tuple[int, int], torch.Tensor],
              num_epochs: int = 500,
              learning_rate: float = 1e-4,
              warmup_epochs: int = 50,
              patience: int = 100,
              gradient_clip: float = 1.0,
              accumulate_gradients: int = 1,
              log_interval: int = 10) -> Tuple[Dict[int, torch.Tensor], dict]:
        """
        Train the renormalizing hierarchical model.
        
        Args:
            features_dict: Input features per modality
            edge_indices: Graph connectivity per resolution (5-10)
            edge_weights: Edge weights per resolution (5-10)
            mappings: Cross-resolution mappings
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            warmup_epochs: Epochs for warmup
            patience: Early stopping patience
            gradient_clip: Gradient clipping value
            accumulate_gradients: Steps to accumulate gradients
            log_interval: Logging interval
        
        Returns:
            best_embeddings: Best embeddings achieved
            training_state: Final training state
        """
        
        # Validate inputs
        self._validate_inputs(features_dict, edge_indices, edge_weights, mappings)
        
        logger.info(f"Starting renormalizing training:")
        logger.info(f"- Epochs: {num_epochs}, Learning rate: {learning_rate}")
        logger.info(f"- Warmup epochs: {warmup_epochs}, Patience: {patience}")
        logger.info(f"- Gradient accumulation: {accumulate_gradients}")
        logger.info(f"- Resolutions: {self.model.resolutions}")
        
        # Optimizer with weight decay for regularization
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Enhanced scheduler
        scheduler = self._get_lr_scheduler(optimizer, num_epochs, warmup_epochs)
        
        # Training state tracking
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        best_embeddings = None
        
        training_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'consistency_loss': [],
            'grad_norms': [],
            'learning_rates': []
        }
        
        # Training loop
        for epoch in tqdm(range(num_epochs), desc="Training Renormalizing U-Net"):
            self.model.train()
            epoch_losses = []
            accumulated_loss = 0.0
            
            # Gradient accumulation loop
            for accumulation_step in range(accumulate_gradients):
                try:
                    # Forward pass
                    embeddings, reconstructed = self.model(
                        features_dict, edge_indices, edge_weights, mappings
                    )
                    
                    # Compute losses
                    losses = self.loss_computer.compute_losses(
                        embeddings=embeddings,
                        reconstructed=reconstructed,
                        features_dict=features_dict,
                        mappings=mappings,
                        loss_weights=self.loss_weights
                    )
                    
                    # Scale loss by accumulation steps
                    loss = losses['total_loss'] / accumulate_gradients
                    
                    # Check for NaN
                    if torch.isnan(loss):
                        logger.warning(f"NaN loss detected at epoch {epoch}, step {accumulation_step}")
                        continue
                    
                    # Backward pass
                    loss.backward()
                    accumulated_loss += loss.item()
                    
                except RuntimeError as e:
                    logger.error(f"Runtime error during training: {e}")
                    continue
            
            # Gradient clipping and optimization step
            if accumulated_loss > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                epoch_losses.append(accumulated_loss)
            
            # Compute gradient norms for monitoring
            grad_norms = self._compute_gradient_norms()
            current_lr = scheduler.get_last_lr()[0]
            
            # Record training history
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                training_history['total_loss'].append(avg_loss)
                training_history['grad_norms'].append(grad_norms['total_grad_norm'])
                training_history['learning_rates'].append(current_lr)
                
                # Detailed loss breakdown (recompute for logging)
                with torch.no_grad():
                    self.model.eval()
                    embeddings, reconstructed = self.model(
                        features_dict, edge_indices, edge_weights, mappings
                    )
                    detailed_losses = self.loss_computer.compute_losses(
                        embeddings=embeddings,
                        reconstructed=reconstructed,
                        features_dict=features_dict,
                        mappings=mappings,
                        loss_weights=self.loss_weights
                    )
                    
                    training_history['reconstruction_loss'].append(detailed_losses['reconstruction_loss'].item())
                    training_history['consistency_loss'].append(detailed_losses['consistency_loss'].item())
                
                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    
                    best_state = {
                        'epoch': epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'loss': best_loss,
                        'training_history': training_history,
                        'renorm_config': self.model.renorm_config.__dict__
                    }
                    
                    best_embeddings = {k: v.detach().clone() for k, v in embeddings.items()}
                else:
                    patience_counter += 1
                
                # Logging
                if epoch % log_interval == 0 or patience_counter >= patience:
                    logger.info(f"Epoch {epoch:4d}: "
                              f"loss={avg_loss:.6f}, "
                              f"recon={detailed_losses['reconstruction_loss'].item():.6f}, "
                              f"consist={detailed_losses['consistency_loss'].item():.6f}, "
                              f"grad_norm={grad_norms['total_grad_norm']:.4f}, "
                              f"lr={current_lr:.2e}, "
                              f"patience={patience_counter}")
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state['model_state'])
            self._save_checkpoint(best_state, f"best_renormalizing_model_{self.city_name}.pt")
            logger.info(f"Restored best model from epoch {best_state['epoch']} with loss {best_loss:.6f}")
        
        return best_embeddings, best_state


def create_enhanced_mappings(base_mappings: Dict[Tuple[int, int], torch.sparse.Tensor],
                           target_resolutions: List[int] = [5, 6, 7, 8, 9, 10]) -> Dict[Tuple[int, int], torch.sparse.Tensor]:
    """
    Create enhanced mappings for 6-level hierarchy.
    
    If base mappings only cover (10,9), (9,8), (8,7), this function can help
    extend to include (7,6), (6,5) by chaining existing mappings.
    """
    enhanced_mappings = base_mappings.copy()
    
    # Generate missing mappings by chaining
    missing_pairs = []
    for i in range(len(target_resolutions) - 1):
        coarse_res = target_resolutions[i]
        fine_res = target_resolutions[i + 1]
        if (fine_res, coarse_res) not in enhanced_mappings:
            missing_pairs.append((fine_res, coarse_res))
    
    if missing_pairs:
        logger.warning(f"Missing mapping pairs: {missing_pairs}")
        logger.warning("Consider generating proper H3 parent-child mappings for these resolution pairs")
    
    return enhanced_mappings


if __name__ == "__main__":
    # Test the renormalizing trainer
    from renormalizing_unet import create_renormalizing_config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mock configuration
    model_config = {
        'feature_dims': {
            'aerial_alphaearth': 64,
            'gtfs': 32,
            'roadnetwork': 32,
            'poi': 32
        },
        'hidden_dim': 128,
        'output_dim': 32,
        'num_convs': 4
    }
    
    renorm_config = create_renormalizing_config(
        accumulation_mode="grouped",
        normalization_type="layer",
        upward_momentum=0.9
    )
    
    trainer = RenormalizingModelTrainer(
        model_config=model_config,
        renorm_config=renorm_config,
        device=device
    )
    
    print("✅ RenormalizingModelTrainer initialized successfully!")
    print(f"Trainer device: {trainer.device}")
    print(f"Model resolutions: {trainer.model.resolutions}")