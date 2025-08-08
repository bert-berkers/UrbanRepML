"""
UNet architecture adapted for hexagonal lattice graphs.
Works with regular lattice connectivity instead of accessibility-based graphs.
Designed for memory-efficient training with spatial batching.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphNorm, BatchNorm
from torch_geometric.utils import add_self_loops, degree
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from .spatial_batching import SpatialBatch

logger = logging.getLogger(__name__)


@dataclass
class LatticeUNetConfig:
    """Configuration for LatticeUNet model."""
    input_dim: int = 100  # Total embedding dimensions
    hidden_dim: int = 128
    output_dim: int = 64
    num_layers: int = 4
    dropout: float = 0.1
    
    # Convolution type
    conv_type: str = "gcn"  # "gcn", "gat"
    
    # GAT-specific parameters
    num_heads: int = 4
    
    # Normalization
    use_batch_norm: bool = False
    use_graph_norm: bool = True
    
    # Skip connections
    use_skip_connections: bool = True
    
    # Activation function
    activation: str = "gelu"
    
    # Loss configuration
    reconstruction_weight: float = 1.0
    consistency_weight: float = 0.5  # For multi-batch consistency
    

class GraphConvBlock(nn.Module):
    """Graph convolution block with normalization and activation."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: LatticeUNetConfig
    ):
        super().__init__()
        
        # Convolution layer
        if config.conv_type == "gcn":
            self.conv = GCNConv(in_dim, out_dim)
        elif config.conv_type == "gat":
            self.conv = GATConv(
                in_dim,
                out_dim // config.num_heads,
                heads=config.num_heads,
                dropout=config.dropout,
                concat=True
            )
        else:
            raise ValueError(f"Unknown conv_type: {config.conv_type}")
        
        # Normalization
        if config.use_batch_norm:
            self.norm = BatchNorm(out_dim)
        elif config.use_graph_norm:
            self.norm = GraphNorm(out_dim)
        else:
            self.norm = None
        
        # Activation
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through graph convolution block.
        
        Args:
            x: Node features [N, D]
            edge_index: Edge connectivity [2, E]  
            edge_weight: Edge weights [E]
            batch: Batch indices [N]
            
        Returns:
            Output features [N, D_out]
        """
        # Apply convolution
        if self.conv.__class__.__name__ == "GCNConv":
            x = self.conv(x, edge_index, edge_weight)
        else:  # GAT doesn't use edge weights
            x = self.conv(x, edge_index)
        
        # Apply normalization
        if self.norm is not None:
            if isinstance(self.norm, GraphNorm):
                x = self.norm(x, batch)
            else:
                x = self.norm(x)
        
        # Apply activation and dropout
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class LatticeUNet(nn.Module):
    """
    UNet architecture for hexagonal lattice graphs.
    
    Processes embeddings through encoder-decoder structure with skip connections.
    Works with regular hexagonal lattice connectivity patterns.
    """
    
    def __init__(self, config: LatticeUNetConfig):
        super().__init__()
        self.config = config
        
        logger.info(f"Initializing LatticeUNet:")
        logger.info(f"  Input dim: {config.input_dim}")
        logger.info(f"  Hidden dim: {config.hidden_dim}")
        logger.info(f"  Output dim: {config.output_dim}")
        logger.info(f"  Num layers: {config.num_layers}")
        logger.info(f"  Conv type: {config.conv_type}")
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        encoder_dims = self._get_encoder_dims()
        
        for i, (in_dim, out_dim) in enumerate(encoder_dims):
            self.encoder_layers.append(
                GraphConvBlock(in_dim, out_dim, config)
            )
        
        # Bottleneck
        self.bottleneck = GraphConvBlock(
            encoder_dims[-1][1],
            encoder_dims[-1][1],
            config
        )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        decoder_dims = self._get_decoder_dims(encoder_dims)
        
        for i, (in_dim, out_dim) in enumerate(decoder_dims):
            # Account for skip connections
            if config.use_skip_connections and i < len(encoder_dims):
                # Skip connection adds features from corresponding encoder layer
                skip_dim = encoder_dims[-(i+1)][1]  # Corresponding encoder output
                actual_in_dim = in_dim + skip_dim
            else:
                actual_in_dim = in_dim
            
            self.decoder_layers.append(
                GraphConvBlock(actual_in_dim, out_dim, config)
            )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Output activation (optional)
        self.output_activation = nn.Tanh()  # Normalize outputs to [-1, 1]
    
    def _get_encoder_dims(self) -> List[Tuple[int, int]]:
        """Get encoder layer dimensions."""
        dims = []
        current_dim = self.config.hidden_dim
        
        for i in range(self.config.num_layers):
            out_dim = current_dim
            dims.append((current_dim, out_dim))
        
        return dims
    
    def _get_decoder_dims(self, encoder_dims: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get decoder layer dimensions (reverse of encoder)."""
        decoder_dims = []
        
        for i in range(len(encoder_dims)):
            # Start from bottleneck, work backwards through encoder dims
            if i == 0:
                in_dim = encoder_dims[-1][1]  # From bottleneck
            else:
                in_dim = decoder_dims[-1][1]  # From previous decoder layer
            
            out_dim = self.config.hidden_dim
            decoder_dims.append((in_dim, out_dim))
        
        return decoder_dims
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LatticeUNet.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge connectivity [2, E]
            edge_weight: Edge weights [E]
            batch: Batch indices [N]
            
        Returns:
            Dictionary with outputs and intermediate features
        """
        # Input projection
        x = self.input_proj(x)
        x = F.gelu(x)
        
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x, edge_index, edge_weight, batch)
            encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x, edge_index, edge_weight, batch)
        
        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder_layers):
            # Add skip connection from corresponding encoder layer
            if self.config.use_skip_connections and i < len(encoder_outputs):
                skip_features = encoder_outputs[-(i+1)]  # Reverse order
                x = torch.cat([x, skip_features], dim=-1)
            
            x = decoder_layer(x, edge_index, edge_weight, batch)
        
        # Output projection
        embeddings = self.output_proj(x)
        embeddings = self.output_activation(embeddings)
        
        return {
            'embeddings': embeddings,
            'encoder_outputs': encoder_outputs,
            'input_features': x  # Features before output projection
        }
    
    def compute_reconstruction_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            outputs: Model outputs dictionary
            target: Target embeddings [N, input_dim]
            mask: Optional mask for valid nodes [N]
            
        Returns:
            Reconstruction loss
        """
        embeddings = outputs['embeddings']
        
        # Project embeddings back to input space for reconstruction
        if hasattr(self, 'reconstruction_proj'):
            reconstructed = self.reconstruction_proj(embeddings)
        else:
            # Create temporary projection layer
            if not hasattr(self, '_temp_recon_proj'):
                self._temp_recon_proj = nn.Linear(
                    self.config.output_dim,
                    self.config.input_dim
                ).to(embeddings.device)
            reconstructed = self._temp_recon_proj(embeddings)
        
        # MSE reconstruction loss
        if mask is not None:
            loss = F.mse_loss(reconstructed[mask], target[mask])
        else:
            loss = F.mse_loss(reconstructed, target)
        
        return loss
    
    def compute_consistency_loss(
        self,
        batch_outputs: List[Dict[str, torch.Tensor]],
        overlap_indices: List[List[int]]
    ) -> torch.Tensor:
        """
        Compute consistency loss between overlapping regions of batches.
        
        Args:
            batch_outputs: List of outputs from different batches
            overlap_indices: List of overlapping node indices for each batch pair
            
        Returns:
            Consistency loss
        """
        if len(batch_outputs) < 2 or not overlap_indices:
            return torch.tensor(0.0, device=batch_outputs[0]['embeddings'].device)
        
        consistency_losses = []
        
        for i, indices in enumerate(overlap_indices):
            if len(indices) < 2:
                continue
            
            # Get embeddings for overlapping regions
            emb1 = batch_outputs[i]['embeddings'][indices[0]]
            emb2 = batch_outputs[i+1]['embeddings'][indices[1]]
            
            # Cosine similarity loss (encourage similar representations)
            similarity = F.cosine_similarity(emb1, emb2, dim=-1)
            consistency_loss = (1 - similarity).mean()
            consistency_losses.append(consistency_loss)
        
        if consistency_losses:
            return torch.stack(consistency_losses).mean()
        else:
            return torch.tensor(0.0, device=batch_outputs[0]['embeddings'].device)


class BatchedLatticeTrainer:
    """Trainer for LatticeUNet with spatial batching support."""
    
    def __init__(
        self,
        model: LatticeUNet,
        device: str = "cuda",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Initialize batched trainer.
        
        Args:
            model: LatticeUNet model
            device: Device for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.config = model.config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100  # Will be updated based on epochs
        )
        
        logger.info(f"BatchedLatticeTrainer initialized on {device}")
    
    def train_step(
        self,
        batch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step on a batch.
        
        Args:
            batch_data: Batch data from spatial batcher
            
        Returns:
            Dictionary with losses
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Extract data
        features = batch_data['features'].to(self.device)
        edge_index = batch_data['edge_index'].to(self.device)
        edge_weights = batch_data['edge_weights'].to(self.device)
        
        # Handle batch indices if available
        batch_boundaries = batch_data.get('batch_boundaries')
        if batch_boundaries is not None:
            batch_boundaries = batch_boundaries.to(self.device)
        
        # Forward pass
        outputs = self.model(
            features,
            edge_index,
            edge_weights,
            batch_boundaries
        )
        
        # Compute reconstruction loss
        recon_loss = self.model.compute_reconstruction_loss(
            outputs,
            features  # Self-reconstruction task
        )
        
        # Total loss
        total_loss = self.config.reconstruction_weight * recon_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item()
        }
    
    def evaluate_step(
        self,
        batch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single evaluation step on a batch.
        
        Args:
            batch_data: Batch data from spatial batcher
            
        Returns:
            Dictionary with losses
        """
        self.model.eval()
        
        with torch.no_grad():
            # Extract data
            features = batch_data['features'].to(self.device)
            edge_index = batch_data['edge_index'].to(self.device)
            edge_weights = batch_data['edge_weights'].to(self.device)
            
            batch_boundaries = batch_data.get('batch_boundaries')
            if batch_boundaries is not None:
                batch_boundaries = batch_boundaries.to(self.device)
            
            # Forward pass
            outputs = self.model(
                features,
                edge_index,
                edge_weights,
                batch_boundaries
            )
            
            # Compute losses
            recon_loss = self.model.compute_reconstruction_loss(
                outputs,
                features
            )
            
            total_loss = self.config.reconstruction_weight * recon_loss
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item()
        }
    
    def train_epoch(
        self,
        dataloader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with spatial batches
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses
        """
        total_losses = []
        recon_losses = []
        
        for batch_idx, batch_data in enumerate(dataloader):
            losses = self.train_step(batch_data)
            
            total_losses.append(losses['total_loss'])
            recon_losses.append(losses['reconstruction_loss'])
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Loss = {losses['total_loss']:.4f}, "
                    f"Recon = {losses['reconstruction_loss']:.4f}"
                )
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'total_loss': np.mean(total_losses),
            'reconstruction_loss': np.mean(recon_losses),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate_epoch(
        self,
        dataloader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Evaluate for one epoch.
        
        Args:
            dataloader: DataLoader with spatial batches
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses
        """
        total_losses = []
        recon_losses = []
        
        for batch_data in dataloader:
            losses = self.evaluate_step(batch_data)
            
            total_losses.append(losses['total_loss'])
            recon_losses.append(losses['reconstruction_loss'])
        
        return {
            'total_loss': np.mean(total_losses),
            'reconstruction_loss': np.mean(recon_losses)
        }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint and return epoch."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from {path} (epoch {epoch})")
        
        return epoch


# Factory function for creating model configurations
def create_lattice_unet_config(
    input_dim: int,
    model_size: str = "medium",
    **kwargs
) -> LatticeUNetConfig:
    """
    Create LatticeUNet configuration with predefined sizes.
    
    Args:
        input_dim: Input embedding dimensions
        model_size: Size preset ("small", "medium", "large")
        **kwargs: Additional config overrides
        
    Returns:
        LatticeUNetConfig object
    """
    size_presets = {
        "small": {
            "hidden_dim": 64,
            "output_dim": 32,
            "num_layers": 3,
            "dropout": 0.1
        },
        "medium": {
            "hidden_dim": 128,
            "output_dim": 64,
            "num_layers": 4,
            "dropout": 0.1
        },
        "large": {
            "hidden_dim": 256,
            "output_dim": 128,
            "num_layers": 6,
            "dropout": 0.2
        }
    }
    
    if model_size not in size_presets:
        raise ValueError(f"Unknown model_size: {model_size}")
    
    preset = size_presets[model_size]
    
    config = LatticeUNetConfig(
        input_dim=input_dim,
        **preset,
        **kwargs
    )
    
    return config