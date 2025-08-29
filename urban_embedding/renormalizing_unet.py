"""
Renormalizing Hierarchical U-Net for Urban Representation Learning

Implements a hierarchical generative model inspired by Friston et al.'s renormalizing 
generative models. Features:

- H3 resolutions 5-10 (sustainability → liveability)
- Upward flow: Accumulated/batched updates with normalization
- Downward flow: Direct pass-through of all updates
- Simple MSE losses: consistency + reconstruction at res 10
- No Active Inference - pure transmission architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RenormalizingConfig:
    """Configuration for renormalizing data flow."""
    accumulation_mode: str = "grouped"  # "grouped", "layered", "adaptive"
    normalization_type: str = "layer"   # "layer", "group", "batch"
    upward_momentum: float = 0.9        # Momentum for upward accumulation
    downward_scaling: float = 1.0       # Scaling for downward pass-through
    residual_connections: bool = True   # Enable residual connections
    gradient_checkpointing: bool = False # Memory efficiency


class RenormalizingLayer(nn.Module):
    """
    Implements renormalizing data flow between resolution levels.
    
    Upward flow: Accumulates and normalizes updates
    Downward flow: Direct pass-through with optional scaling
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 config: RenormalizingConfig,
                 direction: str = "upward"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        self.direction = direction
        
        # Normalization layers
        if config.normalization_type == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        elif config.normalization_type == "group":
            # Assume 8 groups for group normalization
            num_groups = min(8, hidden_dim // 4)
            self.norm = nn.GroupNorm(num_groups, hidden_dim)
        elif config.normalization_type == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = nn.Identity()
        
        # Accumulation mechanism for upward flow
        if direction == "upward":
            self.accumulator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # Momentum tracking for accumulated updates
            self.register_buffer('momentum_state', torch.zeros(1, hidden_dim))
            
        # Direct transformation for downward flow
        else:  # downward
            self.transform = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ) if config.downward_scaling != 1.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply renormalizing transformation.
        
        Args:
            x: Input features [batch_size, hidden_dim]
            prev_state: Previous state for accumulation (upward only)
        
        Returns:
            Transformed features
        """
        if self.direction == "upward":
            return self._upward_flow(x, prev_state)
        else:
            return self._downward_flow(x)
    
    def _upward_flow(self, x: torch.Tensor, prev_state: Optional[torch.Tensor]) -> torch.Tensor:
        """Accumulated/batched upward updates with momentum."""
        # Apply accumulation transformation
        accumulated = self.accumulator(x)
        
        # Add momentum from previous state
        if prev_state is not None:
            momentum_update = self.config.upward_momentum * prev_state
            accumulated = accumulated + momentum_update
        
        # Update momentum state
        self.momentum_state = self.momentum_state * self.config.upward_momentum + \
                             accumulated.mean(dim=0, keepdim=True) * (1 - self.config.upward_momentum)
        
        # Normalize accumulated updates
        normalized = self.norm(accumulated)
        
        # Residual connection if enabled
        if self.config.residual_connections:
            normalized = normalized + x
        
        return F.normalize(normalized, p=2, dim=-1, eps=1e-8)
    
    def _downward_flow(self, x: torch.Tensor) -> torch.Tensor:
        """Direct pass-through downward flow."""
        # Apply optional transformation
        transformed = self.transform(x)
        
        # Apply downward scaling
        if self.config.downward_scaling != 1.0:
            transformed = transformed * self.config.downward_scaling
        
        # Residual connection if enabled
        if self.config.residual_connections:
            transformed = transformed + x
        
        return F.normalize(transformed, p=2, dim=-1, eps=1e-8)


class HierarchicalEncoderBlock(nn.Module):
    """Enhanced encoder block with renormalizing flow."""
    
    def __init__(self, 
                 hidden_dim: int, 
                 num_convs: int,
                 config: RenormalizingConfig):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        
        # Graph convolution layers
        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
            for _ in range(num_convs)
        ])
        
        # Normalization for each conv layer
        self.conv_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_convs)
        ])
        
        # Renormalizing layer for upward flow
        self.renorm_layer = RenormalizingLayer(
            hidden_dim=hidden_dim,
            config=config,
            direction="upward"
        )
        
        # Residual transformation
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_weight: torch.Tensor,
                prev_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with renormalizing upward flow.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]
            prev_state: Previous state for renormalization
        
        Returns:
            Encoded features with renormalizing updates
        """
        # Store input for residual
        identity = self.residual(x)
        
        # Apply graph convolutions
        out = x
        for conv, norm in zip(self.convs, self.conv_norms):
            out = conv(out, edge_index, edge_weight)
            out = norm(out)
            out = F.gelu(out)
        
        # Add residual connection
        out = out + identity
        
        # Apply renormalizing upward flow
        out = self.renorm_layer(out, prev_state)
        
        return out


class HierarchicalDecoderBlock(nn.Module):
    """Enhanced decoder block with renormalizing flow."""
    
    def __init__(self, 
                 hidden_dim: int, 
                 num_convs: int,
                 config: RenormalizingConfig):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        
        # Graph convolution layers  
        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
            for _ in range(num_convs)
        ])
        
        # Normalization for each conv layer
        self.conv_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_convs)
        ])
        
        # Renormalizing layer for downward flow
        self.renorm_layer = RenormalizingLayer(
            hidden_dim=hidden_dim,
            config=config,
            direction="downward"
        )
        
        # Residual transformation
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, 
                x: torch.Tensor, 
                skip: torch.Tensor,
                edge_index: torch.Tensor, 
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with renormalizing downward flow.
        
        Args:
            x: Input features [num_nodes, hidden_dim]
            skip: Skip connection from encoder [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]
        
        Returns:
            Decoded features with direct pass-through updates
        """
        # Combine with skip connection
        combined = x + skip
        identity = self.residual(combined)
        
        # Apply graph convolutions
        out = combined
        for conv, norm in zip(self.convs, self.conv_norms):
            out = conv(out, edge_index, edge_weight)
            out = norm(out)
            out = F.gelu(out)
        
        # Add residual connection
        out = out + identity
        
        # Apply renormalizing downward flow (direct pass-through)
        out = self.renorm_layer(out)
        
        return out


class SharedSparseMapping(nn.Module):
    """Enhanced sparse mapping with renormalizing flow."""
    
    def __init__(self, hidden_dim: int, config: RenormalizingConfig):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        
        # Learnable transformation
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Renormalizing layer for cross-resolution updates
        self.renorm_layer = RenormalizingLayer(
            hidden_dim=hidden_dim,
            config=config,
            direction="upward"  # Cross-resolution is treated as upward
        )
    
    def forward(self, x: torch.Tensor, mapping: torch.sparse.Tensor) -> torch.Tensor:
        """Apply sparse mapping with renormalizing flow."""
        # Apply sparse mapping
        mapped = torch.sparse.mm(mapping, x)
        
        # Apply learnable transformation
        transformed = self.transform(mapped)
        
        # Apply renormalizing flow
        renormalized = self.renorm_layer(transformed)
        
        return renormalized


class RenormalizingUrbanUNet(nn.Module):
    """
    Renormalizing Hierarchical U-Net for Urban Representation Learning.
    
    Features:
    - H3 resolutions 5-10 (6 levels: sustainability → liveability)
    - Renormalizing data flow: upward accumulation, downward pass-through
    - Simple MSE losses: consistency between levels + reconstruction at res 10
    - Pure transmission architecture (no Active Inference)
    """
    
    def __init__(self,
                 feature_dims: Dict[str, int],
                 hidden_dim: int = 128,
                 output_dim: int = 32,
                 num_convs: int = 4,
                 renorm_config: Optional[RenormalizingConfig] = None,
                 device: str = "cuda"):
        super().__init__()
        
        self.device = device
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.resolutions = [10, 9, 8, 7, 6, 5]  # Liveability → Sustainability (definitive order)
        
        # Configuration for renormalizing flow
        self.renorm_config = renorm_config or RenormalizingConfig()
        
        # Initial modality fusion (same as before)
        from urban_embedding.model import ModalityFusion
        self.fusion = ModalityFusion(feature_dims, hidden_dim)
        
        # Enhanced sparse mappings with renormalizing flow
        self.mapping_transforms = nn.ModuleDict()
        for i in range(len(self.resolutions) - 1):
            fine_res = self.resolutions[i + 1]
            coarse_res = self.resolutions[i]
            self.mapping_transforms[f"{fine_res}_{coarse_res}"] = SharedSparseMapping(
                hidden_dim, self.renorm_config
            )
        
        # Hierarchical encoder path (res 10 → 5)
        self.encoders = nn.ModuleDict()
        for i, res in enumerate(self.resolutions):  # 10, 9, 8, 7, 6, 5
            self.encoders[str(res)] = HierarchicalEncoderBlock(
                hidden_dim=hidden_dim,
                num_convs=num_convs,
                config=self.renorm_config
            )
        
        # Hierarchical decoder path (res 5 → 10)  
        self.decoders = nn.ModuleDict()
        for i, res in enumerate(self.resolutions[::-1]):  # 5, 6, 7, 8, 9, 10
            self.decoders[str(res)] = HierarchicalDecoderBlock(
                hidden_dim=hidden_dim,
                num_convs=num_convs,
                config=self.renorm_config
            )
        
        # Output projections for each resolution
        self.output_projections = nn.ModuleDict()
        for res in self.resolutions:
            self.output_projections[str(res)] = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        
        # Reconstruction heads (only for res 10 - the "blanket")
        self.reconstructors = nn.ModuleDict()
        for name, dim in feature_dims.items():
            self.reconstructors[name] = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
                nn.LayerNorm(dim)
            )
        
        logger.info(f"Initialized RenormalizingUrbanUNet with resolutions {self.resolutions}")
        logger.info(f"Renormalizing config: {self.renorm_config}")
    
    def forward(self,
                features_dict: Dict[str, torch.Tensor],
                edge_indices: Dict[int, torch.Tensor],
                edge_weights: Dict[int, torch.Tensor],
                mappings: Dict[Tuple[int, int], torch.Tensor]) -> Tuple[Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with renormalizing hierarchical processing.
        
        Args:
            features_dict: Input features per modality
            edge_indices: Graph connectivity per resolution
            edge_weights: Edge weights per resolution  
            mappings: Cross-resolution mappings (fine->coarse)
        
        Returns:
            embeddings: Learned embeddings per resolution
            reconstructed: Reconstructed features per modality (only from res 10)
        """
        # Initial fusion at resolution 10
        x = self.fusion(features_dict)
        
        # Store encoder states for skip connections
        encoder_states = {}
        prev_state = None
        
        # ENCODER PATH: Upward flow (res 10 → 5) with accumulation
        current_features = x
        for i, res in enumerate(self.resolutions):  # 10, 9, 8, 7, 6, 5
            # Apply encoder with renormalizing upward flow
            encoded = self.encoders[str(res)](
                current_features,
                edge_indices[res], 
                edge_weights[res],
                prev_state=prev_state
            )
            
            encoder_states[res] = encoded
            prev_state = encoded  # Pass state for momentum
            
            # Map to next coarser resolution (except at deepest level)
            if i < len(self.resolutions) - 1:
                next_res = self.resolutions[i + 1]
                mapping_key = f"{res}_{next_res}"
                current_features = self.mapping_transforms[mapping_key](
                    encoded, mappings[(res, next_res)].t()
                )
            else:
                current_features = encoded
        
        # DECODER PATH: Downward flow (res 5 → 10) with direct pass-through
        decoder_states = {}
        current_features = encoder_states[self.resolutions[-1]]  # Start from deepest level (5)
        
        for i, res in enumerate(self.resolutions[::-1]):  # 5, 6, 7, 8, 9, 10
            # Apply decoder with renormalizing downward flow
            if res == 5:
                # Deepest level - no skip connection from "above"
                decoded = self.decoders[str(res)](
                    current_features,
                    encoder_states[res],  # Skip from encoder
                    edge_indices[res],
                    edge_weights[res]
                )
            else:
                decoded = self.decoders[str(res)](
                    current_features,
                    encoder_states[res],  # Skip from encoder
                    edge_indices[res],
                    edge_weights[res]
                )
            
            decoder_states[res] = decoded
            
            # Map to next finer resolution (except at finest level)  
            if i < len(self.resolutions[::-1]) - 1:
                next_res = self.resolutions[::-1][i + 1]  # Next in 5,6,7,8,9,10 order
                mapping_key = f"{next_res}_{res}"  # Note: reverse mapping direction
                current_features = self.mapping_transforms[mapping_key](
                    decoded, mappings[(next_res, res)]  # Direct mapping
                )
            else:
                current_features = decoded
        
        # Generate embeddings at all resolutions
        embeddings = {}
        for res in self.resolutions:
            embedding = F.normalize(
                self.output_projections[str(res)](decoder_states[res]),
                p=2, dim=-1, eps=1e-8
            )
            embeddings[res] = embedding
        
        # Generate reconstructions ONLY from resolution 10 (the "blanket")
        reconstructed = {}
        finest_embedding = embeddings[10]
        for name in features_dict.keys():
            reconstructed[name] = F.normalize(
                self.reconstructors[name](finest_embedding),
                p=2, dim=-1, eps=1e-8
            )
        
        return embeddings, reconstructed


class RenormalizingLossComputer:
    """Enhanced loss computation for 6-level hierarchy."""
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.resolutions = [10, 9, 8, 7, 6, 5]
    
    def compute_losses(self,
                      embeddings: Dict[int, torch.Tensor],
                      reconstructed: Dict[str, torch.Tensor],
                      features_dict: Dict[str, torch.Tensor],
                      mappings: Dict[Tuple[int, int], torch.Tensor],
                      loss_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Compute losses for renormalizing hierarchical model.
        
        Features:
        - Reconstruction loss ONLY at resolution 10 (the blanket)
        - Consistency losses between ALL adjacent resolution pairs
        - Equal weighting across all consistency terms
        """
        
        # 1. Reconstruction loss (ONLY at resolution 10)
        recon_losses = {}
        for name, pred in reconstructed.items():
            target = features_dict[name]
            recon_losses[name] = F.mse_loss(
                F.normalize(pred, p=2, dim=1, eps=self.epsilon),
                F.normalize(target, p=2, dim=1, eps=self.epsilon)
            )
        total_recon_loss = sum(recon_losses.values()) * loss_weights['reconstruction']
        
        # 2. Consistency losses between ALL adjacent resolution pairs
        consistency_losses = {}
        
        # Process all adjacent pairs: (10,9), (9,8), (8,7), (7,6), (6,5)
        for i in range(len(self.resolutions) - 1):
            fine_res = self.resolutions[i]        # 10, 9, 8, 7, 6
            coarse_res = self.resolutions[i + 1]  # 9, 8, 7, 6, 5
            
            if (fine_res, coarse_res) in mappings:
                # Map fine embeddings to coarse resolution
                mapping = mappings[(fine_res, coarse_res)]
                fine_mapped = torch.sparse.mm(mapping.t(), embeddings[fine_res])
                fine_mapped = F.normalize(fine_mapped, p=2, dim=1, eps=self.epsilon)
                coarse_emb = F.normalize(embeddings[coarse_res], p=2, dim=1, eps=self.epsilon)
                
                # Calculate MSE for this resolution pair
                consistency_losses[(fine_res, coarse_res)] = F.mse_loss(fine_mapped, coarse_emb)
        
        # Average consistency losses (equal weight per transition)
        if consistency_losses:
            total_consistency_loss = (sum(consistency_losses.values()) / len(consistency_losses)) * loss_weights['consistency']
        else:
            total_consistency_loss = torch.tensor(0.0)
        
        # Compile all losses
        all_losses = {
            'total_loss': total_recon_loss + total_consistency_loss,
            'reconstruction_loss': total_recon_loss,
            'consistency_loss': total_consistency_loss,
        }
        
        # Add individual reconstruction losses
        for name, loss in recon_losses.items():
            all_losses[f'recon_loss_{name}'] = loss
        
        # Add individual consistency losses  
        for (fine_res, coarse_res), loss in consistency_losses.items():
            all_losses[f'consistency_loss_{fine_res}_{coarse_res}'] = loss
        
        return all_losses


def create_renormalizing_config(
    accumulation_mode: str = "grouped",
    normalization_type: str = "layer", 
    upward_momentum: float = 0.9,
    residual_connections: bool = True
) -> RenormalizingConfig:
    """Create renormalizing configuration with common presets."""
    return RenormalizingConfig(
        accumulation_mode=accumulation_mode,
        normalization_type=normalization_type,
        upward_momentum=upward_momentum,
        downward_scaling=1.0,
        residual_connections=residual_connections,
        gradient_checkpointing=False
    )


if __name__ == "__main__":
    # Test the renormalizing architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mock feature dimensions
    feature_dims = {
        'aerial_alphaearth': 64,
        'gtfs': 32, 
        'roadnetwork': 32,
        'poi': 32
    }
    
    # Create model with renormalizing config
    renorm_config = create_renormalizing_config(
        accumulation_mode="grouped",
        normalization_type="layer",
        upward_momentum=0.9
    )
    
    model = RenormalizingUrbanUNet(
        feature_dims=feature_dims,
        hidden_dim=128,
        output_dim=32,
        renorm_config=renorm_config,
        device=device
    )
    
    print("✅ RenormalizingUrbanUNet initialized successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Resolutions supported: {model.resolutions}")
    print(f"Renormalizing config: {model.renorm_config}")