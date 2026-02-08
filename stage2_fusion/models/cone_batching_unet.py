"""
ConeBatchingUNet - Cone-Based Hierarchical U-Net

CONE-BASED ARCHITECTURE:
Each res5 hexagon defines a cone (center + 5-ring neighbors + all descendants to res10).
Cones OVERLAP significantly - each res10 hexagon appears in ~10 different cones.

Training: Process one cone at a time (completely independent)
Inference: Process all cones → weighted average by distance to cone center

Multi-resolution U-Net operating on hierarchical cones where:
- Resolution 10 (finest) = Markov blanket (observations/input) for THIS cone
- Resolutions 5-9 (coarser) = Internal states (latent representations) for THIS cone
- Per-resolution processing: each layer operates only on nodes at its resolution

Architecture:
    Encoder: res10 → res5 (bottom-up, 2 GCN hops per resolution)
    Decoder: res5 → res10 (top-down, 2 GCN hops per resolution)

Efficient structure - each resolution processed separately:
    - Res10: ~1.5M nodes/cone × 2 GCN (64→64)
    - Res9: ~214K nodes/cone × 2 GCN (128→128)
    - Res8: ~30K nodes/cone × 2 GCN (128→128)
    - Res7: ~4K nodes/cone × 2 GCN (256→256)
    - Res6: ~637 nodes/cone × 2 GCN (256→256)
    - Res5: ~91 nodes/cone × 2 GCN (512→512)

Inspired by hierarchical active inference and predictive coding.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphNorm
from torch_scatter import scatter_mean, scatter_add
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConeBatchingUNetConfig:
    """Configuration for MultiResolutionConeUNet."""

    # Input/output dimensions
    input_dim: int = 64  # AlphaEarth embeddings at res10
    output_dim: int = 64  # Reconstructed res10

    # Hidden dimensions per resolution
    hidden_dims: Dict[int, int] = None

    # Architecture
    lateral_conv_layers: int = 2  # Convolutions per resolution
    conv_type: str = "gcn"  # "gcn" or "gat"
    num_heads: int = 4  # For GAT

    # Regularization
    dropout: float = 0.1
    use_graph_norm: bool = True
    use_skip_connections: bool = True

    # Activation
    activation: str = "gelu"

    def __post_init__(self):
        """Set default hidden dimensions if not provided."""
        if self.hidden_dims is None:
            self.hidden_dims = {
                10: 64,   # Input resolution
                9: 128,
                8: 128,
                7: 256,
                6: 256,
                5: 512    # Bottleneck (highest latent)
            }


class GraphConvBlock(nn.Module):
    """Graph convolution block with normalization and activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        conv_type: str = "gcn",
        dropout: float = 0.1,
        activation: str = "gelu",
        use_graph_norm: bool = True,
        num_heads: int = 4
    ):
        super().__init__()

        # Convolution layer
        if conv_type == "gcn":
            self.conv = GCNConv(in_dim, out_dim)
        elif conv_type == "gat":
            self.conv = GATConv(
                in_dim,
                out_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.conv_type = conv_type

        # Normalization
        self.norm = GraphNorm(out_dim) if use_graph_norm else None

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through graph convolution."""
        # Apply convolution
        if self.conv_type == "gcn":
            x = self.conv(x, edge_index, edge_weight)
        else:  # GAT
            x = self.conv(x, edge_index)

        # Normalization
        if self.norm is not None:
            x = self.norm(x, batch)

        # Activation and dropout
        x = self.activation(x)
        x = self.dropout(x)

        return x


class HierarchicalAggregation(nn.Module):
    """Aggregate children features to parents (bottom-up)."""

    def __init__(self, aggregation: str = "mean"):
        super().__init__()
        self.aggregation = aggregation

    def forward(
        self,
        child_features: torch.Tensor,
        child_to_parent_idx: torch.Tensor,
        num_parents: int
    ) -> torch.Tensor:
        """
        Aggregate children to parents.

        Args:
            child_features: [num_children, dim]
            child_to_parent_idx: [num_children] - parent index for each child
            num_parents: Number of parent nodes

        Returns:
            parent_features: [num_parents, dim]
        """
        if self.aggregation == "mean":
            parent_features = scatter_mean(
                child_features,
                child_to_parent_idx,
                dim=0,
                dim_size=num_parents
            )
        elif self.aggregation == "sum":
            parent_features = scatter_add(
                child_features,
                child_to_parent_idx,
                dim=0,
                dim_size=num_parents
            )
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return parent_features


class HierarchicalBroadcast(nn.Module):
    """Broadcast parent features to children (top-down)."""

    def forward(
        self,
        parent_features: torch.Tensor,
        child_to_parent_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Broadcast parents to children.

        Args:
            parent_features: [num_parents, dim]
            child_to_parent_idx: [num_children] - parent index for each child

        Returns:
            child_features: [num_children, dim]
        """
        child_features = parent_features[child_to_parent_idx]
        return child_features


class ConeBatchingUNet(nn.Module):
    """
    Hierarchical Cone-Based U-Net for Multi-Resolution Processing.

    CONE-BASED APPROACH:
    - Each res5 hexagon defines a cone (center + 5-ring neighbors + all descendants)
    - Cones OVERLAP - each res10 hexagon appears in ~10 different cones
    - Training: Process one cone at a time (completely independent)
    - Inference: Process all cones → weighted average by distance to cone center

    Input: Features at res10 only for hexagons in ONE cone (Markov blanket)
    Output: Reconstructed res10 for that cone + learned internal states (res5-9)

    Architecture:
        Encoder: res10 → res5 (bottom-up inference, 2 GCN hops per resolution)
        Bottleneck: res5 (highest latent, ~91 nodes per cone)
        Decoder: res5 → res10 (top-down generation, 2 GCN hops per resolution)

    Per-resolution processing ensures each layer operates only on nodes at that resolution:
    - Res10: ~1.5M nodes/cone × 2 GCN (64→64 dim)
    - Res9: ~214K nodes/cone × 2 GCN (128→128 dim)
    - Res8: ~30K nodes/cone × 2 GCN (128→128 dim)
    - Res7: ~4K nodes/cone × 2 GCN (256→256 dim)
    - Res6: ~637 nodes/cone × 2 GCN (256→256 dim)
    - Res5: ~91 nodes/cone × 2 GCN (512→512 dim)

    Loss:
        - Reconstruction: MSE(output_res10, input_res10) for this cone
        - Consistency: Parent states match aggregated children within cone
    """

    def __init__(self, config: ConeBatchingUNetConfig):
        super().__init__()
        self.config = config

        logger.info("Initializing ConeBatchingUNet (Cone-Based Architecture):")
        logger.info(f"  Input dim: {config.input_dim} (res10 only, per cone)")
        logger.info(f"  Output dim: {config.output_dim}")
        logger.info(f"  Hidden dims: {config.hidden_dims}")
        logger.info(f"  Lateral conv layers: {config.lateral_conv_layers}")
        logger.info(f"  Conv type: {config.conv_type}")

        # Resolutions (finest to coarsest)
        self.resolutions = [10, 9, 8, 7, 6, 5]

        # Input projection (res10 only)
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[10]),
            nn.GELU()
        )

        # Encoder layers (res10 → res5)
        self.encoder_lateral = nn.ModuleDict()
        self.encoder_transition = nn.ModuleDict()

        for res in self.resolutions:
            # Lateral convolutions at this resolution
            layers = []
            for i in range(config.lateral_conv_layers):
                in_dim = config.hidden_dims[res] if i == 0 else config.hidden_dims[res]
                out_dim = config.hidden_dims[res]

                layers.append(GraphConvBlock(
                    in_dim, out_dim,
                    conv_type=config.conv_type,
                    dropout=config.dropout,
                    activation=config.activation,
                    use_graph_norm=config.use_graph_norm,
                    num_heads=config.num_heads
                ))

            self.encoder_lateral[f'res{res}'] = nn.ModuleList(layers)

            # Transition to parent resolution (if not at bottleneck)
            if res > 5:
                parent_res = res - 1
                self.encoder_transition[f'res{res}_to_{parent_res}'] = nn.Sequential(
                    HierarchicalAggregation(aggregation="mean"),
                    nn.Linear(config.hidden_dims[res], config.hidden_dims[parent_res]),
                    nn.GELU()
                )

        # Decoder layers (res5 → res10)
        self.decoder_lateral = nn.ModuleDict()
        self.decoder_transition = nn.ModuleDict()

        for res in reversed(self.resolutions):
            # Transition from parent resolution (if not at bottleneck)
            if res > 5:
                parent_res = res - 1
                in_dim = config.hidden_dims[parent_res]

                # Add skip connection dimension
                if config.use_skip_connections:
                    in_dim += config.hidden_dims[res]

                self.decoder_transition[f'res{parent_res}_to_{res}'] = nn.Sequential(
                    HierarchicalBroadcast(),
                    nn.Linear(in_dim, config.hidden_dims[res]),
                    nn.GELU()
                )

            # Lateral convolutions at this resolution
            layers = []
            for i in range(config.lateral_conv_layers):
                in_dim = config.hidden_dims[res]
                out_dim = config.hidden_dims[res]

                layers.append(GraphConvBlock(
                    in_dim, out_dim,
                    conv_type=config.conv_type,
                    dropout=config.dropout,
                    activation=config.activation,
                    use_graph_norm=config.use_graph_norm,
                    num_heads=config.num_heads
                ))

            self.decoder_lateral[f'res{res}'] = nn.ModuleList(layers)

        # Output projection (res10 → output_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dims[10], config.output_dim),
            nn.Tanh()  # Normalize outputs
        )

    def encode(
        self,
        features_res10: torch.Tensor,
        spatial_edges: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        hierarchical_mappings: Dict[int, Tuple[torch.Tensor, int]],
        batch: Optional[torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Encoder: Bottom-up inference (res10 → res5).

        Args:
            features_res10: Input features [N_10, input_dim]
            spatial_edges: Dict[res] -> (edge_index, edge_weight)
            hierarchical_mappings: Dict[child_res] -> (child_to_parent_idx, num_parents)
            batch: Optional batch indices

        Returns:
            encoder_states: Dict[res] -> features [N_res, hidden_dim]
        """
        encoder_states = {}

        # Project input
        h = self.input_proj(features_res10)
        encoder_states[10] = h

        # Encoder: res10 → res5
        for res in self.resolutions:
            # Get features at this resolution
            if res < 10:
                # Features come from hierarchical transition
                h = encoder_states[res]

            # Lateral convolutions at this resolution
            edge_index, edge_weight = spatial_edges[res]
            for lateral_layer in self.encoder_lateral[f'res{res}']:
                h = lateral_layer(h, edge_index, edge_weight, batch)

            # Update encoder state
            encoder_states[res] = h

            # Aggregate to parent resolution (if not at bottleneck)
            if res > 5:
                parent_res = res - 1
                child_to_parent_idx, num_parents = hierarchical_mappings[res]

                # Aggregate children to parents
                h_parent = self.encoder_transition[f'res{res}_to_{parent_res}'][0](
                    h, child_to_parent_idx, num_parents
                )

                # Linear projection + activation
                h_parent = self.encoder_transition[f'res{res}_to_{parent_res}'][1:](h_parent)

                # Store for next iteration
                encoder_states[parent_res] = h_parent

        return encoder_states

    def decode(
        self,
        encoder_states: Dict[int, torch.Tensor],
        spatial_edges: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        hierarchical_mappings: Dict[int, Tuple[torch.Tensor, int]],
        batch: Optional[torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Decoder: Top-down generation (res5 → res10).

        Args:
            encoder_states: Encoder states from encode()
            spatial_edges: Dict[res] -> (edge_index, edge_weight)
            hierarchical_mappings: Dict[child_res] -> (child_to_parent_idx, num_parents)
            batch: Optional batch indices

        Returns:
            decoder_outputs: Dict[res] -> features [N_res, hidden_dim]
        """
        decoder_outputs = {}

        # Start from bottleneck (res5)
        h = encoder_states[5]
        decoder_outputs[5] = h

        # Decoder: res5 → res10
        for res in [5, 6, 7, 8, 9, 10]:
            # Lateral convolutions at this resolution
            edge_index, edge_weight = spatial_edges[res]
            for lateral_layer in self.decoder_lateral[f'res{res}']:
                h = lateral_layer(h, edge_index, edge_weight, batch)

            # Update decoder output
            decoder_outputs[res] = h

            # Broadcast to children (if not at finest resolution)
            if res < 10:
                child_res = res + 1
                child_to_parent_idx, num_parents = hierarchical_mappings[child_res]

                # Broadcast parents to children
                h_children = self.decoder_transition[f'res{res}_to_{child_res}'][0](
                    h, child_to_parent_idx
                )

                # Add skip connection from encoder
                if self.config.use_skip_connections:
                    h_children = torch.cat([h_children, encoder_states[child_res]], dim=-1)

                # Linear projection + activation
                h_children = self.decoder_transition[f'res{res}_to_{child_res}'][1:](h_children)

                # Store for next iteration
                h = h_children

        return decoder_outputs

    def forward(
        self,
        features_res10: torch.Tensor,
        spatial_edges: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        hierarchical_mappings: Dict[int, Tuple[torch.Tensor, int]],
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through hierarchical generative model.

        Args:
            features_res10: Input features [N_10, input_dim]
            spatial_edges: Dict[res] -> (edge_index, edge_weight)
            hierarchical_mappings: Dict[child_res] -> (child_to_parent_idx, num_parents)
            batch: Optional batch indices

        Returns:
            Dictionary with:
                - encoder_states: Dict[res] -> encoder features
                - decoder_outputs: Dict[res] -> decoder features
                - reconstruction: Reconstructed res10 [N_10, output_dim]
        """
        # Encoder: res10 → res5 (inference)
        encoder_states = self.encode(
            features_res10,
            spatial_edges,
            hierarchical_mappings,
            batch
        )

        # Decoder: res5 → res10 (generation)
        decoder_outputs = self.decode(
            encoder_states,
            spatial_edges,
            hierarchical_mappings,
            batch
        )

        # Output projection (res10 reconstruction)
        reconstruction = self.output_proj(decoder_outputs[10])

        return {
            'encoder_states': encoder_states,
            'decoder_outputs': decoder_outputs,
            'reconstruction': reconstruction
        }


def create_cone_batching_unet(
    input_dim: int = 64,
    output_dim: int = 64,
    model_size: str = "medium",
    **kwargs
) -> ConeBatchingUNet:
    """
    Factory function for creating cone-based lattice U-Net with preset sizes.

    CONE-BASED ARCHITECTURE:
    Each model instance processes ONE cone at a time. Cones overlap, so during
    inference we process all cones and use weighted averaging.

    Args:
        input_dim: Input dimension (AlphaEarth embeddings)
        output_dim: Output dimension (reconstructed embeddings)
        model_size: "small", "medium", or "large"
        **kwargs: Additional config overrides

    Returns:
        ConeBatchingUNet model configured for cone-based processing
    """
    size_presets = {
        "small": {
            "hidden_dims": {10: 32, 9: 64, 8: 64, 7: 128, 6: 128, 5: 256},
            "lateral_conv_layers": 2,
            "dropout": 0.1
        },
        "medium": {
            "hidden_dims": {10: 64, 9: 128, 8: 128, 7: 256, 6: 256, 5: 512},
            "lateral_conv_layers": 2,
            "dropout": 0.1
        },
        "large": {
            "hidden_dims": {10: 128, 9: 256, 8: 256, 7: 512, 6: 512, 5: 1024},
            "lateral_conv_layers": 3,
            "dropout": 0.2
        }
    }

    if model_size not in size_presets:
        raise ValueError(f"Unknown model_size: {model_size}. Choose from {list(size_presets.keys())}")

    preset = size_presets[model_size]

    config = ConeBatchingUNetConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        **preset,
        **kwargs
    )

    model = ConeBatchingUNet(config)

    logger.info(f"Created {model_size} ConeBatchingUNet (cone-based):")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model
