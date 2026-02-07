"""
Base UNet Abstract Class
========================

Abstract base class for all UNet variants in the urban embedding framework.
Provides common functionality and interfaces for different graph types.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BaseUNetConfig:
    """Base configuration for UNet models."""
    input_dim: int
    hidden_dim: int = 128
    output_dim: int = 64
    num_layers: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    use_skip_connections: bool = True
    use_batch_norm: bool = False
    use_layer_norm: bool = True


class BaseUNet(nn.Module, ABC):
    """
    Abstract base class for UNet architectures.

    Provides common encoder-decoder structure with customizable components
    for different graph types (accessibility, hexagonal lattice, etc.).
    """

    def __init__(self, config: BaseUNetConfig):
        """
        Initialize base UNet.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config

        # Build architecture
        self.encoder = self.build_encoder()
        self.bottleneck = self.build_bottleneck()
        self.decoder = self.build_decoder()

        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)

        # Store intermediate activations for multi-resolution extraction
        self.encoder_outputs = []
        self.decoder_outputs = []

        # Activation function
        self.activation = self._get_activation(config.activation)

        logger.info(f"Initialized {self.__class__.__name__} with {self.count_parameters():,} parameters")

    @abstractmethod
    def build_encoder(self) -> nn.Module:
        """
        Build encoder network.

        Returns:
            Encoder module
        """
        pass

    @abstractmethod
    def build_bottleneck(self) -> nn.Module:
        """
        Build bottleneck layer.

        Returns:
            Bottleneck module
        """
        pass

    @abstractmethod
    def build_decoder(self) -> nn.Module:
        """
        Build decoder network.

        Returns:
            Decoder module
        """
        pass

    @abstractmethod
    def process_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
                     edge_weight: Optional[torch.Tensor] = None,
                     batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input through graph convolutions.

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_weight: Optional edge weights
            batch: Optional batch indices

        Returns:
            Processed features
        """
        pass

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through UNet.

        Args:
            x: Input features [N, input_dim]
            edge_index: Edge connectivity [2, E]
            edge_weight: Optional edge weights [E]
            batch: Optional batch indices [N]

        Returns:
            Dictionary with embeddings and intermediate outputs
        """
        # Clear previous outputs
        self.encoder_outputs = []
        self.decoder_outputs = []

        # Encoder pass
        encoded = self.encode(x, edge_index, edge_weight, batch)

        # Bottleneck
        bottleneck = self.process_bottleneck(encoded, edge_index, edge_weight, batch)

        # Decoder pass
        decoded = self.decode(bottleneck, edge_index, edge_weight, batch)

        # Output projection
        embeddings = self.output_projection(decoded)

        # Apply final activation if configured
        if hasattr(self, 'output_activation'):
            embeddings = self.output_activation(embeddings)

        return {
            'embeddings': embeddings,
            'encoder_outputs': self.encoder_outputs,
            'decoder_outputs': self.decoder_outputs,
            'bottleneck': bottleneck
        }

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor,
              edge_weight: Optional[torch.Tensor] = None,
              batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode input through encoder layers.

        Stores intermediate outputs for skip connections and multi-resolution extraction.
        """
        x = self.process_graph(x, edge_index, edge_weight, batch)

        # Process through encoder layers
        for layer in self.encoder:
            x = layer(x, edge_index, edge_weight, batch) if hasattr(layer, 'forward') else x
            self.encoder_outputs.append(x.clone())

        return x

    def process_bottleneck(self, x: torch.Tensor, edge_index: torch.Tensor,
                          edge_weight: Optional[torch.Tensor] = None,
                          batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process through bottleneck layer."""
        if self.bottleneck is not None:
            return self.bottleneck(x, edge_index, edge_weight, batch)
        return x

    def decode(self, x: torch.Tensor, edge_index: torch.Tensor,
              edge_weight: Optional[torch.Tensor] = None,
              batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode through decoder layers.

        Uses skip connections from encoder if configured.
        Stores intermediate outputs for multi-resolution extraction.
        """
        # Process through decoder layers
        for i, layer in enumerate(self.decoder):
            # Add skip connection if configured
            if self.config.use_skip_connections and i < len(self.encoder_outputs):
                encoder_features = self.encoder_outputs[-(i+1)]  # Reverse order
                x = torch.cat([x, encoder_features], dim=-1)

            x = layer(x, edge_index, edge_weight, batch) if hasattr(layer, 'forward') else x
            self.decoder_outputs.append(x.clone())

        return x

    def extract_embeddings(self, resolution_level: int = -1) -> torch.Tensor:
        """
        Extract embeddings from specific decoder resolution level.

        Args:
            resolution_level: Decoder level to extract from (-1 for final)

        Returns:
            Embeddings from specified level
        """
        if not self.decoder_outputs:
            raise ValueError("No decoder outputs available. Run forward pass first.")

        if resolution_level == -1 or resolution_level >= len(self.decoder_outputs):
            return self.decoder_outputs[-1]

        return self.decoder_outputs[resolution_level]

    def extract_multi_resolution_embeddings(self) -> Dict[int, torch.Tensor]:
        """
        Extract embeddings from all decoder levels.

        Returns:
            Dictionary mapping resolution to embeddings
        """
        if not self.decoder_outputs:
            raise ValueError("No decoder outputs available. Run forward pass first.")

        embeddings = {}
        for i, output in enumerate(self.decoder_outputs):
            # Map decoder level to H3 resolution (customize as needed)
            # This is a simple mapping - adjust based on architecture
            resolution = 10 - i  # Example: level 0 -> res 10, level 1 -> res 9, etc.
            if resolution >= 5:  # Only store resolutions 5-10
                embeddings[resolution] = output

        return embeddings

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(name, nn.GELU())

    def reset_intermediate_outputs(self):
        """Reset stored intermediate outputs."""
        self.encoder_outputs = []
        self.decoder_outputs = []

    def get_embedding_dims(self) -> Dict[str, int]:
        """
        Get embedding dimensions at different stages.

        Returns:
            Dictionary of stage -> dimension
        """
        return {
            'input': self.config.input_dim,
            'hidden': self.config.hidden_dim,
            'output': self.config.output_dim,
            'num_encoder_levels': len(self.encoder_outputs) if self.encoder_outputs else 0,
            'num_decoder_levels': len(self.decoder_outputs) if self.decoder_outputs else 0
        }