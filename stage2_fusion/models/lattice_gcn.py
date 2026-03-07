"""
LatticeGCN: Simple GCN on the H3 hexagonal lattice.

A graph convolutional network operating on the res9 hexagonal lattice graph.
Uses self-supervised reconstruction loss (encode then decode back to input)
to learn fused embeddings.

This is a stepping stone toward the full UNet architecture -- it validates
that explicit message-passing improves over ring aggregation's k-ring averaging.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class LatticeGCN(nn.Module):
    """Simple GCN encoder-decoder on the H3 hexagonal lattice.

    Architecture:
        Encoder: input_dim -> hidden_dim -> ... -> embedding_dim (GCNConv layers)
        Decoder: embedding_dim -> hidden_dim -> input_dim (linear layers)

    The encoder output is the fused embedding we save. The decoder is used
    only during training for the reconstruction loss.

    Args:
        input_dim: Dimension of input node features.
        hidden_dim: Hidden layer dimension.
        embedding_dim: Output embedding dimension.
        num_layers: Number of GCNConv layers in the encoder.
        dropout: Dropout rate between layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # --- Encoder (GCNConv layers) ---
        encoder_layers = []
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim
            out_dim = embedding_dim if i == num_layers - 1 else hidden_dim
            encoder_layers.append(GCNConv(in_dim, out_dim))
        self.encoder = nn.ModuleList(encoder_layers)

        # --- Decoder (linear layers, no graph convolution) ---
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode node features through GCN layers.

        Args:
            x: Node feature matrix [N, input_dim].
            edge_index: Graph connectivity [2, E].

        Returns:
            Embedding matrix [N, embedding_dim].
        """
        for i, conv in enumerate(self.encoder):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings back to input space.

        Args:
            z: Embedding matrix [N, embedding_dim].

        Returns:
            Reconstructed features [N, input_dim].
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode.

        Args:
            x: Node feature matrix [N, input_dim].
            edge_index: Graph connectivity [2, E].

        Returns:
            Tuple of (embedding, reconstruction).
        """
        z = self.encode(x, edge_index)
        x_hat = self.decode(z)
        return z, x_hat
