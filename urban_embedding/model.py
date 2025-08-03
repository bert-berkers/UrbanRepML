import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, Tuple, Optional
import wandb
from datetime import datetime
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class ModalityFusion(nn.Module):
    """Simple modality fusion with consistent dimensionality"""
    def __init__(self, modality_dims: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for name, dim in modality_dims.items()
        })

        # Simple learnable weights
        self.modality_weights = nn.Parameter(torch.ones(len(modality_dims)) / len(modality_dims))

    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected = {}
        for name, features in features_dict.items():
            features = torch.nan_to_num(features, nan=0.0)
            proj = self.projections[name](features)
            projected[name] = F.normalize(proj, p=2, dim=-1, eps=1e-8)

        # Simple weighted sum with softmax
        weights = F.softmax(self.modality_weights, dim=0)
        fused = sum(proj * w for (_, proj), w in zip(projected.items(), weights))
        return F.normalize(fused, p=2, dim=-1, eps=1e-8)

class SharedSparseMapping(nn.Module):
    """Learnable transformation for cross-resolution mapping.
    Applied after H3 sparse matrix mapping between resolutions."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, mapping: torch.sparse.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [num_nodes, hidden_dim]
            mapping: Sparse mapping matrix [target_nodes, source_nodes]
        Returns:
            Transformed features [target_nodes, hidden_dim]
        """
        # Apply sparse mapping first
        mapped = torch.sparse.mm(mapping, x)
        # Then learnable transform
        transformed = self.transform(mapped)
        # Final normalization
        return F.normalize(transformed, p=2, dim=-1, eps=1e-8)

class EncoderBlock(nn.Module):
    """GCN encoder block with residual connections and layer normalization."""
    def __init__(self, hidden_dim: int, num_convs: int = 4):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
            for _ in range(num_convs)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_convs)
        ])

        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [num_nodes, hidden_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]
        Returns:
            Processed features [num_nodes, hidden_dim]
        """
        identity = self.residual(x)

        out = x
        for conv, norm in zip(self.convs, self.norms):
            out = conv(out, edge_index, edge_weight)
            out = norm(out)
            out = F.gelu(out)

        out = out + identity
        return F.normalize(out, p=2, dim=-1, eps=1e-8)

class DecoderBlock(nn.Module):
    """GCN decoder block with skip connections from encoder."""
    def __init__(self, hidden_dim: int, num_convs: int = 4):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
            for _ in range(num_convs)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_convs)
        ])

        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [num_nodes, hidden_dim]
            skip: Skip connection from encoder [num_nodes, hidden_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]
        Returns:
            Processed features [num_nodes, hidden_dim]
        """
        combined = x + skip
        identity = self.residual(combined)

        out = combined
        for conv, norm in zip(self.convs, self.norms):
            out = conv(out, edge_index, edge_weight)
            out = norm(out)
            out = F.gelu(out)

        out = out + identity
        return F.normalize(out, p=2, dim=-1, eps=1e-8)

class UrbanUNet(nn.Module):
    """Multi-resolution U-Net for urban representation learning."""
    def __init__(
            self,
            feature_dims: Dict[str, int],
            hidden_dim: int = 128,
            output_dim: int = 32,
            num_convs: int = 4,
            device: str = "cuda"
    ):
        super().__init__()
        self.device = device

        # Initial fusion
        self.fusion = ModalityFusion(feature_dims, hidden_dim)

        # Shared mapping transformation
        self.mapping_transform = SharedSparseMapping(hidden_dim)

        # Encoder path with consistent hidden_dim
        self.enc1 = EncoderBlock(hidden_dim, num_convs)
        self.enc2 = EncoderBlock(hidden_dim, num_convs)
        self.enc3 = EncoderBlock(hidden_dim, num_convs)

        # Decoder path with consistent hidden_dim
        self.dec3 = DecoderBlock(hidden_dim, num_convs)
        self.dec2 = DecoderBlock(hidden_dim, num_convs)
        self.dec1 = DecoderBlock(hidden_dim, num_convs)

        # Final projection to output dimension
        self.output = nn.ModuleDict({
            str(res): nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            ) for res in [8, 9, 10]
        })

        # Reconstruction heads
        self.reconstructors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
                nn.LayerNorm(dim)
            ) for name, dim in feature_dims.items()
        })

    def forward(
            self,
            features_dict: Dict[str, torch.Tensor],
            edge_indices: Dict[int, torch.Tensor],
            edge_weights: Dict[int, torch.Tensor],
            mappings: Dict[Tuple[int, int], torch.Tensor]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            features_dict: Input features per modality
            edge_indices: Graph connectivity per resolution
            edge_weights: Edge weights per resolution
            mappings: Cross-resolution mappings (fine->coarse)
        Returns:
            embeddings: Learned embeddings per resolution
            reconstructed: Reconstructed features per modality
        """
        # Initial fusion
        x = self.fusion(features_dict)

        # Encoder path (fine to coarse)
        e1 = self.enc1(x, edge_indices[10], edge_weights[10])
        e1_mapped = self.mapping_transform(e1, mappings[(10, 9)].t())

        e2 = self.enc2(e1_mapped, edge_indices[9], edge_weights[9])
        e2_mapped = self.mapping_transform(e2, mappings[(9, 8)].t())

        e3 = self.enc3(e2_mapped, edge_indices[8], edge_weights[8])

        # Decoder path with skip connections (coarse to fine)
        d3 = self.dec3(e3, e3, edge_indices[8], edge_weights[8])
        d3_mapped = self.mapping_transform(d3, mappings[(9, 8)])  # Map 8->9

        d2 = self.dec2(d3_mapped, e2, edge_indices[9], edge_weights[9])
        d2_mapped = self.mapping_transform(d2, mappings[(10, 9)])  # Map 9->10

        d1 = self.dec1(d2_mapped, e1, edge_indices[10], edge_weights[10])

        # Generate embeddings
        embeddings = {
            10: F.normalize(self.output['10'](d1), p=2, dim=-1, eps=1e-8),
            9: F.normalize(self.output['9'](d2), p=2, dim=-1, eps=1e-8),
            8: F.normalize(self.output['8'](d3), p=2, dim=-1, eps=1e-8)
        }

        # Generate reconstructions from finest resolution
        reconstructed = {
            name: F.normalize(self.reconstructors[name](embeddings[10]), p=2, dim=-1, eps=1e-8)
            for name in features_dict.keys()
        }

        return embeddings, reconstructed

class LossComputer:
    """Simplified loss computation focusing on balanced MSE across scales"""
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def compute_losses(
            self,
            embeddings: Dict[int, torch.Tensor],
            reconstructed: Dict[str, torch.Tensor],
            features_dict: Dict[str, torch.Tensor],
            mappings: Dict[Tuple[int, int], torch.Tensor],
            loss_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        # 1. Reconstruction loss (fine scale only)
        recon_losses = {}
        for name, pred in reconstructed.items():
            target = features_dict[name]
            recon_losses[name] = F.mse_loss(
                F.normalize(pred, p=2, dim=1, eps=1e-8),
                F.normalize(target, p=2, dim=1, eps=1e-8)
            )
        total_recon_loss = sum(recon_losses.values()) * loss_weights['reconstruction']

        # 2. Consistency loss between scales (with equal weighting)
        consistency_losses = {}
        for (res_fine, res_coarse), mapping in mappings.items():
            # Map fine embeddings to coarse resolution
            fine_mapped = torch.sparse.mm(mapping.t(), embeddings[res_fine])
            fine_mapped = F.normalize(fine_mapped, p=2, dim=1, eps=1e-8)
            coarse_emb = F.normalize(embeddings[res_coarse], p=2, dim=1, eps=1e-8)

            # Calculate MSE for this scale pair
            consistency_losses[(res_fine, res_coarse)] = F.mse_loss(fine_mapped, coarse_emb)

        # Average the consistency losses (equal weight per scale transition)
        total_consistency_loss = (sum(consistency_losses.values()) / len(consistency_losses)) * loss_weights['consistency']

        return {
            'total_loss': total_recon_loss + total_consistency_loss,
            'reconstruction_loss': total_recon_loss,
            'consistency_loss': total_consistency_loss,
            **{f'recon_loss_{name}': loss for name, loss in recon_losses.items()},
            **{f'consistency_loss_{k[0]}_{k[1]}': v for k, v in consistency_losses.items()}
        }

class UrbanModelTrainer:
    """Trainer for the Urban U-Net model."""

    def __init__(
            self,
            model_config: dict,
            loss_weights: Dict[str, float] = None,
            city_name: str = "south_holland_threshold80",
            wandb_project: str = "urban-embedding",
            checkpoint_dir: Optional[Path] = None
    ):
        logger.info("Initializing UrbanModelTrainer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UrbanUNet(**model_config, device=self.device)
        self.model.to(self.device)
        logger.info(f"Model initialized on {self.device}")

        self.loss_computer = LossComputer()
        self.loss_weights = loss_weights or {
            'reconstruction': 1.0,
            'consistency': 1.0
        }
        logger.info(f"Loss weights set: {self.loss_weights}")

        self.city_name = city_name
        self.wandb_project = wandb_project
        self.checkpoint_dir = checkpoint_dir

    def _get_scheduler(self, optimizer, num_epochs: int, warmup_epochs: int):
        """Get learning rate scheduler with warmup."""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            return 0.5 * (1 + np.cos(
                np.pi * float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
            ))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _save_checkpoint(self, state: dict, filename: str):
        """Save model checkpoint."""
        if self.checkpoint_dir:
            checkpoint_path = self.checkpoint_dir / filename
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self, features_dict: Dict[str, torch.Tensor],
              edge_indices: Dict[int, torch.Tensor],
              edge_weights: Dict[int, torch.Tensor],
              mappings: Dict[Tuple[int, int], torch.Tensor],
              num_epochs: int = 100,
              learning_rate: float = 1e-4,
              warmup_epochs: int = 10,
              patience: int = 100,
              gradient_clip: float = 1.0):

        logger.info(f"Starting training with lr={learning_rate}, epochs={num_epochs}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )

        scheduler = self._get_scheduler(optimizer, num_epochs, warmup_epochs)
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        best_embeddings = None

        for epoch in tqdm(range(num_epochs), desc="Training"):
            self.model.train()
            optimizer.zero_grad()

            try:
                embeddings, reconstructed = self.model(
                    features_dict, edge_indices, edge_weights, mappings
                )

                losses = self.loss_computer.compute_losses(
                    embeddings=embeddings,
                    reconstructed=reconstructed,
                    features_dict=features_dict,
                    mappings=mappings,
                    loss_weights=self.loss_weights
                )

                total_loss = losses['total_loss']

                if torch.isnan(total_loss):
                    logger.warning(f"NaN loss detected at epoch {epoch}")
                    continue

                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=gradient_clip
                )

                optimizer.step()
                scheduler.step()

                # Log gradient norms for monitoring
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5

                wandb.log({
                    'epoch': epoch,
                    'learning_rate': scheduler.get_last_lr()[0],
                    **{k: v.item() for k, v in losses.items()},
                    'grad_norm': grad_norm,
                    'patience_counter': patience_counter
                })

                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    patience_counter = 0
                    best_state = {
                        'epoch': epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'loss': best_loss
                    }
                    best_embeddings = {k: v.detach().clone() for k, v in embeddings.items()}
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch} epochs")
                    break

            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                logger.error("Traceback:", exc_info=True)
                break

        if best_state:
            self.model.load_state_dict(best_state['model_state'])
            logger.info(f"Restored best model from epoch {best_state['epoch']}")

        return best_embeddings, best_state
