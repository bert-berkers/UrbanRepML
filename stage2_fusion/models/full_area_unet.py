import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class ModalityFusion(nn.Module):
    """Simple modality fusion with consistent dimensionality.

    NOTE (2026-03-08): The modality_weights parameter is effectively dead code.
    MultiResolutionLoader feeds a single "fused" tensor (pre-concatenated by
    stage2_fusion.concat), so modality_weights is a 1-element tensor whose
    softmax is always 1.0 — it receives zero gradient during training.
    The projection layers (Linear -> LayerNorm -> GELU) DO receive gradients
    and act as the input projection to hidden_dim.

    Decision: KEEP as-is (conservative). Removing would break the trained
    checkpoint (epoch 499, loss 1.52e-4) since the state_dict keys would change.
    When we eventually feed per-modality tensors (rather than pre-fused), the
    learnable weights will become active without any architecture change.

    TODO: When adding true multi-modality input to FullAreaUNet, update
    MultiResolutionLoader to return per-modality tensors instead of a single
    "fused" key. ModalityFusion will then work as designed.
    """
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
    Applied after H3 sparse matrix mapping between resolutions.

    Supports dimension change: in_dim -> out_dim via the linear transform.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, mapping: torch.sparse.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [num_nodes, in_dim]
            mapping: Sparse mapping matrix [target_nodes, source_nodes]
        Returns:
            Transformed features [target_nodes, out_dim]
        """
        # Apply sparse mapping first (aggregates spatially, keeps in_dim)
        mapped = torch.sparse.mm(mapping, x)
        # Then learnable transform (in_dim -> out_dim)
        transformed = self.transform(mapped)
        return transformed

class EncoderBlock(nn.Module):
    """GCNConv encoder block with optional input projection for dimension changes.

    If in_dim != out_dim, an input projection maps features before GCNConv processing.
    All GCNConv layers operate at out_dim. Residual connection projects to match.
    """
    def __init__(self, in_dim: int, out_dim: int, num_convs: int = 10):
        super().__init__()
        self.needs_projection = (in_dim != out_dim)

        if self.needs_projection:
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU()
            )

        self.convs = nn.ModuleList([
            GCNConv(out_dim, out_dim)
            for _ in range(num_convs)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _ in range(num_convs)
        ])

        self.residual = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional, passed to GCNConv)
        Returns:
            Processed features [num_nodes, out_dim]
        """
        identity = self.residual(x)

        if self.needs_projection:
            out = self.input_proj(x)
        else:
            out = x

        for conv, norm in zip(self.convs, self.norms):
            out = conv(out, edge_index, edge_weight=edge_weight)
            out = norm(out)
            out = F.gelu(out)

        out = out + identity
        return out

class DecoderBlock(nn.Module):
    """GCNConv decoder block with skip connections and optional down-projection.

    Skip connection is added before GCNConv (both at in_dim).
    If in_dim != out_dim, a down-projection maps output to out_dim after GCNConv.
    """
    def __init__(self, in_dim: int, out_dim: int, num_convs: int = 10):
        super().__init__()
        self.needs_projection = (in_dim != out_dim)

        self.convs = nn.ModuleList([
            GCNConv(in_dim, in_dim)
            for _ in range(num_convs)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(in_dim) for _ in range(num_convs)
        ])

        # Residual from combined input to output
        self.residual = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

        if self.needs_projection:
            self.down_proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim)
            )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [num_nodes, in_dim]
            skip: Skip connection from encoder [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional, passed to GCNConv)
        Returns:
            Processed features [num_nodes, out_dim]
        """
        combined = x + skip
        identity = self.residual(combined)

        out = combined
        for conv, norm in zip(self.convs, self.norms):
            out = conv(out, edge_index, edge_weight=edge_weight)
            out = norm(out)
            out = F.gelu(out)

        if self.needs_projection:
            out = self.down_proj(out)

        out = out + identity
        return out

class FullAreaUNet(nn.Module):
    """Multi-resolution U-Net for urban representation learning.

    Pyramid architecture with dimension funneling: features expand through the
    encoder (fine->coarse) and contract through the decoder (coarse->fine).

    Default dims=[64, 128, 256] at [res_fine, res_mid, res_coarse]:

        Input: 208D concat
        ModalityFusion: 208D -> 64D

        Encoder:
          enc1 (res_fine):    64D -> 64D     <- skip_A
          mapping fine->mid:  64D -> 128D
          enc2 (res_mid):    128D -> 128D    <- skip_B
          mapping mid->coarse: 128D -> 256D
          enc3 (res_coarse): 256D -> 256D    <- bottleneck

        Decoder:
          dec3 (res_coarse): 256D+skip(256D) -> 128D
          mapping coarse->mid: 128D -> 128D
          dec2 (res_mid):    128D+skip(128D) -> 64D
          mapping mid->fine:  64D -> 64D
          dec1 (res_fine):    64D+skip(64D)  -> 64D

        Output heads: all resolutions -> 64D
        Reconstruction: 64D -> 208D (res_fine only)

    Supports any 3-level resolution hierarchy (e.g. [10,9,8] or [9,8,7]).
    Resolutions are ordered finest-to-coarsest internally.
    """
    def __init__(
            self,
            feature_dims: Dict[str, int],
            dims: List[int] = None,
            num_convs: int = 10,
            device: str = "cuda",
            resolutions: Optional[list] = None,
            # Legacy args — ignored but accepted for backward compat
            hidden_dim: int = None,
            output_dim: int = None,
    ):
        super().__init__()
        self.device = device

        # Resolution configuration: finest to coarsest
        resolutions = resolutions or [10, 9, 8]
        self.resolutions = sorted(resolutions, reverse=True)  # [finest, mid, coarsest]
        assert len(self.resolutions) == 3, "FullAreaUNet requires exactly 3 resolution levels"
        self.res_fine, self.res_mid, self.res_coarse = self.resolutions

        # Dimension pyramid: fine -> mid -> coarse
        dims = dims or [64, 128, 256]
        assert len(dims) == 3, "dims must have exactly 3 values [fine, mid, coarse]"
        self.dims = dims
        dim_fine, dim_mid, dim_coarse = dims

        # Compute total input dim for reconstruction head
        self.input_dim = sum(feature_dims.values())

        # Initial fusion: input -> dim_fine
        self.fusion = ModalityFusion(feature_dims, dim_fine)

        # Encoder path (fine to coarse, dimensions expand)
        self.enc1 = EncoderBlock(dim_fine, dim_fine, num_convs)      # 64 -> 64
        self.enc2 = EncoderBlock(dim_mid, dim_mid, num_convs)        # 128 -> 128
        self.enc3 = EncoderBlock(dim_coarse, dim_coarse, num_convs)  # 256 -> 256

        # Cross-resolution mappings (encoder direction: fine -> coarse)
        self.mapping_fine_to_mid = SharedSparseMapping(dim_fine, dim_mid)       # 64 -> 128
        self.mapping_mid_to_coarse = SharedSparseMapping(dim_mid, dim_coarse)   # 128 -> 256

        # Cross-resolution mappings (decoder direction: coarse -> fine)
        self.mapping_coarse_to_mid = SharedSparseMapping(dim_mid, dim_mid)      # 128 -> 128
        self.mapping_mid_to_fine = SharedSparseMapping(dim_fine, dim_fine)       # 64 -> 64

        # Decoder path (coarse to fine, dimensions contract)
        self.dec3 = DecoderBlock(dim_coarse, dim_mid, num_convs)   # 256+256 -> 128
        self.dec2 = DecoderBlock(dim_mid, dim_fine, num_convs)     # 128+128 -> 64
        self.dec1 = DecoderBlock(dim_fine, dim_fine, num_convs)    # 64+64   -> 64

        # Output heads: project each decoder output to dim_fine for consistency loss
        # Decoder outputs: d1=dim_fine, d2=dim_fine (dec2 down-projects), d3=dim_mid (dec3 down-projects)
        self.output = nn.ModuleDict({
            str(self.res_fine): nn.Sequential(
                nn.Linear(dim_fine, dim_fine),
                nn.LayerNorm(dim_fine)
            ),
            str(self.res_mid): nn.Sequential(
                nn.Linear(dim_fine, dim_fine),
                nn.LayerNorm(dim_fine)
            ),
            str(self.res_coarse): nn.Sequential(
                nn.Linear(dim_mid, dim_fine),
                nn.LayerNorm(dim_fine)
            ),
        })

        # Reconstruction heads (from dim_fine -> input_dim)
        self.reconstructors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim_fine, dim_fine),
                nn.LayerNorm(dim_fine),
                nn.GELU(),
                nn.Linear(dim_fine, dim),
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
            mappings: Cross-resolution mappings (fine->coarse sparse matrices)
        Returns:
            embeddings: Learned embeddings per resolution (all dim_fine)
            reconstructed: Reconstructed features per modality
        """
        rf, rm, rc = self.res_fine, self.res_mid, self.res_coarse

        # Initial fusion: input_dim -> dim_fine
        x = self.fusion(features_dict)

        # Encoder path (fine to coarse)
        e1 = self.enc1(x, edge_indices[rf], edge_weights[rf])                     # 64D at res_fine
        e1_mapped = self.mapping_fine_to_mid(e1, mappings[(rf, rm)].t())           # 128D at res_mid
        e2 = self.enc2(e1_mapped, edge_indices[rm], edge_weights[rm])              # 128D at res_mid
        e2_mapped = self.mapping_mid_to_coarse(e2, mappings[(rm, rc)].t())         # 256D at res_coarse
        e3 = self.enc3(e2_mapped, edge_indices[rc], edge_weights[rc])              # 256D at res_coarse

        # Decoder path with skip connections (coarse to fine)
        d3 = self.dec3(e3, e3, edge_indices[rc], edge_weights[rc])                 # 256+256 -> 128D
        d3_mapped = self.mapping_coarse_to_mid(d3, mappings[(rm, rc)])             # 128D at res_mid
        d2 = self.dec2(d3_mapped, e2, edge_indices[rm], edge_weights[rm])          # 128+128 -> 64D
        d2_mapped = self.mapping_mid_to_fine(d2, mappings[(rf, rm)])               # 64D at res_fine
        d1 = self.dec1(d2_mapped, e1, edge_indices[rf], edge_weights[rf])          # 64+64 -> 64D

        # Generate embeddings at each resolution (all projected to dim_fine)
        # d1=64D, d2=64D (dec2 down-projected 128->64), d3=128D (dec3 down-projected 256->128)
        embeddings = {
            rf: self.output[str(rf)](d1),      # 64D -> 64D
            rm: self.output[str(rm)](d2),      # 64D -> 64D
            rc: self.output[str(rc)](d3)       # 128D -> 64D
        }

        # Generate reconstructions from finest resolution
        reconstructed = {
            name: self.reconstructors[name](embeddings[rf])
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

class FullAreaModelTrainer:
    """Trainer for the Urban U-Net model."""

    def __init__(
            self,
            model_config: dict,
            loss_weights: Dict[str, float] = None,
            city_name: str = "south_holland_threshold80",
            wandb_project: str = "urban-embedding",
            checkpoint_dir: Optional[Path] = None,
            year: str = "2022",
    ):
        logger.info("Initializing FullAreaModelTrainer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = model_config
        self.model = FullAreaUNet(**model_config, device=self.device)
        self.model.to(self.device)
        logger.info(f"Model initialized on {self.device}")

        self.loss_computer = LossComputer()
        self.loss_weights = loss_weights or {
            'reconstruction': 1.0,
            'consistency': 0.3
        }
        logger.info(f"Loss weights set: {self.loss_weights}")

        self.city_name = city_name
        self.wandb_project = wandb_project
        self.checkpoint_dir = checkpoint_dir
        self.year = year

    def _get_scheduler(self, optimizer, num_epochs: int, warmup_epochs: int):
        """CosineAnnealingWarmRestarts: tapers LR then restarts periodically.

        T_0 = num_epochs // 3 gives ~3 warm restarts over training.
        T_mult = 1 keeps each cycle the same length.
        eta_min = max_lr / 50 so LR never fully dies.
        warmup_epochs is accepted but unused (restarts serve as warmup).
        """
        max_lr = optimizer.defaults['lr']
        t_0 = max(num_epochs // 3, 1)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,
            T_mult=1,
            eta_min=max_lr / 50,
        )

    def _save_checkpoint(self, state: dict, filename: str):
        """Save versioned model checkpoint with config metadata.

        Saves as best_model_{year}_{dim}D_{date}.pt and creates a
        best_model.pt copy pointing to the latest versioned checkpoint.
        Uses shutil.copy2 instead of symlinks for Windows compatibility.
        """
        import shutil
        from datetime import date

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Enrich state with model config for reproducibility
            state['model_config'] = self.model_config
            state['loss_weights'] = self.loss_weights
            state['lr_schedule'] = 'CosineAnnealingWarmRestarts'

            # Versioned filename: best_model_{year}_{dim}D_{date}.pt
            output_dim = self.model.dims[0]  # finest resolution dim
            today = date.today().isoformat()
            versioned_name = f"best_model_{self.year}_{output_dim}D_{today}.pt"

            versioned_path = self.checkpoint_dir / versioned_name
            torch.save(state, versioned_path)
            logger.info(f"Saved versioned checkpoint to {versioned_path}")

            # Also save as best_model.pt (copy, not symlink — Windows compat)
            latest_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy2(versioned_path, latest_path)
            logger.info(f"Copied to {latest_path}")

    def train(self, features_dict: Dict[str, torch.Tensor],
              edge_indices: Dict[int, torch.Tensor],
              edge_weights: Dict[int, torch.Tensor],
              mappings: Dict[Tuple[int, int], torch.Tensor],
              num_epochs: int = 100,
              learning_rate: float = 1e-4,
              warmup_epochs: int = 10,
              patience: int = 100,
              gradient_clip: float = 1.0) -> dict:
        """Train the model and return a result dict.

        Returns
        -------
        dict with keys:
            best_embeddings : Dict[int, Tensor] — best embeddings per resolution
            best_state      : dict — checkpoint state (model weights, epoch, loss, ...)
            loss_history    : List[dict] — per-epoch loss/lr/grad_norm records
            best_epoch      : int — epoch index of best model (early stopping point)
        """
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
        best_epoch = -1
        loss_history: List[dict] = []

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

                # Record per-epoch metrics
                loss_history.append({
                    "epoch": epoch,
                    "total_loss": total_loss.item(),
                    "reconstruction_loss": losses['reconstruction_loss'].item(),
                    "consistency_loss": losses['consistency_loss'].item(),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                })

                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_epoch = epoch
                    patience_counter = 0
                    best_state = {
                        'epoch': epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'loss': best_loss,
                        'loss_history': loss_history,
                        'best_epoch': epoch,
                    }
                    best_embeddings = {k: v.detach().clone() for k, v in embeddings.items()}
                    self._save_checkpoint(best_state.copy(), "best_model.pt")
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
            # Update loss_history in best_state to the full history (not just up to best)
            best_state['loss_history'] = loss_history
            best_state['best_epoch'] = best_epoch
            self.model.load_state_dict(best_state['model_state'])
            logger.info(f"Restored best model from epoch {best_state['epoch']}")

        return {
            'best_embeddings': best_embeddings,
            'best_state': best_state,
            'loss_history': loss_history,
            'best_epoch': best_epoch,
        }
