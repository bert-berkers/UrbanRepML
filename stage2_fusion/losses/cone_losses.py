"""
Cone-Based Hierarchical Loss Functions

Implements loss functions for hierarchical generative cone model:
1. Reconstruction Loss: MSE between reconstructed and observed res10
2. Consistency Loss: Parent states match aggregated children
3. Optional: Smoothness and sparsity regularization
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class ConeReconstructionLoss(nn.Module):
    """
    Reconstruction loss for resolution 10 (Markov blanket).

    Forces model to accurately reconstruct observations from learned hierarchy.
    """

    def __init__(self, loss_type: str = "mse"):
        """
        Initialize reconstruction loss.

        Args:
            loss_type: "mse", "mae", or "huber"
        """
        super().__init__()
        self.loss_type = loss_type

        logger.info(f"Initialized ConeReconstructionLoss (type={loss_type})")

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            reconstructed: Reconstructed features [N, D]
            target: Target features [N, D]
            mask: Optional mask for valid nodes [N]

        Returns:
            Reconstruction loss (scalar)
        """
        if mask is not None:
            reconstructed = reconstructed[mask]
            target = target[mask]

        if self.loss_type == "mse":
            loss = F.mse_loss(reconstructed, target)
        elif self.loss_type == "mae":
            loss = F.l1_loss(reconstructed, target)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(reconstructed, target)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


class ConeConsistencyLoss(nn.Module):
    """
    Hierarchical consistency loss.

    Ensures parent internal states are consistent with aggregated children:
        L = Σ_R MSE(aggregate(children_R), parent_{R-1})

    This enforces hierarchical coherence and prevents resolution layers
    from learning disconnected representations.
    """

    def __init__(self, aggregation: str = "mean"):
        """
        Initialize consistency loss.

        Args:
            aggregation: "mean" or "sum" for child→parent aggregation
        """
        super().__init__()
        self.aggregation = aggregation

        logger.info(f"Initialized ConeConsistencyLoss (aggregation={aggregation})")

    def aggregate_children_to_parent(
        self,
        child_features: torch.Tensor,
        child_to_parent_idx: torch.Tensor,
        num_parents: int
    ) -> torch.Tensor:
        """
        Aggregate children features to parents.

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
            from torch_scatter import scatter_add
            parent_features = scatter_add(
                child_features,
                child_to_parent_idx,
                dim=0,
                dim_size=num_parents
            )
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return parent_features

    def forward(
        self,
        encoder_states: Dict[int, torch.Tensor],
        hierarchical_mappings: Dict[int, Tuple[torch.Tensor, int]]
    ) -> torch.Tensor:
        """
        Compute consistency loss across all resolution pairs.

        Args:
            encoder_states: Dict[res] -> encoder features [N_res, dim]
            hierarchical_mappings: Dict[child_res] -> (child_to_parent_idx, num_parents)

        Returns:
            Consistency loss (scalar)
        """
        total_loss = 0.0
        num_pairs = 0

        # For each child→parent resolution pair
        for child_res in [6, 7, 8, 9, 10]:
            parent_res = child_res - 1

            if child_res not in encoder_states or parent_res not in encoder_states:
                continue

            if child_res not in hierarchical_mappings:
                continue

            # Get encoder states
            child_state = encoder_states[child_res]
            parent_state = encoder_states[parent_res]

            # Get hierarchical mapping
            child_to_parent_idx, num_parents = hierarchical_mappings[child_res]

            # Aggregate children to parents
            children_aggregated = self.aggregate_children_to_parent(
                child_state,
                child_to_parent_idx,
                num_parents
            )

            # Consistency loss: aggregated children should match parent state
            loss = F.mse_loss(children_aggregated, parent_state)
            total_loss += loss
            num_pairs += 1

        # Average over resolution pairs
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
        else:
            total_loss = torch.tensor(0.0, device=encoder_states[10].device)

        return total_loss


class ConeSmoothnessLoss(nn.Module):
    """
    Smoothness regularization for latent representations.

    Encourages spatial neighbors at the same resolution to have similar features.
    """

    def __init__(self, weight_by_distance: bool = True):
        """
        Initialize smoothness loss.

        Args:
            weight_by_distance: Weight loss by edge distance
        """
        super().__init__()
        self.weight_by_distance = weight_by_distance

    def forward(
        self,
        features: Dict[int, torch.Tensor],
        spatial_edges: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        resolutions: list = [5, 6, 7, 8, 9, 10]
    ) -> torch.Tensor:
        """
        Compute smoothness loss across spatial neighbors.

        Args:
            features: Dict[res] -> features [N_res, dim]
            spatial_edges: Dict[res] -> (edge_index, edge_weight)
            resolutions: Resolutions to apply smoothness loss

        Returns:
            Smoothness loss (scalar)
        """
        total_loss = 0.0
        num_resolutions = 0

        for res in resolutions:
            if res not in features or res not in spatial_edges:
                continue

            feat = features[res]
            edge_index, edge_weight = spatial_edges[res]

            # Get source and target node features
            src_features = feat[edge_index[0]]
            tgt_features = feat[edge_index[1]]

            # L2 distance between neighbors
            diff = src_features - tgt_features
            distances = (diff ** 2).sum(dim=-1)

            # Weight by edge distance if specified
            if self.weight_by_distance and edge_weight is not None:
                distances = distances * edge_weight

            # Average smoothness loss for this resolution
            loss = distances.mean()
            total_loss += loss
            num_resolutions += 1

        # Average over resolutions
        if num_resolutions > 0:
            total_loss = total_loss / num_resolutions
        else:
            total_loss = torch.tensor(0.0, device=features[10].device)

        return total_loss


class ConeHierarchicalLoss(nn.Module):
    """
    Combined loss for hierarchical generative cone model.

    Total loss = reconstruction + consistency + regularization

    Components:
    1. Reconstruction: Reconstruct res10 (Markov blanket)
    2. Consistency: Parent states match aggregated children
    3. Smoothness (optional): Spatial smoothness regularization
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        consistency_weight: float = 0.5,
        smoothness_weight: float = 0.0,
        reconstruction_type: str = "mse",
        consistency_aggregation: str = "mean"
    ):
        """
        Initialize hierarchical loss.

        Args:
            reconstruction_weight: Weight for reconstruction loss
            consistency_weight: Weight for consistency loss
            smoothness_weight: Weight for smoothness regularization
            reconstruction_type: "mse", "mae", or "huber"
            consistency_aggregation: "mean" or "sum"
        """
        super().__init__()

        self.recon_weight = reconstruction_weight
        self.consist_weight = consistency_weight
        self.smooth_weight = smoothness_weight

        # Loss components
        self.reconstruction_loss = ConeReconstructionLoss(reconstruction_type)
        self.consistency_loss = ConeConsistencyLoss(consistency_aggregation)

        if smoothness_weight > 0:
            self.smoothness_loss = ConeSmoothnessLoss()
        else:
            self.smoothness_loss = None

        logger.info("Initialized ConeHierarchicalLoss:")
        logger.info(f"  Reconstruction weight: {reconstruction_weight}")
        logger.info(f"  Consistency weight: {consistency_weight}")
        logger.info(f"  Smoothness weight: {smoothness_weight}")

    def forward(
        self,
        model_output: Dict,
        target_res10: torch.Tensor,
        spatial_edges: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        hierarchical_mappings: Dict[int, Tuple[torch.Tensor, int]],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            model_output: Output from MultiResolutionConeUNet
                - reconstruction: [N_10, dim]
                - encoder_states: Dict[res] -> features
                - decoder_outputs: Dict[res] -> features
            target_res10: Target features at res10 [N_10, dim]
            spatial_edges: Dict[res] -> (edge_index, edge_weight)
            hierarchical_mappings: Dict[child_res] -> (child_to_parent_idx, num_parents)
            mask: Optional mask for valid nodes

        Returns:
            Dictionary with:
                - total: Total loss
                - reconstruction: Reconstruction loss
                - consistency: Consistency loss
                - smoothness: Smoothness loss (if enabled)
        """
        losses = {}

        # 1. Reconstruction loss (res10 only)
        loss_recon = self.reconstruction_loss(
            model_output['reconstruction'],
            target_res10,
            mask
        )
        losses['reconstruction'] = loss_recon

        # 2. Consistency loss (all resolutions)
        loss_consist = self.consistency_loss(
            model_output['encoder_states'],
            hierarchical_mappings
        )
        losses['consistency'] = loss_consist

        # 3. Smoothness regularization (optional)
        if self.smoothness_loss is not None and self.smooth_weight > 0:
            loss_smooth = self.smoothness_loss(
                model_output['encoder_states'],
                spatial_edges
            )
            losses['smoothness'] = loss_smooth
        else:
            losses['smoothness'] = torch.tensor(0.0, device=target_res10.device)

        # 4. Total loss
        loss_total = (
            self.recon_weight * loss_recon
            + self.consist_weight * loss_consist
            + self.smooth_weight * losses['smoothness']
        )
        losses['total'] = loss_total

        return losses


def create_cone_loss(
    reconstruction_weight: float = 1.0,
    consistency_weight: float = 0.5,
    smoothness_weight: float = 0.0,
    **kwargs
) -> ConeHierarchicalLoss:
    """
    Factory function for creating cone loss.

    Args:
        reconstruction_weight: Weight for reconstruction (typically 1.0)
        consistency_weight: Weight for consistency (typically 0.3-0.7)
        smoothness_weight: Weight for smoothness (typically 0.0-0.1)
        **kwargs: Additional config

    Returns:
        ConeHierarchicalLoss instance
    """
    loss_fn = ConeHierarchicalLoss(
        reconstruction_weight=reconstruction_weight,
        consistency_weight=consistency_weight,
        smoothness_weight=smoothness_weight,
        **kwargs
    )

    return loss_fn
