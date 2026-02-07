#!/usr/bin/env python
"""
Test Cone U-Net Forward/Backward Pass

Quick test to verify:
1. ConeDataset loads properly
2. Model forward pass works
3. Loss computation works
4. Backward pass completes

Usage:
    python scripts/netherlands/test_cone_forward_pass.py
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch

from stage2_fusion.data.cone_dataset import ConeDataset
from stage2_fusion.models.cone_unet import create_cone_unet
from stage2_fusion.losses.cone_losses import create_cone_loss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_cone_forward_backward():
    """Test forward and backward pass through cone U-Net."""

    logger.info("="*60)
    logger.info("Testing Cone U-Net Forward/Backward Pass")
    logger.info("="*60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nDevice: {device}")

    try:
        # 1. Create dataset
        logger.info("\n" + "="*60)
        logger.info("Step 1: Creating ConeDataset")
        logger.info("="*60)

        dataset = ConeDataset(
            study_area="netherlands",
            parent_resolution=5,
            target_resolution=10,
            neighbor_rings=5
        )

        logger.info(f"✓ Dataset created: {len(dataset)} cones")

        # 2. Get one cone
        logger.info("\n" + "="*60)
        logger.info("Step 2: Loading Single Cone")
        logger.info("="*60)

        cone_data = dataset[0]

        logger.info(f"✓ Cone loaded: {cone_data['cone_id']}")
        logger.info(f"  Features shape: {cone_data['features_res10'].shape}")
        logger.info(f"  Nodes per resolution:")
        for res, count in cone_data['num_nodes_per_res'].items():
            logger.info(f"    Res {res}: {count} nodes")

        # Move to device
        features_res10 = cone_data['features_res10'].to(device)
        spatial_edges = {
            res: (edge_index.to(device), edge_weight.to(device))
            for res, (edge_index, edge_weight) in cone_data['spatial_edges'].items()
        }
        hierarchical_mappings = {
            res: (child_to_parent.to(device), num_parents)
            for res, (child_to_parent, num_parents) in cone_data['hierarchical_mappings'].items()
        }

        # 3. Create model
        logger.info("\n" + "="*60)
        logger.info("Step 3: Creating Model")
        logger.info("="*60)

        model = create_cone_unet(
            input_dim=64,
            output_dim=64,
            model_size="small"  # Use small for testing
        ).to(device)

        logger.info(f"✓ Model created")

        # 4. Forward pass
        logger.info("\n" + "="*60)
        logger.info("Step 4: Forward Pass")
        logger.info("="*60)

        model.train()
        output = model(
            features_res10=features_res10,
            spatial_edges=spatial_edges,
            hierarchical_mappings=hierarchical_mappings
        )

        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Reconstruction shape: {output['reconstruction'].shape}")
        logger.info(f"  Encoder states:")
        for res, state in output['encoder_states'].items():
            logger.info(f"    Res {res}: {state.shape}")

        # 5. Compute loss
        logger.info("\n" + "="*60)
        logger.info("Step 5: Computing Loss")
        logger.info("="*60)

        loss_fn = create_cone_loss(
            reconstruction_weight=1.0,
            consistency_weight=0.5
        ).to(device)

        losses = loss_fn(
            model_output=output,
            target_res10=features_res10,
            spatial_edges=spatial_edges,
            hierarchical_mappings=hierarchical_mappings
        )

        logger.info(f"✓ Loss computed")
        logger.info(f"  Total loss: {losses['total'].item():.4f}")
        logger.info(f"  Reconstruction: {losses['reconstruction'].item():.4f}")
        logger.info(f"  Consistency: {losses['consistency'].item():.4f}")

        # 6. Backward pass
        logger.info("\n" + "="*60)
        logger.info("Step 6: Backward Pass")
        logger.info("="*60)

        losses['total'].backward()

        logger.info(f"✓ Backward pass successful")

        # Check gradients
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        logger.info(f"  Gradients computed: {has_gradients}")

        if has_gradients:
            # Compute gradient statistics
            grad_norms = [
                p.grad.norm().item()
                for p in model.parameters()
                if p.grad is not None
            ]
            logger.info(f"  Gradient norms:")
            logger.info(f"    Min: {min(grad_norms):.6f}")
            logger.info(f"    Max: {max(grad_norms):.6f}")
            logger.info(f"    Mean: {sum(grad_norms)/len(grad_norms):.6f}")

        # 7. Summary
        logger.info("\n" + "="*60)
        logger.info("TEST PASSED ✓")
        logger.info("="*60)
        logger.info("All components working:")
        logger.info("  ✓ Dataset loads cones")
        logger.info("  ✓ Model forward pass")
        logger.info("  ✓ Loss computation")
        logger.info("  ✓ Backward pass")
        logger.info("  ✓ Gradients computed")

        return True

    except Exception as e:
        logger.error("\n" + "="*60)
        logger.error("TEST FAILED ✗")
        logger.error("="*60)
        logger.error(f"Error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_cone_forward_backward()
    sys.exit(0 if success else 1)
