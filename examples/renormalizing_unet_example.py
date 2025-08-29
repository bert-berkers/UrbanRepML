"""
Example Usage: Renormalizing Hierarchical U-Net for Urban Representation Learning

This example demonstrates how to use the renormalizing U-Net architecture
inspired by Friston et al.'s renormalizing generative models.

Features:
- H3 resolutions 5-10 (sustainability → liveability)
- Upward accumulation with momentum-based batching
- Downward pass-through with direct updates
- Simple MSE losses: reconstruction at res 10 + consistency between levels
- No Active Inference - pure transmission architecture
"""

import sys
from pathlib import Path
import logging
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from urban_embedding.renormalizing_pipeline import RenormalizingUrbanPipeline, create_renormalizing_config_preset
from urban_embedding.renormalizing_unet import create_renormalizing_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('renormalizing_example.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_basic_example():
    """Run basic renormalizing U-Net example."""
    
    logger.info("🚀 Starting Renormalizing U-Net Example")
    logger.info("=" * 60)
    
    # Create configuration
    config = create_renormalizing_config_preset("default")
    
    # Log configuration details
    logger.info("Configuration:")
    logger.info(f"  City: {config['city_name']}")
    logger.info(f"  Resolutions: 5-10 (6 levels)")
    logger.info(f"  Hidden dim: {config['model']['hidden_dim']}")
    logger.info(f"  Output dim: {config['model']['output_dim']}")
    logger.info(f"  Renormalizing mode: {config['renormalizing']['accumulation_mode']}")
    logger.info(f"  Normalization: {config['renormalizing']['normalization_type']}")
    logger.info(f"  Upward momentum: {config['renormalizing']['upward_momentum']}")
    
    try:
        # Initialize pipeline
        logger.info("\n📊 Initializing Pipeline...")
        pipeline = RenormalizingUrbanPipeline(config)
        
        # Run the pipeline
        logger.info("\n🔄 Running Renormalizing Pipeline...")
        embeddings_by_res = pipeline.run()
        
        # Display results
        logger.info("\n✅ Pipeline Completed Successfully!")
        logger.info("Results Summary:")
        logger.info("-" * 40)
        
        for res in sorted(embeddings_by_res.keys()):
            emb_df = embeddings_by_res[res]
            logger.info(f"Resolution {res}: {emb_df.shape[0]} cells, {emb_df.shape[1]} dimensions")
            logger.info(f"  Embedding range: [{emb_df.values.min():.3f}, {emb_df.values.max():.3f}]")
            logger.info(f"  Mean norm: {np.linalg.norm(emb_df.values, axis=1).mean():.3f}")
        
        return embeddings_by_res
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return None


def run_fast_example():
    """Run fast example with reduced parameters for testing."""
    
    logger.info("🚀 Starting Fast Renormalizing U-Net Example")
    logger.info("=" * 60)
    
    # Create fast configuration
    config = create_renormalizing_config_preset("fast")
    
    logger.info("Fast Configuration:")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")
    logger.info(f"  Hidden dim: {config['model']['hidden_dim']}")
    logger.info(f"  Patience: {config['training']['patience']}")
    
    try:
        # Initialize and run pipeline
        pipeline = RenormalizingUrbanPipeline(config)
        embeddings_by_res = pipeline.run()
        
        logger.info("\n✅ Fast Example Completed!")
        return embeddings_by_res
        
    except Exception as e:
        logger.error(f"❌ Fast example failed: {str(e)}")
        return None


def run_architecture_test():
    """Test the renormalizing architecture without full pipeline."""
    
    logger.info("🧪 Testing Renormalizing Architecture")
    logger.info("=" * 50)
    
    from urban_embedding.renormalizing_unet import RenormalizingUrbanUNet, RenormalizingLossComputer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Mock configuration
    feature_dims = {
        'aerial_alphaearth': 64,
        'gtfs': 32,
        'roadnetwork': 32,
        'poi': 32
    }
    
    renorm_config = create_renormalizing_config(
        accumulation_mode="grouped",
        normalization_type="layer",
        upward_momentum=0.9
    )
    
    try:
        # Initialize model
        model = RenormalizingUrbanUNet(
            feature_dims=feature_dims,
            hidden_dim=64,  # Smaller for testing
            output_dim=16,
            num_convs=2,
            renorm_config=renorm_config,
            device=device
        )
        
        logger.info(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create mock data
        batch_size = 100
        
        mock_features = {
            name: torch.randn(batch_size, dim).to(device)
            for name, dim in feature_dims.items()
        }
        
        # Mock edge data for all resolutions
        mock_edge_indices = {}
        mock_edge_weights = {}
        
        for res in [5, 6, 7, 8, 9, 10]:
            # Simple linear graph for testing
            num_edges = batch_size - 1
            edge_indices = torch.stack([
                torch.arange(num_edges),
                torch.arange(1, batch_size)
            ]).to(device)
            edge_weights = torch.ones(num_edges).to(device)
            
            mock_edge_indices[res] = edge_indices
            mock_edge_weights[res] = edge_weights
        
        # Mock mappings (identity for simplicity)
        mock_mappings = {}
        for i in range(5):
            fine_res = 10 - i
            coarse_res = 9 - i
            
            # Identity mapping for testing
            indices = torch.stack([torch.arange(batch_size), torch.arange(batch_size)])
            values = torch.ones(batch_size)
            mapping = torch.sparse_coo_tensor(indices, values, (batch_size, batch_size)).to(device)
            
            mock_mappings[(fine_res, coarse_res)] = mapping
        
        # Test forward pass
        logger.info("🔄 Testing forward pass...")
        with torch.no_grad():
            embeddings, reconstructed = model(
                mock_features, 
                mock_edge_indices, 
                mock_edge_weights, 
                mock_mappings
            )
        
        # Test loss computation
        logger.info("📊 Testing loss computation...")
        loss_computer = RenormalizingLossComputer()
        losses = loss_computer.compute_losses(
            embeddings=embeddings,
            reconstructed=reconstructed,
            features_dict=mock_features,
            mappings=mock_mappings,
            loss_weights={'reconstruction': 1.0, 'consistency': 2.0}
        )
        
        logger.info("✅ Architecture test completed successfully!")
        logger.info("Results:")
        for res, emb in embeddings.items():
            logger.info(f"  Resolution {res}: {emb.shape}")
        
        logger.info("Losses:")
        for name, loss in losses.items():
            if isinstance(loss, torch.Tensor):
                logger.info(f"  {name}: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Architecture test failed: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return False


def compare_with_base_model():
    """Compare renormalizing model with base UrbanUNet."""
    
    logger.info("🔄 Comparing Renormalizing vs Base Model")
    logger.info("=" * 50)
    
    from urban_embedding.model import UrbanUNet
    from urban_embedding.renormalizing_unet import RenormalizingUrbanUNet
    
    feature_dims = {
        'aerial_alphaearth': 64,
        'gtfs': 32,
        'roadnetwork': 32,
        'poi': 32
    }
    
    hidden_dim = 128
    output_dim = 32
    
    try:
        # Base model (3 resolutions: 8, 9, 10)
        base_model = UrbanUNet(
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_convs=4,
            device="cpu"
        )
        
        # Renormalizing model (6 resolutions: 5, 6, 7, 8, 9, 10)
        renorm_config = create_renormalizing_config()
        renorm_model = RenormalizingUrbanUNet(
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_convs=4,
            renorm_config=renorm_config,
            device="cpu"
        )
        
        base_params = sum(p.numel() for p in base_model.parameters())
        renorm_params = sum(p.numel() for p in renorm_model.parameters())
        
        logger.info("Model Comparison:")
        logger.info(f"  Base model (res 8-10): {base_params:,} parameters")
        logger.info(f"  Renormalizing (res 5-10): {renorm_params:,} parameters")
        logger.info(f"  Parameter ratio: {renorm_params / base_params:.2f}x")
        logger.info(f"  Extra resolutions: 3 → 6 levels")
        logger.info(f"  Architecture: Standard U-Net → Renormalizing Flow")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🏙️  Renormalizing Urban U-Net Example")
    print("=" * 60)
    print()
    print("Choose an example to run:")
    print("1. Architecture Test (recommended first)")
    print("2. Fast Example (reduced parameters)")
    print("3. Full Example (complete pipeline)")
    print("4. Model Comparison")
    print("5. All Tests")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            success = run_architecture_test()
            
        elif choice == "2":
            success = run_fast_example()
            
        elif choice == "3":
            success = run_basic_example()
            
        elif choice == "4":
            success = compare_with_base_model()
            
        elif choice == "5":
            logger.info("🔄 Running all tests...")
            success = (
                run_architecture_test() and
                compare_with_base_model()
            )
            
            if success:
                logger.info("✅ All tests passed! You can now try the fast or full example.")
            
        else:
            logger.error("Invalid choice. Please run again and choose 1-5.")
            success = False
        
        if success:
            print("\n✅ Example completed successfully!")
        else:
            print("\n❌ Example failed. Check logs for details.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Example interrupted by user.")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        print("\n❌ Example failed unexpectedly. Check logs for details.")