"""
Test multi-modal pipeline with all available modalities.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from urban_embedding.multimodal_loader import MultiModalLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_pipeline():
    """Test the multi-modal pipeline."""
    logger.info("Testing multi-modal pipeline...")
    
    # Check if config exists
    config_path = Path("configs/netherlands_pipeline.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False
    
    # Initialize loader
    loader = MultiModalLoader(config_path=str(config_path))
    
    # Test 1: Load modalities
    logger.info("\n=== TEST 1: Loading Modalities ===")
    modalities = loader.load_all_modalities()
    
    if not modalities:
        logger.error("No modalities loaded!")
        return False
    
    for name, df in modalities.items():
        logger.info(f"{name}: {df.shape} ({df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB)")
    
    # Test 2: Alignment
    logger.info("\n=== TEST 2: Aligning Modalities ===")
    aligned = loader.align_modalities(method='intersection')
    
    for name, df in aligned.items():
        logger.info(f"{name} aligned: {df.shape}")
    
    # Test 3: Fusion
    logger.info("\n=== TEST 3: Fusing Modalities ===")
    fused = loader.fuse_modalities(method='concatenate')
    logger.info(f"Fused shape: {fused.shape}")
    logger.info(f"Total features: {fused.shape[1]}")
    logger.info(f"Memory usage: {fused.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Test 4: Feature groups
    logger.info("\n=== TEST 4: Feature Groups ===")
    feature_groups = loader.get_feature_groups()
    for modality, features in feature_groups.items():
        logger.info(f"{modality}: {len(features)} features")
        logger.info(f"  Sample features: {features[:3]}...")
    
    # Test 5: Save output
    logger.info("\n=== TEST 5: Saving Output ===")
    output_dir = Path("data/processed/multimodal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "netherlands_multimodal_res10.parquet"
    loader.save_aligned_data(str(output_path))
    
    # Verify saved file
    if output_path.exists():
        import pandas as pd
        saved_df = pd.read_parquet(output_path)
        logger.info(f"Saved file verified: {saved_df.shape}")
        logger.info("✓ All tests passed!")
        return True
    else:
        logger.error("Failed to save output file")
        return False


def check_prerequisites():
    """Check if all required files exist."""
    logger.info("Checking prerequisites...")
    
    required_files = [
        "configs/netherlands_pipeline.yaml",
        "data/processed/embeddings/alphaearth/alphaearth_embeddings_res10.parquet",
        "data/processed/embeddings/poi/poi_embeddings_res10.parquet",
        # Roads embeddings might not exist yet
    ]
    
    missing = []
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            logger.info(f"✓ {file_path}")
        else:
            logger.warning(f"✗ {file_path}")
            missing.append(file_path)
    
    # Check optional roads embeddings
    roads_path = Path("data/processed/embeddings/roads/roads_embeddings_res10.parquet")
    if roads_path.exists():
        logger.info(f"✓ {roads_path} (optional)")
    else:
        logger.warning(f"✗ {roads_path} (optional - will skip roads modality)")
    
    if missing and "config" in str(missing[0]):
        logger.error("Configuration file is missing!")
        return False
    
    return True


def main():
    """Main test execution."""
    logger.info("=" * 80)
    logger.info("MULTI-MODAL PIPELINE TEST")
    logger.info("=" * 80)
    
    if not check_prerequisites():
        logger.error("Prerequisites check failed!")
        return
    
    try:
        success = test_pipeline()
        
        if success:
            logger.info("\n" + "=" * 80)
            logger.info("✓ PIPELINE TEST SUCCESSFUL")
            logger.info("=" * 80)
            logger.info("\nNext steps:")
            logger.info("1. Generate roads embeddings: python scripts/generate_roads_netherlands.py")
            logger.info("2. Run full pipeline with model training")
            logger.info("3. Perform clustering and analysis")
        else:
            logger.error("\n✗ Pipeline test failed")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)


if __name__ == "__main__":
    main()