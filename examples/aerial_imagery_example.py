"""
Example: Process PDOK aerial imagery with DINOv3 for Netherlands study areas.

This example demonstrates:
1. Fetching RGB aerial images from PDOK (free Dutch aerial imagery)
2. Encoding with DINOv3 (including remote sensing variant)
3. Hierarchical aggregation to H3 hexagons
4. Integration with other modalities (AlphaEarth)
"""

import sys
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from modalities.aerial_imagery import AerialImageryProcessor
from modalities.alphaearth import AlphaEarthProcessor
from urban_embedding.pipeline import UrbanEmbeddingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_aerial_imagery_standalone():
    """Process aerial imagery as a standalone modality."""
    
    # Configuration
    config = {
        'study_area': 'south_holland',
        'output_dir': 'data/processed/embeddings/aerial_imagery',
        'pdok_year': '2023',
        'model_name': 'dinov3_rs_base',  # Remote sensing variant
        'target_h3_resolution': 10,
        'fine_h3_resolution': 12,
        'hierarchical_levels': 2,
        'image_resolution': 512,
        'batch_size': 8
    }
    
    # Initialize processor
    processor = AerialImageryProcessor(config)
    
    # Run pipeline
    logger.info("Starting aerial imagery processing for South Holland")
    output_path = processor.run_pipeline(
        study_area='south_holland',
        h3_resolution=10,
        output_dir=config['output_dir']
    )
    
    # Load and inspect results
    embeddings_df = pd.read_parquet(output_path)
    logger.info(f"Processed {len(embeddings_df)} H3 cells")
    logger.info(f"Embedding dimensions: {len([c for c in embeddings_df.columns if c.startswith('dim_')])}")
    
    return embeddings_df


def combine_with_alphaearth():
    """
    Combine PDOK aerial imagery with AlphaEarth embeddings.
    
    This creates a multi-source representation:
    - PDOK: Recent high-res RGB images from Netherlands
    - AlphaEarth: Global satellite embeddings from Google Earth Engine
    """
    
    results = {}
    
    # Process PDOK aerial imagery
    logger.info("Processing PDOK aerial imagery...")
    aerial_config = {
        'study_area': 'south_holland',
        'output_dir': 'data/processed/embeddings/aerial_imagery',
        'pdok_year': 'current',
        'model_name': 'dinov3_rs_base',
        'target_h3_resolution': 10
    }
    
    aerial_processor = AerialImageryProcessor(aerial_config)
    aerial_path = aerial_processor.run_pipeline(
        study_area='south_holland',
        h3_resolution=10,
        output_dir=aerial_config['output_dir']
    )
    results['aerial_imagery'] = pd.read_parquet(aerial_path)
    
    # Process AlphaEarth if available
    try:
        logger.info("Processing AlphaEarth embeddings...")
        alphaearth_config = {
            'study_area': 'south_holland',
            'output_dir': 'data/processed/embeddings/alphaearth',
            'source_dir': 'path/to/alphaearth/tiles'  # Update this path
        }
        
        alphaearth_processor = AlphaEarthProcessor(alphaearth_config)
        alphaearth_path = alphaearth_processor.run_pipeline(
            study_area='south_holland',
            h3_resolution=10,
            output_dir=alphaearth_config['output_dir']
        )
        results['alphaearth'] = pd.read_parquet(alphaearth_path)
    except Exception as e:
        logger.warning(f"AlphaEarth processing skipped: {e}")
    
    # Merge embeddings
    if len(results) > 1:
        logger.info("Merging multi-source embeddings...")
        
        # Join on H3 index
        merged = results['aerial_imagery'].set_index('h3_index')
        
        for name, df in results.items():
            if name != 'aerial_imagery':
                df_indexed = df.set_index('h3_index')
                # Rename columns to avoid conflicts
                df_indexed.columns = [f"{name}_{col}" if col != 'resolution' else col 
                                     for col in df_indexed.columns]
                merged = merged.join(df_indexed, how='outer', rsuffix=f'_{name}')
        
        logger.info(f"Created multi-source embeddings with {len(merged)} cells")
        logger.info(f"Total dimensions: {len(merged.columns)}")
        
        # Save merged embeddings
        output_path = Path('data/processed/embeddings/multi_source')
        output_path.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_path / 'aerial_alphaearth_merged.parquet')
        
        return merged
    
    return results.get('aerial_imagery')


def hierarchical_active_inference_example():
    """
    Demonstrate hierarchical processing with active inference principles.
    
    This shows how the nested structure works:
    1. Fine-scale image patches (H3 res 12-13)
    2. DINOv3 encoding with attention
    3. Hierarchical aggregation with Fisher information
    4. Coarse-scale H3 embeddings (res 10)
    """
    
    from modalities.aerial_imagery.dinov3_encoder import DINOv3Encoder
    import torch
    
    # Initialize encoder with hierarchical extraction
    encoder = DINOv3Encoder(
        model_name='dinov3_rs_base',
        extract_hierarchical=True,
        use_registers=True
    )
    
    # Simulate processing an image
    logger.info("Demonstrating hierarchical encoding...")
    
    # Create dummy image (would be real PDOK image in practice)
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Encode with hierarchical features
    result = encoder.encode_image(dummy_image, return_attention=True)
    
    logger.info(f"Embedding shape: {result.embeddings.shape}")
    if result.patch_features is not None:
        logger.info(f"Patch features shape: {result.patch_features.shape}")
    if result.attention_maps is not None:
        logger.info(f"Attention maps shape: {result.attention_maps.shape}")
    
    # Compute Fisher information for active inference
    if result.patch_features is not None:
        fisher = encoder.compute_fisher_information(result.patch_features)
        logger.info(f"Fisher information matrix shape: {fisher.shape}")
        
        # Natural gradient would use this Fisher information
        # to adaptively weight different spatial scales
        eigenvalues = torch.linalg.eigvalsh(fisher)
        logger.info(f"Fisher eigenvalue range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
    
    return result


def main():
    """Run example workflows."""
    
    print("=" * 80)
    print("PDOK Aerial Imagery Processing with DINOv3")
    print("=" * 80)
    
    # Example 1: Standalone processing
    print("\n1. Processing PDOK aerial imagery...")
    aerial_embeddings = process_aerial_imagery_standalone()
    print(f"   ✓ Generated embeddings for {len(aerial_embeddings)} hexagons")
    
    # Example 2: Multi-source combination
    print("\n2. Combining with AlphaEarth embeddings...")
    multi_source = combine_with_alphaearth()
    if multi_source is not None:
        print(f"   ✓ Created multi-source embeddings")
    
    # Example 3: Hierarchical active inference
    print("\n3. Demonstrating hierarchical active inference...")
    encoding = hierarchical_active_inference_example()
    print(f"   ✓ Hierarchical encoding with Fisher information")
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("\nKey insights:")
    print("- PDOK provides free, high-resolution aerial imagery for Netherlands")
    print("- DINOv3 (especially RS variant) creates rich visual embeddings")
    print("- Hierarchical aggregation preserves multi-scale information")
    print("- Fisher information enables active inference dynamics")
    print("- Combines well with AlphaEarth for multi-source representation")


if __name__ == "__main__":
    main()