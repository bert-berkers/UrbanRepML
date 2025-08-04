#!/usr/bin/env python3
"""
Test the cleaned up pipeline with FSI 0.1 data
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from urban_embedding import UrbanEmbeddingPipeline

def test_clean_pipeline():
    """Test the pipeline with clean data structure."""
    
    print("=== Testing Clean Pipeline Configuration ===")
    
    try:
        # Create FSI 0.1 configuration
        config = UrbanEmbeddingPipeline.create_south_holland_fsi01_config()
        
        print(f"[CONFIG] Created configuration")
        print(f"   City: {config['city_name']}")
        print(f"   FSI Threshold: {config['fsi_threshold']}")
        print(f"   Project Dir: {config['project_dir']}")
        
        # Initialize pipeline
        print("\n[INIT] Initializing pipeline...")
        pipeline = UrbanEmbeddingPipeline(config)
        print("[OK] Pipeline initialized successfully!")
        
        # Verify data structure
        print(f"\n[PATHS] Data structure:")
        print(f"   Data dir: {pipeline.data_dir}")
        print(f"   Embeddings dir: {pipeline.embeddings_dir}")
        print(f"   Cache dir: {pipeline.cache_dir}")
        print(f"   Output dir: {pipeline.output_dir}")
        
        # Test data loading
        print("\n[DATA] Testing data loading...")
        area_gdf, regions_by_res, hex_indices_by_res = pipeline.load_data()
        
        print("[OK] Geographic data loaded!")
        print(f"   Available resolutions: {list(regions_by_res.keys())}")
        for res, regions in regions_by_res.items():
            in_study = regions['in_study_area'].sum() if 'in_study_area' in regions.columns else len(regions)
            print(f"   Resolution {res}: {len(regions)} total, {in_study} in study area")
            if 'FSI_24' in regions.columns:
                fsi_stats = regions['FSI_24'].describe()
                print(f"     FSI range: [{fsi_stats['min']:.3f}, {fsi_stats['max']:.3f}], mean: {fsi_stats['mean']:.3f}")
        
        # Test feature loading
        print("\n[FEATURES] Testing feature loading...")
        raw_features = pipeline.load_features(hex_indices_by_res[10])
        
        print("[OK] Features loaded!")
        print(f"   Available modalities: {list(raw_features.keys())}")
        for modality, features in raw_features.items():
            print(f"   {modality}: {features.shape}")
            print(f"     Range: [{features.values.min():.3f}, {features.values.max():.3f}]")
            print(f"     Non-zero: {(features.values != 0).sum()}/{features.size} ({100*(features.values != 0).sum()/features.size:.1f}%)")
        
        # Test accessibility graphs
        print(f"\n[GRAPHS] Testing accessibility graphs...")
        data_networks_dir = Path(config['project_dir']) / 'data' / 'networks' / 'accessibility'
        print(f"   Looking for graphs in: {data_networks_dir}")
        
        graph_files = list(data_networks_dir.glob("*.pkl"))
        print(f"   Found {len(graph_files)} graph files:")
        for graph_file in graph_files:
            print(f"     {graph_file.name}")
        
        print(f"\n[SUCCESS] All tests passed! Pipeline is ready.")
        print(f"\n[SUMMARY] Clean data structure verified:")
        print(f"   - FSI 0.1 preprocessed data: {len(regions_by_res[10])} regions at res-10")
        print(f"   - AlphaEarth embeddings: {raw_features['aerial_alphaearth'].shape}")
        print(f"   - Accessibility graphs: {len(graph_files)} files")
        print(f"   - All paths organized under data/ directory")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clean_pipeline()
    sys.exit(0 if success else 1)