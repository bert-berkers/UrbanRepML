"""
Script to rebuild the complete UrbanRepML pipeline data from scratch.
Run this after project cleanup to regenerate all preprocessed [TODO SORT & CLEAN UP] data.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from urban_embedding import UrbanEmbeddingPipeline
from scripts.experiment_utils import ExperimentManager

def rebuild_south_holland_data():
    """Rebuild South Holland data with different thresholds."""
    
    thresholds = [50, 70, 80, 90]
    
    for threshold in thresholds:
        print(f"\n🏗️ Building South Holland data with {threshold}% threshold...")
        
        # Create experiment for this rebuild
        exp_manager = ExperimentManager()
        exp_path = exp_manager.create_experiment(
            name=f"south_holland_threshold{threshold}_rebuild",
            description=f"Rebuild South Holland pipeline data with {threshold}% building density threshold"
        )
        
        # Create configuration
        config = UrbanEmbeddingPipeline.create_default_config(
            city_name="south_holland",
            threshold=threshold
        )
        
        # Initialize pipeline
        pipeline = UrbanEmbeddingPipeline(config)
        
        # Run preprocessing auxiliary data steps
        print("  📍 Preprocessing regions...")
        pipeline.preprocess_data()
        
        print("  🕸️ Building accessibility graphs...")
        pipeline.build_graphs()
        
        print("  🧠 Training model...")
        embeddings = pipeline.run()
        
        print(f"  ✅ Completed threshold {threshold}%")
        print(f"     Results saved to: {exp_path}")

def rebuild_delft_data():
    """Rebuild Delft comparison data."""
    
    print("\n🏗️ Building Delft comparison data...")
    
    exp_manager = ExperimentManager()
    exp_path = exp_manager.create_experiment(
        name="delft_comparison_rebuild",
        description="Rebuild Delft urban embeddings for comparison with South Holland"
    )
    
    config = UrbanEmbeddingPipeline.create_default_config(
        city_name="delft",
        threshold=80  # Standard threshold for comparison
    )
    
    pipeline = UrbanEmbeddingPipeline(config)
    embeddings = pipeline.run()
    
    print(f"  ✅ Completed Delft rebuild")
    print(f"     Results saved to: {exp_path}")

if __name__ == "__main__":
    print("🚀 Rebuilding UrbanRepML Pipeline Data")
    print("=" * 50)
    
    # Check if embeddings are available
    embeddings_path = project_root / "data" / "embeddings"
    if not embeddings_path.exists():
        print("❌ Error: Input embeddings not found!")
        print(f"   Please ensure embeddings are available in: {embeddings_path}")
        print("   See PROJECT_REBUILD_GUIDE.md for data requirements")
        sys.exit(1)
    
    print("📋 Available rebuilding options:")
    print("1. South Holland (all thresholds)")
    print("2. Delft comparison")
    print("3. Both")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice in ["1", "3"]:
        rebuild_south_holland_data()
    
    if choice in ["2", "3"]:
        rebuild_delft_data()
    
    print("\n🎉 Pipeline rebuild completed!")
    print("   All results organized in experiments/ folder")
    print("   See PROJECT_REBUILD_GUIDE.md for usage instructions")