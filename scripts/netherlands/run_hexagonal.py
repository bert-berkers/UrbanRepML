#!/usr/bin/env python3
"""
South Holland Hexagonal Lattice Experiment.

This experiment uses fully connected hexagonal lattice graphs instead of 
accessibility-based graphs. It applies to the full South Holland province
without FSI filtering, using regular convolutions within each layer of the UNet 
while still mapping between resolutions with reconstruction + consistency losses.
"""

import sys
import os
from pathlib import Path

# Add project root to path so we can import stage2_fusion
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stage2_fusion import UrbanEmbeddingPipeline
import logging

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

# Choose a preset or customize individual parameters below
PRESET = "QUICK_TEST"  # Options: "QUICK_TEST", "FULL_RUN", "DEBUG", "CUSTOM"

# ------------------------------------------------------------------------------
# PRESETS - Predefined configurations for common use cases
# ------------------------------------------------------------------------------
PRESETS = {
    "QUICK_TEST": {
        "epochs": 10,
        "hidden_dim": 32,
        "output_dim": 8,
        "num_convs": 2,
        "learning_rate": 1e-3,
        "warmup_epochs": 2,
        "patience": 5
    },
    "FULL_RUN": {
        "epochs": 1000,
        "hidden_dim": 128,
        "output_dim": 32,
        "num_convs": 6,
        "learning_rate": 1e-4,
        "warmup_epochs": 100,
        "patience": 100
    },
    "DEBUG": {
        "epochs": 1,
        "hidden_dim": 16,
        "output_dim": 4,
        "num_convs": 1,
        "learning_rate": 1e-3,
        "warmup_epochs": 0,
        "patience": 1
    },
    "MEDIUM": {
        "epochs": 100,
        "hidden_dim": 64,
        "output_dim": 16,
        "num_convs": 4,
        "learning_rate": 5e-4,
        "warmup_epochs": 10,
        "patience": 20
    }
}

# ------------------------------------------------------------------------------
# CUSTOM CONFIGURATION (used when PRESET = "CUSTOM")
# ------------------------------------------------------------------------------

# Data Selection
CITY_NAME = "south_holland"        # City/region to process

# Graph Construction - Hexagonal Lattice Parameters
NEIGHBOR_RINGS = 1                 # 1 = direct neighbors only, 2 = include second ring
EDGE_WEIGHT = 1.0                  # Uniform weight for all edges
INCLUDE_SELF_LOOPS = False         # Whether to add self-connections

# Model Architecture
HIDDEN_DIM = 128                   # Hidden layer size (affects memory usage)
OUTPUT_DIM = 32                    # Final embedding dimension
NUM_CONVS = 6                      # Number of GCN layers per block

# Training Parameters
EPOCHS = 100                       # Number of training iterations
LEARNING_RATE = 1e-4               # Learning rate
WARMUP_EPOCHS = 10                 # Linear warmup period
PATIENCE = 50                      # Early stopping patience
GRADIENT_CLIP = 1.0                # Gradient clipping value
BATCH_SIZE = 1                     # Graphs per batch (usually 1 for full graph)

# Loss Configuration
RECONSTRUCTION_WEIGHT = 1          # Weight for reconstruction loss
CONSISTENCY_WEIGHT = 3             # Weight for cross-scale consistency loss

# Feature Processing
VARIANCE_THRESHOLD = 0.95          # PCA variance to retain
MAX_PCA_COMPONENTS = 32            # Maximum PCA dimensions

# Visualization
N_CLUSTERS = 8                     # Number of clusters for visualization

# Miscellaneous
DEBUG_MODE = True                  # Enable debug logging
WANDB_PROJECT = None               # Set to string to enable WandB logging (e.g., "urban-hexagonal")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO if not DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'stage2_fusion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run the South Holland Hexagonal Lattice experiment."""
    
    print("="*70)
    print("SOUTH HOLLAND HEXAGONAL LATTICE EXPERIMENT")
    print("="*70)
    print("Graph Type: Fully Connected Hexagonal Lattice")
    print("Coverage: Full South Holland Province (no FSI filtering)")
    print("Architecture: UrbanUNet with regular convolutions")
    print(f"Configuration Preset: {PRESET}")
    print("="*70)
    
    try:
        # Get configuration based on preset
        if PRESET in PRESETS:
            preset_config = PRESETS[PRESET]
            print(f"\n[INFO] Using {PRESET} preset configuration")
        elif PRESET == "CUSTOM":
            preset_config = None
            print("\n[INFO] Using custom configuration")
        else:
            raise ValueError(f"Unknown preset: {PRESET}")
        
        # Create base configuration for hexagonal lattice experiment
        config = UrbanEmbeddingPipeline.create_hexagonal_lattice_config(
            city_name=CITY_NAME,
            neighbor_rings=NEIGHBOR_RINGS if PRESET == "CUSTOM" else 1,
            edge_weight=EDGE_WEIGHT if PRESET == "CUSTOM" else 1.0
        )
        
        # Apply preset or custom configuration
        if preset_config:
            # Apply preset values
            config['training']['num_epochs'] = preset_config['epochs']
            config['training']['learning_rate'] = preset_config['learning_rate']
            config['training']['warmup_epochs'] = preset_config['warmup_epochs']
            config['training']['patience'] = preset_config['patience']
            config['model']['hidden_dim'] = preset_config['hidden_dim']
            config['model']['output_dim'] = preset_config['output_dim']
            config['model']['num_convs'] = preset_config['num_convs']
        else:
            # Apply custom configuration
            config['training']['num_epochs'] = EPOCHS
            config['training']['learning_rate'] = LEARNING_RATE
            config['training']['warmup_epochs'] = WARMUP_EPOCHS
            config['training']['patience'] = PATIENCE
            config['training']['gradient_clip'] = GRADIENT_CLIP
            config['training']['batch_size'] = BATCH_SIZE
            config['training']['loss_weights']['reconstruction'] = RECONSTRUCTION_WEIGHT
            config['training']['loss_weights']['consistency'] = CONSISTENCY_WEIGHT
            
            config['model']['hidden_dim'] = HIDDEN_DIM
            config['model']['output_dim'] = OUTPUT_DIM
            config['model']['num_convs'] = NUM_CONVS
            
            config['feature_processing']['pca']['variance_threshold'] = VARIANCE_THRESHOLD
            config['feature_processing']['pca']['max_components'] = MAX_PCA_COMPONENTS
            
            config['hexagonal']['neighbor_rings'] = NEIGHBOR_RINGS
            config['hexagonal']['edge_weight'] = EDGE_WEIGHT
            config['hexagonal']['include_self_loops'] = INCLUDE_SELF_LOOPS
            
            config['visualization']['n_clusters'] = {8: N_CLUSTERS, 9: N_CLUSTERS, 10: N_CLUSTERS}
        
        # Set debug mode and WandB
        config['debug'] = DEBUG_MODE
        if WANDB_PROJECT:
            config['wandb_project'] = WANDB_PROJECT
        
        # Log key configuration parameters
        logger.info("\n[CONFIG] HEXAGONAL LATTICE EXPERIMENT CONFIGURATION:")
        logger.info(f"Preset: {PRESET}")
        logger.info(f"City: {config['city_name']}")
        logger.info(f"Graph Type: {config['graph_type']}")
        logger.info(f"\nHexagonal Parameters:")
        logger.info(f"  - Neighbor rings: {config['hexagonal']['neighbor_rings']}")
        logger.info(f"  - Edge weight: {config['hexagonal']['edge_weight']}")
        logger.info(f"  - Self loops: {config['hexagonal']['include_self_loops']}")
        logger.info(f"\nModel Parameters:")
        logger.info(f"  - Hidden dim: {config['model']['hidden_dim']}")
        logger.info(f"  - Output dim: {config['model']['output_dim']}")
        logger.info(f"  - Convolutions: {config['model']['num_convs']}")
        logger.info(f"\nTraining Parameters:")
        logger.info(f"  - Epochs: {config['training']['num_epochs']}")
        logger.info(f"  - Learning rate: {config['training']['learning_rate']}")
        logger.info(f"  - Warmup epochs: {config['training']['warmup_epochs']}")
        logger.info(f"  - Patience: {config['training']['patience']}")
        logger.info(f"  - Loss weights: {config['training']['loss_weights']}")
        
        # Initialize and run pipeline
        logger.info("\n[INIT] Initializing pipeline...")
        pipeline = UrbanEmbeddingPipeline(config)
        
        logger.info("[RUN] Running hexagonal lattice experiment...")
        embeddings = pipeline.run()
        
        if embeddings:
            logger.info("\n[SUCCESS] Hexagonal lattice experiment completed!")
            logger.info(f"Generated embeddings for {len(embeddings)} resolutions")
            
            for res, emb_df in embeddings.items():
                logger.info(f"  - Resolution {res}: {emb_df.shape[0]} hexagons, {emb_df.shape[1]} dimensions")
                
            return 0
        else:
            logger.error("\n[FAILED] No embeddings were generated!")
            return 1
            
    except Exception as e:
        logger.error(f"\n[ERROR] EXPERIMENT FAILED: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())