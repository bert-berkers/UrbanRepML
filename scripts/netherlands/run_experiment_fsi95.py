#!/usr/bin/env python3
"""
South Holland FSI 95% Experiment.

This experiment focuses on the top 5% densest urban areas in South Holland,
using accessibility-based graphs for multi-scale urban representation learning.
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
        "learning_rate": 1e-5,
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
FSI_PERCENTILE = 95                # Top X% densest areas (95 = top 5%)

# Graph Construction - Accessibility Parameters
SPEEDS = {                         # Travel speeds (m/s)
    'walk': 1.4,
    'bike': 4.17,
    'drive': 11.11
}
MAX_TRAVEL_TIME = {                # Maximum travel time (seconds)
    'walk': 300,
    'bike': 450,
    'drive': 600
}
SEARCH_RADIUS = {                  # Search radius (meters)
    'walk': 75,
    'bike': 150,
    'drive': 300
}
BETA = {                           # Distance decay parameter
    'walk': 0.0020,
    'bike': 0.0012,
    'drive': 0.0008
}

# Model Architecture
HIDDEN_DIM = 128                   # Hidden layer size (affects memory usage)
OUTPUT_DIM = 32                    # Final embedding dimension
NUM_CONVS = 6                      # Number of GCN layers per block

# Training Parameters
EPOCHS = 100                       # Number of training iterations
LEARNING_RATE = 1e-5               # Learning rate
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
WANDB_PROJECT = None               # Set to string to enable WandB logging (e.g., "urban-fsi95")

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
    """Run the South Holland FSI 95% experiment."""
    
    print("="*60)
    print("SOUTH HOLLAND FSI 95% EXPERIMENT")
    print("="*60)
    print(f"Coverage: Top {100-FSI_PERCENTILE}% densest urban areas")
    print("Graph Type: Accessibility-based multi-modal networks")
    print("Architecture: FullAreaUNet with multi-scale processing_modalities")
    print(f"Configuration Preset: {PRESET}")
    print("="*60)
    
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
        
        # Create base configuration for FSI experiment
        config = UrbanEmbeddingPipeline.create_default_config(
            city_name=CITY_NAME,
            threshold=FSI_PERCENTILE
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
            config['threshold'] = FSI_PERCENTILE
            
            config['graph']['speeds'] = SPEEDS
            config['graph']['max_travel_time'] = MAX_TRAVEL_TIME
            config['graph']['search_radius'] = SEARCH_RADIUS
            config['graph']['beta'] = BETA
            
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
            
            config['visualization']['n_clusters'] = {8: N_CLUSTERS, 9: N_CLUSTERS, 10: N_CLUSTERS}
        
        # Set debug mode and WandB
        config['debug'] = DEBUG_MODE
        if WANDB_PROJECT:
            config['wandb_project'] = WANDB_PROJECT
        
        # Log key configuration parameters
        logger.info("\n[CONFIG] FSI 95% EXPERIMENT CONFIGURATION:")
        logger.info(f"Preset: {PRESET}")
        logger.info(f"City: {config['city_name']}")
        logger.info(f"FSI Percentile: {config.get('threshold', FSI_PERCENTILE)}%")
        logger.info(f"\nGraph Parameters:")
        logger.info(f"  - Speeds: {config['graph']['speeds']}")
        logger.info(f"  - Max travel times: {config['graph']['max_travel_time']}")
        logger.info(f"  - Beta values: {config['graph']['beta']}")
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
        
        logger.info("[RUN] Running FSI 95% experiment...")
        embeddings = pipeline.run()
        
        if embeddings:
            logger.info("\n[SUCCESS] FSI 95% experiment completed!")
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