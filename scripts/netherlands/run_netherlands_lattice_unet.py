#!/usr/bin/env python3
"""
Run Lattice U-Net on Netherlands AlphaEarth data
================================================

This script runs the hexagonal lattice U-Net model on Netherlands data
using the urban embedding pipeline with hexagonal graph construction.
"""

import logging
import sys
import torch
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stage2_fusion.pipeline import UrbanEmbeddingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('netherlands_lattice_unet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_netherlands_config():
    """Create configuration for Netherlands with hexagonal lattice."""

    config = {
        'project_dir': str(project_root),
        'city_name': 'netherlands',

        # Feature processing configuration
        'feature_processing': {
            'pca': {
                'variance_threshold': 0.95,
                'min_components': {
                    'alphaearth': 16,
                    'poi': 10,
                    'roads': 10
                },
                'max_components': 50
            }
        },

        # IMPORTANT: Use hexagonal lattice graph
        'graph_type': 'hexagonal',

        # Hexagonal lattice parameters
        'hexagonal': {
            'neighbor_rings': 2,  # Include 2 rings of neighbors
            'edge_weight': 1.0,
            'include_self_loops': False
        },

        # Model configuration for Lattice U-Net
        'model': {
            'architecture': 'LatticeUNet',
            'input_dim': 64,  # AlphaEarth embeddings
            'hidden_dim': 128,
            'output_dim': 64,
            'num_layers': 4,
            'dropout': 0.1,
            'conv_type': 'gcn',  # Graph convolution type
            'use_batch_norm': False,
            'use_graph_norm': True,
            'use_skip_connections': True,
            'activation': 'gelu'
        },

        # Training configuration
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'early_stopping_patience': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },

        # Modalities - focusing on AlphaEarth for now
        'stage1_modalities': ['alphaearth'],

        # Graph construction modes
        'modes': {
            8: 'drive',
            9: 'bike',
            10: 'walk'
        },

        # Graph parameters (fallback if not using hexagonal)
        'graph': {
            'speeds': {'walk': 5.0, 'bike': 15.0, 'drive': 50.0},
            'max_travel_time': {'walk': 15, 'bike': 15, 'drive': 15},
            'search_radius': {'walk': 2000, 'bike': 5000, 'drive': 15000},
            'beta': {'walk': 0.1, 'bike': 0.05, 'drive': 0.01}
        },

        # Visualization
        'visualization': {
            'cmap': 'viridis',
            'dpi': 150,
            'figsize': [12, 10]
        },

        # Output directory
        'output_dir': 'results/netherlands_lattice_unet',

        # Cache directory
        'cache_dir': 'cache',

        # Monitoring
        'use_wandb': False,
        'log_interval': 10
    }

    return config


def main():
    """Main execution."""

    logger.info("="*80)
    logger.info("RUNNING LATTICE U-NET ON NETHERLANDS DATA")
    logger.info("="*80)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create configuration
    config = create_netherlands_config()

    logger.info("\nConfiguration:")
    logger.info(f"- Study area: {config['city_name']}")
    logger.info(f"- Graph type: {config['graph_type']}")
    logger.info(f"- Model: {config['model']['architecture']}")
    logger.info(f"- Hidden dim: {config['model']['hidden_dim']}")
    logger.info(f"- Num layers: {config['model']['num_layers']}")
    logger.info(f"- Device: {config['training']['device']}")

    # Initialize pipeline
    logger.info("\nInitializing Urban Embedding Pipeline...")
    try:
        pipeline = UrbanEmbeddingPipeline(config)

        # Run the pipeline
        logger.info("\nRunning pipeline...")
        logger.info("This will:")
        logger.info("1. Load Netherlands AlphaEarth embeddings")
        logger.info("2. Build hexagonal lattice graph")
        logger.info("3. Train Lattice U-Net model")
        logger.info("4. Generate urban embeddings")
        logger.info("5. Perform clustering and analysis")

        embeddings = pipeline.run()

        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)

        if embeddings is not None:
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            logger.info(f"Output saved to: {config['output_dir']}")

        return embeddings

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error:")
        sys.exit(1)