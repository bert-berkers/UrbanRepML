#!/usr/bin/env python3
"""
Test script to run South Holland FSI 99% experiment with optimized parameters.
"""
import sys
import logging
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

from urban_embedding import UrbanEmbeddingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('south_holland_fsi99_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_optimized_config():
    """Create optimized configuration for South Holland FSI 99%."""
    config = {
        "city_name": "south_holland",
        "project_dir": str(project_dir),
        "fsi_threshold": 0.99,  # Top 1% densest areas
        "feature_processing": {
            "pca": {
                "variance_threshold": 0.95,
                "max_components": 32,
                "min_components": {
                    "aerial_alphaearth": 16,
                    "gtfs": 16,
                    "roadnetwork": 16,
                    "poi": 16
                },
                "eps": 1e-8
            }
        },
        "graph": {
            "speeds": {
                'walk': 1.4,
                'bike': 4.17,
                'drive': 11.11
            },
            "max_travel_time": {
                'walk': 300,    # 5 minutes
                'bike': 450,    # 7.5 minutes  
                'drive': 600    # 10 minutes (optimized from 15)
            },
            "search_radius": {
                'walk': 75,
                'bike': 150,
                'drive': 300
            },
            "beta": {
                'walk': 0.0020,
                'bike': 0.0012,
                'drive': 0.0008
            }
        },
        "model": {
            "hidden_dim": 64,    # Reduced from 128
            "output_dim": 32,
            "num_convs": 3       # Reduced from 6
        },
        "training": {
            "learning_rate": 1e-4,   # Slightly higher for faster convergence
            "num_epochs": 50,        # Much reduced from 10000
            "warmup_epochs": 5,      # Reduced from 1000
            "patience": 10,          # Reduced from 100
            "gradient_clip": 1.0,
            "loss_weights": {
                "reconstruction": 1,
                "consistency": 3
            }
        },
        "visualization": {
            "n_clusters": {8: 5, 9: 5, 10: 5},  # Fewer clusters for small dataset
            "cmap": "Accent",
            "dpi": 300,          # Reduced from 600 for speed
            "figsize": (10, 10)
        },
        "modes": {
            8: 'drive',
            9: 'bike', 
            10: 'walk'
        },
        "wandb_project": "urban-embedding-south-holland-fsi99",
        "wandb_mode": "offline",  # Run WandB in offline mode
        "debug": True
    }
    
    return config

def main():
    """Run the South Holland FSI 99% experiment."""
    try:
        logger.info("Starting South Holland FSI 99% experiment")
        
        # Create optimized configuration
        config = create_optimized_config()
        
        # Log key parameters
        logger.info(f"City: {config['city_name']}")
        logger.info(f"FSI threshold: {config['fsi_threshold']} (top 1% densest)")
        logger.info(f"Model: {config['model']['hidden_dim']}D hidden, {config['model']['num_convs']} convs")
        logger.info(f"Training: {config['training']['num_epochs']} epochs, lr={config['training']['learning_rate']}")
        logger.info(f"Graph: max travel times {config['graph']['max_travel_time']}")
        
        # Initialize and run pipeline
        logger.info("Initializing pipeline...")
        pipeline = UrbanEmbeddingPipeline(config)
        
        logger.info("Running pipeline...")
        embeddings = pipeline.run()
        
        if embeddings is not None:
            logger.info("Pipeline completed successfully!")
            for res, emb_df in embeddings.items():
                logger.info(f"Resolution {res}: {emb_df.shape} embeddings generated")
            return embeddings
        else:
            logger.error("Pipeline failed to generate embeddings")
            return None
            
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        result = main()
        if result is not None:
            print("✅ South Holland FSI 99% experiment completed successfully!")
        else:
            print("❌ South Holland FSI 99% experiment failed")
    except Exception as e:
        print(f"❌ South Holland FSI 99% experiment failed with error: {str(e)}")
        sys.exit(1)