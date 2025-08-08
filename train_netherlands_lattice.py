"""
Train LatticeUNet on Netherlands-wide embeddings using spatial batching.
This script orchestrates the complete training pipeline for large-scale urban representation learning.
"""

import logging
import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd
import torch
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from urban_embedding.spatial_batching import (
    SpatialBatcher, 
    BatchingConfig,
    SpatialBatchDataset
)
from urban_embedding.lattice_unet import (
    LatticeUNet,
    LatticeUNetConfig,
    BatchedLatticeTrainer,
    create_lattice_unet_config
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NetherlandsTrainingPipeline:
    """Complete training pipeline for Netherlands-wide urban embeddings."""
    
    def __init__(
        self,
        embeddings_path: Path,
        regions_path: Path,
        output_dir: Path,
        config: Dict = None
    ):
        """
        Initialize training pipeline.
        
        Args:
            embeddings_path: Path to combined Netherlands embeddings
            regions_path: Path to Netherlands regions GeoDataFrame
            output_dir: Output directory for results
            config: Training configuration dictionary
        """
        self.embeddings_path = embeddings_path
        self.regions_path = regions_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            # Data parameters
            "resolution": 10,
            "train_split": 0.8,
            "val_split": 0.2,
            
            # Batching parameters
            "batch_size": 8000,  # Hexagons per spatial batch
            "overlap_ratio": 0.15,
            "grouping_resolution": 6,
            "dataloader_batch_size": 1,  # Spatial batches per training step
            
            # Model parameters
            "model_size": "medium",  # "small", "medium", "large"
            "conv_type": "gcn",
            "num_layers": 4,
            "dropout": 0.1,
            "use_skip_connections": True,
            
            # Training parameters
            "epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "reconstruction_weight": 1.0,
            "consistency_weight": 0.3,
            
            # Device and optimization
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_workers": 0,
            "pin_memory": True,
            
            # Logging and checkpointing
            "log_interval": 10,
            "save_interval": 20,
            "wandb_project": "netherlands-urban-embeddings",
            "experiment_name": None
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Set device
        self.device = self.config["device"]
        logger.info(f"Using device: {self.device}")
        
        # Initialize WandB if available and configured
        self.use_wandb = WANDB_AVAILABLE and self.config.get("wandb_project")
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        experiment_name = (
            self.config.get("experiment_name") or 
            f"netherlands_lattice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        wandb.init(
            project=self.config["wandb_project"],
            name=experiment_name,
            config=self.config
        )
        
        logger.info(f"Initialized WandB: {self.config['wandb_project']}/{experiment_name}")
    
    def load_data(self) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Load embeddings and regions data.
        
        Returns:
            Tuple of (embeddings_df, regions_gdf)
        """
        logger.info("Loading data...")
        
        # Load embeddings
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")
        
        embeddings_df = pd.read_parquet(self.embeddings_path)
        logger.info(f"Loaded embeddings: {embeddings_df.shape}")
        
        # Load regions
        if not self.regions_path.exists():
            raise FileNotFoundError(f"Regions file not found: {self.regions_path}")
        
        regions_gdf = gpd.read_parquet(self.regions_path)
        logger.info(f"Loaded regions: {len(regions_gdf)} hexagons")
        
        # Ensure common index
        common_indices = list(
            set(embeddings_df.index).intersection(set(regions_gdf.index))
        )
        
        if not common_indices:
            raise ValueError("No common indices between embeddings and regions")
        
        embeddings_df = embeddings_df.loc[common_indices]
        regions_gdf = regions_gdf.loc[common_indices]
        
        logger.info(f"Common hexagons: {len(common_indices)}")
        
        return embeddings_df, regions_gdf
    
    def create_batches(
        self,
        embeddings_df: pd.DataFrame,
        regions_gdf: gpd.GeoDataFrame
    ) -> tuple[List, List]:
        """
        Create spatial batches for training and validation.
        
        Args:
            embeddings_df: Embeddings DataFrame
            regions_gdf: Regions GeoDataFrame
            
        Returns:
            Tuple of (train_batches, val_batches)
        """
        logger.info("Creating spatial batches...")
        
        # Create batching configuration
        batch_config = BatchingConfig(
            batch_size=self.config["batch_size"],
            overlap_ratio=self.config["overlap_ratio"],
            grouping_resolution=self.config["grouping_resolution"]
        )
        
        # Initialize spatial batcher
        cache_dir = self.output_dir / "cache" / "spatial_batches"
        spatial_batcher = SpatialBatcher(
            config=batch_config,
            device=self.device,
            cache_dir=cache_dir
        )
        
        # Create all batches
        all_batches = spatial_batcher.create_spatial_batches(
            regions_gdf,
            embeddings_df
        )
        
        # Split batches into train/val
        n_train = int(len(all_batches) * self.config["train_split"])
        
        # Shuffle batches for random split
        np.random.seed(42)  # For reproducibility
        batch_indices = np.random.permutation(len(all_batches))
        
        train_batches = [all_batches[i] for i in batch_indices[:n_train]]
        val_batches = [all_batches[i] for i in batch_indices[n_train:]]
        
        logger.info(f"Created {len(train_batches)} training batches")
        logger.info(f"Created {len(val_batches)} validation batches")
        
        # Log batch statistics
        batch_stats = spatial_batcher.get_batch_statistics(all_batches)
        
        if self.use_wandb:
            wandb.log({
                "data/total_batches": len(all_batches),
                "data/train_batches": len(train_batches),
                "data/val_batches": len(val_batches),
                "data/avg_hexagons_per_batch": batch_stats["num_hexagons"].mean(),
                "data/total_hexagons": batch_stats["num_hexagons"].sum()
            })
        
        return train_batches, val_batches
    
    def create_model(self, input_dim: int) -> LatticeUNet:
        """
        Create LatticeUNet model.
        
        Args:
            input_dim: Input embedding dimensions
            
        Returns:
            LatticeUNet model
        """
        logger.info("Creating model...")
        
        # Create model configuration
        model_config = create_lattice_unet_config(
            input_dim=input_dim,
            model_size=self.config["model_size"],
            conv_type=self.config["conv_type"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            use_skip_connections=self.config["use_skip_connections"],
            reconstruction_weight=self.config["reconstruction_weight"],
            consistency_weight=self.config["consistency_weight"]
        )
        
        # Create model
        model = LatticeUNet(model_config)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
        
        if self.use_wandb:
            wandb.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/input_dim": input_dim,
                "model/output_dim": model_config.output_dim
            })
        
        return model
    
    def train(self) -> Dict:
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting Netherlands urban embedding training...")
        start_time = datetime.now()
        
        # Load data
        embeddings_df, regions_gdf = self.load_data()
        
        # Create spatial batches
        train_batches, val_batches = self.create_batches(embeddings_df, regions_gdf)
        
        # Create data loaders
        train_dataset = SpatialBatchDataset(train_batches)
        val_dataset = SpatialBatchDataset(val_batches)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config["dataloader_batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"]
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config["dataloader_batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"]
        )
        
        # Create model
        input_dim = embeddings_df.shape[1]
        model = self.create_model(input_dim)
        
        # Create trainer
        trainer = BatchedLatticeTrainer(
            model=model,
            device=self.device,
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config["epochs"]):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Training
            train_metrics = trainer.train_epoch(train_loader, epoch)
            train_losses.append(train_metrics['total_loss'])
            
            # Validation
            val_metrics = trainer.evaluate_epoch(val_loader, epoch)
            val_losses.append(val_metrics['total_loss'])
            
            # Logging
            if (epoch + 1) % self.config["log_interval"] == 0:
                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"Train Loss = {train_metrics['total_loss']:.4f}, "
                    f"Val Loss = {val_metrics['total_loss']:.4f}, "
                    f"LR = {train_metrics['learning_rate']:.6f}"
                )
            
            # WandB logging
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/total_loss": train_metrics['total_loss'],
                    "train/reconstruction_loss": train_metrics['reconstruction_loss'],
                    "val/total_loss": val_metrics['total_loss'],
                    "val/reconstruction_loss": val_metrics['reconstruction_loss'],
                    "train/learning_rate": train_metrics['learning_rate']
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config["save_interval"] == 0:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                trainer.save_checkpoint(
                    str(checkpoint_path),
                    epoch + 1,
                    {'train_loss': train_metrics['total_loss'], 'val_loss': val_metrics['total_loss']}
                )
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                best_model_path = self.output_dir / "best_model.pt"
                trainer.save_checkpoint(
                    str(best_model_path),
                    epoch + 1,
                    {'train_loss': train_metrics['total_loss'], 'val_loss': val_metrics['total_loss']}
                )
                
                if self.use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"] = epoch + 1
        
        # Training completed
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Save final results
        results = {
            "training_time": str(training_time),
            "total_epochs": self.config["epochs"],
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "config": self.config,
            "total_hexagons": len(embeddings_df),
            "input_dimensions": input_dim,
            "output_dimensions": model.config.output_dim
        }
        
        # Save results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for k, v in results.items():
                if isinstance(v, (np.integer, np.floating)):
                    json_results[k] = v.item()
                elif isinstance(v, np.ndarray):
                    json_results[k] = v.tolist()
                else:
                    json_results[k] = v
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        if self.use_wandb:
            wandb.finish()
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Train LatticeUNet on Netherlands embeddings"
    )
    
    # Data paths
    parser.add_argument(
        "--embeddings_path",
        type=Path,
        required=True,
        help="Path to combined Netherlands embeddings"
    )
    parser.add_argument(
        "--regions_path",
        type=Path,
        required=True,
        help="Path to Netherlands regions GeoDataFrame"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    
    # Model parameters
    parser.add_argument(
        "--model_size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size preset"
    )
    parser.add_argument(
        "--conv_type",
        choices=["gcn", "gat"],
        default="gcn",
        help="Graph convolution type"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8000,
        help="Hexagons per spatial batch"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    # Experiment tracking
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name for experiment tracking"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="netherlands-urban-embeddings",
        help="WandB project name"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable WandB logging"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "model_size": args.model_size,
        "conv_type": args.conv_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "experiment_name": args.experiment_name,
        "wandb_project": args.wandb_project if not args.no_wandb else None
    }
    
    # Initialize pipeline
    pipeline = NetherlandsTrainingPipeline(
        embeddings_path=args.embeddings_path,
        regions_path=args.regions_path,
        output_dir=args.output_dir,
        config=config
    )
    
    # Run training
    results = pipeline.train()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Training time: {results['training_time']}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Total hexagons: {results['total_hexagons']:,}")
    print(f"Input dimensions: {results['input_dimensions']}")
    print(f"Output dimensions: {results['output_dimensions']}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()