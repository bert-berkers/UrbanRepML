"""
Semantic Segmentation Processor - AlphaEarth + DINOv3 Fusion

Main orchestrator that:
1. Loads AlphaEarth embeddings from Google Earth Engine
2. Fetches and encodes aerial imagery with DINOv3 
3. Trains the conditioned segmentation network
4. Generates categorical semantic segmentation maps
5. Exports to H3 hexagon format
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import h3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

from modalities.base import ModalityProcessor
from modalities.alphaearth import AlphaEarthProcessor
from modalities.aerial_imagery import AerialImageryProcessor
from .fusion_network import AlphaEarthConditionedUNet, ConditioningConfig
from .segmentation_classes import SegmentationClasses, NetherlandsLandCover

logger = logging.getLogger(__name__)


class H3SegmentationDataset(Dataset):
    """
    Dataset for training semantic segmentation with H3 spatial structure.
    
    Combines:
    - AlphaEarth embeddings at H3 resolution 10
    - DINOv3 features from aerial images
    - Ground truth segmentation masks
    """
    
    def __init__(self,
                 alphaearth_embeddings: pd.DataFrame,
                 dinov3_features: pd.DataFrame,
                 segmentation_masks: Optional[Dict[str, np.ndarray]] = None,
                 augment: bool = False):
        """
        Initialize dataset.
        
        Args:
            alphaearth_embeddings: DataFrame with H3 index and AlphaEarth features
            dinov3_features: DataFrame with H3 index and DINOv3 features
            segmentation_masks: Optional ground truth masks for training
            augment: Whether to apply data augmentation
        """
        self.alphaearth_df = alphaearth_embeddings.set_index('h3_index')
        self.dinov3_df = dinov3_features.set_index('h3_index')
        self.segmentation_masks = segmentation_masks or {}
        self.augment = augment
        
        # Find common H3 cells
        self.h3_cells = list(
            set(self.alphaearth_df.index) & 
            set(self.dinov3_df.index)
        )
        
        logger.info(f"Dataset initialized with {len(self.h3_cells)} H3 cells")
    
    def __len__(self) -> int:
        return len(self.h3_cells)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h3_cell = self.h3_cells[idx]
        
        # Get AlphaEarth embedding
        alphaearth_row = self.alphaearth_df.loc[h3_cell]
        alphaearth_features = torch.FloatTensor([
            alphaearth_row[col] for col in alphaearth_row.index
            if col.startswith('dim_')
        ])
        
        # Get DINOv3 features
        dinov3_row = self.dinov3_df.loc[h3_cell]
        dinov3_features = torch.FloatTensor([
            dinov3_row[col] for col in dinov3_row.index
            if col.startswith('dim_')
        ])
        
        sample = {
            'h3_cell': h3_cell,
            'alphaearth_features': alphaearth_features.unsqueeze(0),  # Add region dimension
            'dinov3_features': dinov3_features.unsqueeze(0),  # Add patch dimension
        }
        
        # Add ground truth if available
        if h3_cell in self.segmentation_masks:
            mask = torch.LongTensor(self.segmentation_masks[h3_cell])
            sample['segmentation_target'] = mask
        
        return sample


class SemanticSegmentationProcessor(ModalityProcessor):
    """
    Main processor for AlphaEarth-conditioned semantic segmentation.
    
    This implements the full pipeline:
    1. Queue study area processing in Google Earth Engine
    2. Load/process AlphaEarth and DINOv3 features
    3. Train conditional segmentation network
    4. Generate semantic segmentation maps
    5. Convert to categorical variables at H3 resolution
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize processor.
        
        Config should include:
        - study_area: Name of study area
        - gee_queue: Whether to queue Earth Engine processing
        - model_config: Configuration for conditioning network
        - training: Training parameters
        - output_classes: Segmentation class configuration
        """
        super().__init__(config)
        
        # Subprocessors
        self.alphaearth_processor = AlphaEarthProcessor(
            config.get('alphaearth_config', {})
        )
        self.aerial_processor = AerialImageryProcessor(
            config.get('aerial_config', {})
        )
        
        # Model configuration
        model_config = config.get('model_config', {})
        self.conditioning_config = ConditioningConfig(
            alphaearth_dim=model_config.get('alphaearth_dim', 64),
            dinov3_dim=model_config.get('dinov3_dim', 768),
            conditioning_dim=model_config.get('conditioning_dim', 256),
            num_conditioning_layers=model_config.get('num_conditioning_layers', 3),
            use_cross_attention=model_config.get('use_cross_attention', True),
            attention_heads=model_config.get('attention_heads', 8)
        )
        
        # Training configuration
        self.training_config = config.get('training', {
            'epochs': 50,
            'batch_size': 4,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'use_class_weights': True
        })
        
        # Initialize model
        self.model = AlphaEarthConditionedUNet(
            config=self.conditioning_config,
            image_size=config.get('image_size', 512)
        )
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Initialized SemanticSegmentationProcessor on {self.device}")
    
    def validate_config(self):
        """Validate configuration parameters."""
        required = ['study_area', 'output_dir']
        for param in required:
            if param not in self.config:
                raise ValueError(f"Missing required config: {param}")
    
    def queue_earth_engine_processing(self, study_area: str) -> str:
        """
        Queue AlphaEarth processing in Google Earth Engine.
        
        Args:
            study_area: Name of study area
            
        Returns:
            Task ID or status message
        """
        logger.info(f"Queuing Earth Engine processing for {study_area}")
        
        try:
            # Load study area boundaries
            config_path = Path(f"study_areas/configs/{study_area}.yaml")
            if not config_path.exists():
                raise ValueError(f"Study area config not found: {study_area}")
            
            import yaml
            with open(config_path) as f:
                area_config = yaml.safe_load(f)
            
            # Get bounding box
            bbox = area_config['boundaries']['bbox']
            
            # Queue AlphaEarth processing
            task_id = self.alphaearth_processor.queue_gee_export(
                study_area=study_area,
                bbox=bbox,
                resolution=10,  # H3 resolution 10
                years=[2023, 2024]  # Recent years
            )
            
            logger.info(f"Queued Earth Engine task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to queue Earth Engine processing: {e}")
            return f"Failed: {e}"
    
    def load_data(self, study_area: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load AlphaEarth and DINOv3 data for study area.
        
        Args:
            study_area: Name of study area
            
        Returns:
            Tuple of (alphaearth_embeddings, dinov3_features)
        """
        # Load AlphaEarth embeddings
        alphaearth_path = Path(f"data/processed/embeddings/alphaearth/{study_area}_res10.parquet")
        if not alphaearth_path.exists():
            logger.info("AlphaEarth embeddings not found, processing...")
            alphaearth_path = self.alphaearth_processor.run_pipeline(
                study_area=study_area,
                h3_resolution=10,
                output_dir="data/processed/embeddings/alphaearth"
            )
        
        alphaearth_df = pd.read_parquet(alphaearth_path)
        logger.info(f"Loaded {len(alphaearth_df)} AlphaEarth embeddings")
        
        # Load DINOv3 features
        dinov3_path = Path(f"data/processed/embeddings/aerial_imagery/{study_area}_res10.parquet")
        if not dinov3_path.exists():
            logger.info("DINOv3 features not found, processing...")
            dinov3_path = self.aerial_processor.run_pipeline(
                study_area=study_area,
                h3_resolution=10,
                output_dir="data/processed/embeddings/aerial_imagery"
            )
        
        dinov3_df = pd.read_parquet(dinov3_path)
        logger.info(f"Loaded {len(dinov3_df)} DINOv3 features")
        
        return alphaearth_df, dinov3_df
    
    def create_training_dataset(self,
                              alphaearth_df: pd.DataFrame,
                              dinov3_df: pd.DataFrame,
                              ground_truth_masks: Optional[Dict[str, np.ndarray]] = None) -> H3SegmentationDataset:
        """
        Create training dataset from embeddings.
        
        Args:
            alphaearth_df: AlphaEarth embeddings
            dinov3_df: DINOv3 features
            ground_truth_masks: Optional ground truth segmentation masks
            
        Returns:
            Training dataset
        """
        return H3SegmentationDataset(
            alphaearth_embeddings=alphaearth_df,
            dinov3_features=dinov3_df,
            segmentation_masks=ground_truth_masks,
            augment=self.training_config.get('augment', False)
        )
    
    def train_model(self,
                   train_dataset: H3SegmentationDataset,
                   val_dataset: Optional[H3SegmentationDataset] = None) -> Dict[str, List[float]]:
        """
        Train the conditioned segmentation model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            
        Returns:
            Training history with losses and metrics
        """
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=False,
                num_workers=2
            )
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_config['epochs']
        )
        
        # Class weights for imbalanced data
        class_weights = None
        if self.training_config.get('use_class_weights', False):
            class_weights = torch.FloatTensor(
                SegmentationClasses.get_class_weights()
            ).to(self.device)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(self.training_config['epochs']):
            # Training phase
            self.model.train()
            train_losses = []
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.training_config['epochs']}")
            
            for batch in train_pbar:
                optimizer.zero_grad()
                
                # Move to device
                alphaearth_features = batch['alphaearth_features'].to(self.device)
                dinov3_features = batch['dinov3_features'].to(self.device)
                
                # Forward pass
                outputs = self.model(alphaearth_features, dinov3_features)
                predictions = outputs['segmentation']
                
                # Compute loss (if ground truth available)
                if 'segmentation_target' in batch:
                    targets = batch['segmentation_target'].to(self.device)
                    loss_dict = self.model.compute_loss(predictions, targets, class_weights)
                    loss = loss_dict['total_loss']
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                    train_pbar.set_postfix({'loss': loss.item()})
            
            # Update learning rate
            scheduler.step()
            
            # Record training loss
            if train_losses:
                avg_train_loss = np.mean(train_losses)
                history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_dataset and val_loader:
                self.model.eval()
                val_losses = []
                correct_predictions = 0
                total_pixels = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        alphaearth_features = batch['alphaearth_features'].to(self.device)
                        dinov3_features = batch['dinov3_features'].to(self.device)
                        
                        outputs = self.model(alphaearth_features, dinov3_features)
                        predictions = outputs['segmentation']
                        
                        if 'segmentation_target' in batch:
                            targets = batch['segmentation_target'].to(self.device)
                            loss_dict = self.model.compute_loss(predictions, targets, class_weights)
                            val_losses.append(loss_dict['total_loss'].item())
                            
                            # Accuracy
                            pred_classes = torch.argmax(predictions, dim=1)
                            correct_predictions += (pred_classes == targets).sum().item()
                            total_pixels += targets.numel()
                
                if val_losses:
                    avg_val_loss = np.mean(val_losses)
                    val_accuracy = correct_predictions / total_pixels if total_pixels > 0 else 0
                    
                    history['val_loss'].append(avg_val_loss)
                    history['val_accuracy'].append(val_accuracy)
                    
                    logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                              f"val_loss={avg_val_loss:.4f}, val_acc={val_accuracy:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")
        
        return history
    
    def generate_segmentation_maps(self,
                                 alphaearth_df: pd.DataFrame,
                                 dinov3_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate semantic segmentation maps for all H3 cells.
        
        Args:
            alphaearth_df: AlphaEarth embeddings
            dinov3_df: DINOv3 features
            
        Returns:
            Dictionary mapping H3 cells to segmentation maps
        """
        # Create inference dataset
        dataset = self.create_training_dataset(alphaearth_df, dinov3_df)
        data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
        
        segmentation_maps = {}
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Generating segmentation maps"):
                alphaearth_features = batch['alphaearth_features'].to(self.device)
                dinov3_features = batch['dinov3_features'].to(self.device)
                h3_cells = batch['h3_cell']
                
                # Forward pass
                outputs = self.model(alphaearth_features, dinov3_features)
                predictions = outputs['segmentation']
                
                # Convert to class predictions
                pred_classes = torch.argmax(predictions, dim=1)
                pred_classes_np = pred_classes.cpu().numpy()
                
                # Store results
                for i, h3_cell in enumerate(h3_cells):
                    segmentation_maps[h3_cell] = pred_classes_np[i]
        
        logger.info(f"Generated segmentation maps for {len(segmentation_maps)} H3 cells")
        return segmentation_maps
    
    def process_to_h3(self,
                     data: Tuple[pd.DataFrame, pd.DataFrame],
                     h3_resolution: int) -> pd.DataFrame:
        """
        Process to H3 format with categorical segmentation variables.
        
        Args:
            data: Tuple of (alphaearth_df, dinov3_df)
            h3_resolution: Target H3 resolution
            
        Returns:
            DataFrame with H3 indices and categorical segmentation classes
        """
        alphaearth_df, dinov3_df = data
        
        # Generate segmentation maps
        segmentation_maps = self.generate_segmentation_maps(alphaearth_df, dinov3_df)
        
        # Convert to categorical variables
        records = []
        for h3_cell, seg_map in segmentation_maps.items():
            # Get dominant class for this H3 cell
            dominant_class = np.bincount(seg_map.flatten()).argmax()
            
            # Get class metadata
            class_name = SegmentationClasses.class_id_to_name(dominant_class)
            is_urban = dominant_class in [cls.value for cls in SegmentationClasses.URBAN_CLASSES]
            
            # Calculate class distribution
            unique_classes, counts = np.unique(seg_map.flatten(), return_counts=True)
            total_pixels = seg_map.size
            
            record = {
                'h3_index': h3_cell,
                'resolution': h3.h3_get_resolution(h3_cell),
                'dominant_class_id': dominant_class,
                'dominant_class_name': class_name,
                'is_urban': is_urban,
                'class_diversity': len(unique_classes),  # Number of different classes
                'dominant_class_ratio': counts[unique_classes == dominant_class][0] / total_pixels
            }
            
            # Add class probabilities as separate columns
            for class_id in range(SegmentationClasses.get_num_classes()):
                class_ratio = counts[unique_classes == class_id][0] / total_pixels if class_id in unique_classes else 0.0
                class_name_short = SegmentationClasses.class_id_to_name(class_id).replace(' ', '_')
                record[f"class_{class_id}_{class_name_short}"] = class_ratio
            
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Created categorical segmentation DataFrame with {len(df)} H3 cells")
        
        return df
    
    def run_pipeline(self,
                    study_area: str,
                    h3_resolution: int,
                    output_dir: str,
                    queue_gee: bool = True) -> str:
        """
        Execute complete semantic segmentation pipeline.
        
        Args:
            study_area: Name of study area
            h3_resolution: Target H3 resolution
            output_dir: Output directory
            queue_gee: Whether to queue Google Earth Engine processing
            
        Returns:
            Path to output file
        """
        self.validate_config()
        
        # Queue Earth Engine processing if requested
        if queue_gee:
            gee_task_id = self.queue_earth_engine_processing(study_area)
            logger.info(f"Earth Engine task queued: {gee_task_id}")
        
        # Load embeddings
        logger.info(f"Processing semantic segmentation for {study_area}")
        alphaearth_df, dinov3_df = self.load_data(study_area)
        
        # Process to H3 with categorical variables
        segmentation_df = self.process_to_h3((alphaearth_df, dinov3_df), h3_resolution)
        
        # Save results
        output_path = Path(output_dir) / f"semantic_segmentation_{study_area}_res{h3_resolution}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        segmentation_df.to_parquet(output_path, index=False)
        
        # Save metadata
        metadata = {
            'study_area': study_area,
            'h3_resolution': h3_resolution,
            'num_cells': len(segmentation_df),
            'num_classes': SegmentationClasses.get_num_classes(),
            'model_config': self.conditioning_config.__dict__,
            'class_distribution': segmentation_df['dominant_class_id'].value_counts().to_dict()
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved semantic segmentation results to {output_path}")
        return str(output_path)