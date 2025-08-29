"""
Enhanced Pipeline for Renormalizing Urban U-Net

Extends the existing UrbanEmbeddingPipeline to support:
- H3 resolutions 5-10 (6-level hierarchy)
- Renormalizing data flow (upward accumulation, downward pass-through)
- Simple MSE losses (reconstruction at res 10 + consistency between levels)
- Integration with existing data loading and preprocessing
"""

import logging
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import geopandas as gpd
from tqdm.auto import tqdm

from urban_embedding.pipeline import UrbanEmbeddingPipeline
from .renormalizing_unet import RenormalizingUrbanUNet, create_renormalizing_config, RenormalizingConfig
from .renormalizing_trainer import RenormalizingModelTrainer, create_enhanced_mappings

logger = logging.getLogger(__name__)


class RenormalizingUrbanPipeline(UrbanEmbeddingPipeline):
    """
    Enhanced pipeline for renormalizing hierarchical U-Net.
    
    Extends the base pipeline to support:
    - 6-level hierarchy (H3 resolutions 5-10)
    - Renormalizing data flow patterns
    - Enhanced cross-resolution mapping generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize renormalizing pipeline with enhanced configuration."""
        
        # Update config to include renormalizing settings
        config = self._enhance_config(config)
        
        # Initialize base pipeline
        super().__init__(config)
        
        # Renormalizing-specific configuration
        self.renorm_config = create_renormalizing_config(**config.get('renormalizing', {}))
        
        # Extended resolution range
        self.extended_resolutions = [10, 9, 8, 7, 6, 5]
        
        logger.info("Initialized RenormalizingUrbanPipeline")
        logger.info(f"Extended resolutions: {self.extended_resolutions}")
        logger.info(f"Renormalizing config: {self.renorm_config}")
    
    def _enhance_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance configuration for renormalizing model."""
        enhanced_config = config.copy()
        
        # Update modes to include all resolutions 10-5
        enhanced_config['modes'] = {
            10: 'walk',    # Liveability (daily patterns, short-term)
            9: 'bike',     # Neighborhood  
            8: 'drive',    # District level
            7: 'drive',    # Urban structure
            6: 'drive',    # Regional planning
            5: 'drive'     # Sustainability (infrastructure, long-term)
        }
        
        # Default renormalizing settings
        if 'renormalizing' not in enhanced_config:
            enhanced_config['renormalizing'] = {
                'accumulation_mode': 'grouped',
                'normalization_type': 'layer',
                'upward_momentum': 0.9,
                'residual_connections': True
            }
        
        # Adjust training parameters for deeper hierarchy
        if 'training' in enhanced_config:
            training_config = enhanced_config['training']
            # Slightly lower learning rate for stability
            training_config['learning_rate'] = training_config.get('learning_rate', 1e-4) * 0.8
            # Increase consistency weight for 6-level hierarchy
            if 'loss_weights' in training_config:
                training_config['loss_weights']['consistency'] = training_config['loss_weights'].get('consistency', 2.0) * 1.5
        
        return enhanced_config
    
    def _generate_extended_mappings(self, 
                                  hex_indices_by_res: Dict[int, List[str]],
                                  base_mappings: Optional[Dict[Tuple[int, int], torch.Tensor]] = None) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Generate mappings for extended resolution range (5-10).
        
        Creates parent-child mappings between adjacent H3 resolutions:
        (10,9), (9,8), (8,7), (7,6), (6,5)
        """
        import h3
        
        mappings = {}
        
        # Generate mappings for all adjacent pairs
        for i in range(len(self.extended_resolutions) - 1):
            fine_res = self.extended_resolutions[i]        # 10, 9, 8, 7, 6  
            coarse_res = self.extended_resolutions[i + 1]  # 9, 8, 7, 6, 5
            
            logger.info(f"Generating mapping from resolution {fine_res} to {coarse_res}")
            
            # Use existing mapping if available
            if base_mappings and (fine_res, coarse_res) in base_mappings:
                mappings[(fine_res, coarse_res)] = base_mappings[(fine_res, coarse_res)]
                logger.info(f"Using existing mapping for {fine_res}→{coarse_res}")
                continue
            
            # Generate new mapping using H3 parent-child relationships
            fine_indices = hex_indices_by_res.get(fine_res, [])
            coarse_indices = hex_indices_by_res.get(coarse_res, [])
            
            if not fine_indices or not coarse_indices:
                logger.warning(f"Missing hex indices for mapping {fine_res}→{coarse_res}")
                continue
            
            # Create mapping matrix
            fine_to_idx = {h3_cell: idx for idx, h3_cell in enumerate(fine_indices)}
            coarse_to_idx = {h3_cell: idx for idx, h3_cell in enumerate(coarse_indices)}
            
            # Find parent-child relationships
            mapping_data = []
            
            for fine_idx, fine_cell in enumerate(fine_indices):
                try:
                    # Get parent at coarser resolution
                    parent_cell = h3.cell_to_parent(fine_cell, coarse_res)
                    
                    if parent_cell in coarse_to_idx:
                        coarse_idx = coarse_to_idx[parent_cell]
                        mapping_data.append([coarse_idx, fine_idx, 1.0])  # [row, col, value]
                    else:
                        # Handle edge case: parent not in coarse set
                        # Find nearest coarse cell (simplified)
                        fine_lat, fine_lng = h3.cell_to_latlng(fine_cell)
                        
                        min_dist = float('inf')
                        best_coarse_idx = 0
                        
                        for coarse_idx, coarse_cell in enumerate(coarse_indices[:min(100, len(coarse_indices))]):  # Limit search
                            coarse_lat, coarse_lng = h3.cell_to_latlng(coarse_cell)
                            dist = (fine_lat - coarse_lat)**2 + (fine_lng - coarse_lng)**2
                            if dist < min_dist:
                                min_dist = dist
                                best_coarse_idx = coarse_idx
                        
                        mapping_data.append([best_coarse_idx, fine_idx, 1.0])
                        
                except Exception as e:
                    logger.warning(f"Error mapping cell {fine_cell}: {e}")
                    continue
            
            if mapping_data:
                # Convert to sparse tensor
                mapping_array = np.array(mapping_data)
                indices = torch.LongTensor(mapping_array[:, :2].T)  # [2, num_edges]
                values = torch.FloatTensor(mapping_array[:, 2])
                shape = (len(coarse_indices), len(fine_indices))
                
                sparse_mapping = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
                mappings[(fine_res, coarse_res)] = sparse_mapping
                
                logger.info(f"Created mapping {fine_res}→{coarse_res}: {len(mapping_data)} connections")
            else:
                logger.error(f"Failed to create mapping for {fine_res}→{coarse_res}")
        
        return mappings
    
    def _extend_graph_construction(self, 
                                 hex_indices_by_res: Dict[int, List[str]],
                                 regions_by_res: Dict[int, gpd.GeoDataFrame]) -> Dict[int, Any]:
        """
        Extend graph construction to cover resolutions 5-10.
        
        For missing resolutions, creates simple hexagonal lattice graphs.
        """
        # Get existing graphs from base constructor
        existing_edge_features = self.graph_constructor.construct_graphs(
            self.project_dir,
            self.config['city_name'],
            {k: v for k, v in hex_indices_by_res.items() if k in [8, 9, 10]},  # Original resolutions
            {k: v for k, v in regions_by_res.items() if k in [8, 9, 10]}
        )
        
        # Extend to missing resolutions (7, 6, 5)
        extended_edge_features = existing_edge_features.copy()
        
        for res in [7, 6, 5]:
            if res not in extended_edge_features and res in hex_indices_by_res:
                logger.info(f"Creating hexagonal lattice graph for resolution {res}")
                
                # Create simple hexagonal connectivity
                hex_indices = hex_indices_by_res[res]
                edges = []
                weights = []
                
                # Use H3 neighbor relationships
                import h3
                hex_to_idx = {cell: idx for idx, cell in enumerate(hex_indices)}
                
                for idx, cell in enumerate(hex_indices):
                    try:
                        # Get immediate neighbors
                        neighbors = h3.grid_disk(cell, 1)  # k=1 ring
                        neighbors.discard(cell)  # Remove self
                        
                        for neighbor in neighbors:
                            if neighbor in hex_to_idx:
                                neighbor_idx = hex_to_idx[neighbor]
                                edges.append([idx, neighbor_idx])
                                weights.append(1.0)  # Uniform weights for lattice
                                
                    except Exception as e:
                        logger.warning(f"Error processing cell {cell}: {e}")
                        continue
                
                if edges:
                    # Convert to tensors
                    edge_array = np.array(edges)
                    edge_indices = torch.LongTensor(edge_array.T).to(self.device)  # [2, num_edges]
                    edge_weights = torch.FloatTensor(weights).to(self.device)
                    
                    # Create mock edge features object
                    from urban_embedding.graph_construction import EdgeFeatures
                    extended_edge_features[res] = EdgeFeatures(
                        index=edge_indices,
                        accessibility=edge_weights,
                        travel_time=edge_weights,  # Mock travel times
                        distance=edge_weights       # Mock distances
                    )
                    
                    logger.info(f"Created lattice graph for resolution {res}: {len(edges)} edges")
                else:
                    logger.warning(f"Failed to create graph for resolution {res}")
        
        return extended_edge_features
    
    def run(self) -> Dict[int, pd.DataFrame]:
        """
        Run the complete renormalizing pipeline.
        
        Returns:
            embeddings_by_res: Embeddings for all resolutions 5-10
        """
        try:
            # Load data (base pipeline functionality)
            area_gdf, base_regions_by_res, base_hex_indices_by_res = self.load_data()
            
            # Extend to cover resolutions 7-5 (generate if missing)
            logger.info("Extending data to resolutions 7-5...")
            extended_regions_by_res, extended_hex_indices_by_res = self._extend_data_coverage(
                base_regions_by_res, base_hex_indices_by_res
            )
            
            # Load features (using finest resolution data)
            raw_features = self.load_features(extended_hex_indices_by_res[10])
            
            # Process features
            features = self.feature_processor.fit_transform(
                {name: df.values for name, df in raw_features.items()},
                self.config['city_name']
            )
            
            # Convert to tensors
            features = {
                name: torch.tensor(feat, dtype=torch.float32).to(self.device)
                for name, feat in features.items()
            }
            
            # Extended graph construction
            edge_features = self._extend_graph_construction(
                extended_hex_indices_by_res, extended_regions_by_res
            )
            
            # Process edge features
            edge_indices = {}
            edge_weights = {}
            
            for res, ef in edge_features.items():
                edge_indices[res] = ef.index.to(self.device)
                edge_weights[res] = (ef.accessibility / (ef.accessibility.max() + 1e-8)).to(self.device)
                
                logger.info(f"Resolution {res}: {edge_indices[res].shape[1]} edges")
            
            # Generate extended mappings
            logger.info("Generating extended cross-resolution mappings...")
            base_mappings = self.feature_processor.load_cross_scale_mappings(
                city_name=self.config['city_name'],
                resolutions=[8, 9, 10]  # Base resolutions
            )
            
            extended_mappings = self._generate_extended_mappings(
                extended_hex_indices_by_res, base_mappings
            )
            
            # Initialize renormalizing trainer
            logger.info("Initializing renormalizing trainer...")
            trainer = RenormalizingModelTrainer(
                model_config={
                    'feature_dims': self.feature_processor.feature_dims,
                    'hidden_dim': self.config['model']['hidden_dim'],
                    'output_dim': self.config['model']['output_dim'],
                    'num_convs': self.config['model']['num_convs']
                },
                renorm_config=self.renorm_config,
                loss_weights=self.config['training']['loss_weights'],
                city_name=self.config['city_name'],
                checkpoint_dir=self.cache_dir / 'checkpoints'
            )
            
            # Training
            logger.info("Starting renormalizing training...")
            embeddings_dict, training_state = trainer.train(
                features_dict=features,
                edge_indices=edge_indices,
                edge_weights=edge_weights,
                mappings=extended_mappings,
                **self.config['training']
            )
            
            # Convert to DataFrames
            embeddings_by_res = {}
            for res, emb_tensor in embeddings_dict.items():
                emb_array = emb_tensor.detach().cpu().numpy()
                emb_df = pd.DataFrame(
                    emb_array,
                    index=extended_hex_indices_by_res[res],
                    columns=[f'emb_{i}' for i in range(emb_array.shape[1])]
                )
                embeddings_by_res[res] = emb_df
                
                logger.info(f"Resolution {res} embeddings: {emb_array.shape}")
            
            # Save results
            logger.info("Saving renormalizing embeddings...")
            self._save_extended_results(embeddings_dict, extended_hex_indices_by_res, training_state)
            
            return embeddings_by_res
            
        except Exception as e:
            logger.error(f"Renormalizing pipeline failed: {str(e)}")
            logger.error("Pipeline error details:", exc_info=True)
            raise
    
    def _extend_data_coverage(self, 
                            base_regions_by_res: Dict[int, gpd.GeoDataFrame],
                            base_hex_indices_by_res: Dict[int, List[str]]) -> Tuple[Dict[int, gpd.GeoDataFrame], Dict[int, List[str]]]:
        """
        Extend data coverage to include resolutions 5-7.
        
        For missing resolutions, generates H3 cells covering the same geographic area.
        """
        extended_regions = base_regions_by_res.copy()
        extended_indices = base_hex_indices_by_res.copy()
        
        # Get bounding box from finest resolution
        if 10 in base_regions_by_res:
            base_gdf = base_regions_by_res[10]
            bounds = base_gdf.total_bounds  # [minx, miny, maxx, maxy]
            
            import h3
            from shapely.geometry import Polygon
            
            # Generate H3 cells for missing resolutions
            for res in [7, 6, 5]:
                if res not in extended_regions:
                    logger.info(f"Generating H3 cells for resolution {res}")
                    
                    # Generate H3 cells covering the bounding box
                    cells = set()
                    
                    # Sample points within bounds and get H3 cells
                    lat_step = (bounds[3] - bounds[1]) / (20 * (res + 1))  # Coarser sampling for lower res
                    lng_step = (bounds[2] - bounds[0]) / (20 * (res + 1))
                    
                    lat = bounds[1]
                    while lat <= bounds[3]:
                        lng = bounds[0]
                        while lng <= bounds[2]:
                            try:
                                cell = h3.latlng_to_cell(lat, lng, res)
                                cells.add(cell)
                            except:
                                pass
                            lng += lng_step
                        lat += lat_step
                    
                    cells = list(cells)
                    
                    # Create geometries
                    geometries = []
                    for cell in cells:
                        try:
                            boundary = h3.cell_to_boundary(cell)
                            poly = Polygon([(lng, lat) for lat, lng in boundary])
                            geometries.append(poly)
                        except:
                            geometries.append(None)
                    
                    # Create GeoDataFrame
                    regions_gdf = gpd.GeoDataFrame({
                        'geometry': geometries
                    }, index=cells)
                    
                    # Remove invalid geometries
                    regions_gdf = regions_gdf.dropna()
                    
                    # Add mock density data
                    regions_gdf['FSI_24'] = 0.0
                    regions_gdf['in_study_area'] = True
                    
                    extended_regions[res] = regions_gdf
                    extended_indices[res] = list(regions_gdf.index)
                    
                    logger.info(f"Generated {len(regions_gdf)} H3 cells for resolution {res}")
        
        return extended_regions, extended_indices
    
    def _save_extended_results(self, 
                             embeddings_dict: Dict[int, torch.Tensor],
                             hex_indices_by_res: Dict[int, List[str]],
                             training_state: dict):
        """Save results for extended resolution range."""
        
        # Create results directory
        results_dir = self.output_dir / 'embeddings' / f'{self.config["city_name"]}_renormalizing'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings for each resolution
        for res, emb_tensor in embeddings_dict.items():
            emb_array = emb_tensor.detach().cpu().numpy()
            emb_df = pd.DataFrame(
                emb_array,
                index=hex_indices_by_res[res],
                columns=[f'emb_{i}' for i in range(emb_array.shape[1])]
            )
            
            output_path = results_dir / f'renormalizing_embeddings_res{res}.parquet'
            emb_df.to_parquet(output_path)
            logger.info(f"Saved resolution {res} embeddings to {output_path}")
        
        # Save training metadata
        import json
        metadata = {
            'city_name': self.config['city_name'],
            'resolutions': self.extended_resolutions,
            'renorm_config': self.renorm_config.__dict__,
            'training_config': self.config['training'],
            'final_loss': training_state.get('loss', 0.0),
            'num_parameters': sum(p.numel() for p in training_state.get('model_state', {}).values() if isinstance(p, torch.Tensor))
        }
        
        metadata_path = results_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved training metadata to {metadata_path}")


def create_renormalizing_config_preset(preset: str = "default") -> Dict[str, Any]:
    """Create renormalizing pipeline configuration presets."""
    
    base_config = {
        "city_name": "south_holland",
        "project_dir": r"C:\Users\Bert Berkers\PycharmProjects\UrbanRepML",
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
        "model": {
            "hidden_dim": 128,
            "output_dim": 32,
            "num_convs": 4
        },
        "training": {
            "learning_rate": 1e-4,
            "num_epochs": 500,
            "warmup_epochs": 50,
            "patience": 100,
            "gradient_clip": 1.0,
            "loss_weights": {
                "reconstruction": 1.0,
                "consistency": 3.0  # Higher weight for 6-level consistency
            }
        },
        "renormalizing": {
            "accumulation_mode": "grouped",
            "normalization_type": "layer",
            "upward_momentum": 0.9,
            "residual_connections": True
        },
        "visualization": {
            "n_clusters": {5: 6, 6: 6, 7: 7, 8: 8, 9: 8, 10: 8},
            "cmap": "Accent",
            "dpi": 600,
            "figsize": (12, 12)
        }
    }
    
    if preset == "fast":
        base_config["training"]["num_epochs"] = 100
        base_config["training"]["patience"] = 50
        base_config["model"]["hidden_dim"] = 64
        
    elif preset == "high_quality":
        base_config["training"]["num_epochs"] = 1000
        base_config["training"]["learning_rate"] = 5e-5
        base_config["model"]["hidden_dim"] = 256
        base_config["model"]["num_convs"] = 6
        
    return base_config


if __name__ == "__main__":
    # Test the renormalizing pipeline
    config = create_renormalizing_config_preset("default")
    
    pipeline = RenormalizingUrbanPipeline(config)
    
    print("✅ RenormalizingUrbanPipeline initialized successfully!")
    print(f"Extended resolutions: {pipeline.extended_resolutions}")
    print(f"Renormalizing config: {pipeline.renorm_config}")