# D:\Projects\UrbanRepML\stage2_fusion\pipeline.py

import logging
from pathlib import Path
import sys
import wandb
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from tqdm.auto import tqdm
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import ListedColormap
import time

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('stage2_fusion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from .data.feature_processing import UrbanFeatureProcessor
from .graphs.graph_construction import SpatialGraphConstructor, EdgeFeatures
from .graphs.hexagonal_graph_constructor import HexagonalLatticeConstructor
from .models.urban_unet import UrbanUNet
from .analysis.analytics import UrbanEmbeddingAnalyzer
# from .threshold_prep import ThresholdPreprocessor  # Using custom FSI filtering

class UrbanEmbeddingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Setup directories first
        self._setup_directories()

        # Check and create threshold variant if needed
        self._create_threshold_variant()

        # Initialize components AFTER potential threshold processing_modalities
        self._init_components()

    def _setup_directories(self):
        """Setup directory structure."""
        self.project_dir = Path(self.config['project_dir'])
        self.output_dir = self.project_dir / 'results'
        self.data_dir = self.project_dir / 'data' / 'preprocessed [TODO SORT & CLEAN UP]'
        self.embeddings_dir = self.project_dir / 'data' / 'embeddings'
        self.cache_dir = self.project_dir / 'cache'

        directories = [
            self.output_dir,
            self.data_dir,
            self.embeddings_dir,
            self.cache_dir,
            self.cache_dir / 'pca_models',
            self.cache_dir / 'checkpoints',
            self.data_dir / self.config['city_name']
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def _init_components(self):
        """Initialize pipeline components with updated configurations."""
        logger.info("Initializing components...")

        # Extract configuration for feature processing_modalities
        pca_config = self.config['feature_processing']['pca']
        min_components = pca_config.get('min_components', {})
        max_components = pca_config.get('max_components', None)

        # Initialize feature processor with explicit parameters
        self.feature_processor = UrbanFeatureProcessor(
            variance_threshold=pca_config['variance_threshold'],
            min_components=min_components,
            max_components=max_components,
            device=self.device,
            cache_dir=self.cache_dir / 'pca_models',
            preprocessed_dir=self.data_dir
        )

        # Initialize graph constructor based on graph type
        graph_type = self.config.get('graph_type', 'accessibility')
        logger.info(f"Graph type: {graph_type}")
        
        if graph_type == 'hexagonal' or graph_type == 'hexagonal_lattice':
            # Initialize hexagonal lattice constructor
            hexagonal_params = self.config.get('hexagonal', {})
            logger.info("Hexagonal lattice parameters:")
            logger.info(f"Neighbor rings: {hexagonal_params.get('neighbor_rings', 1)}")
            logger.info(f"Edge weight: {hexagonal_params.get('edge_weight', 1.0)}")
            logger.info(f"Include self loops: {hexagonal_params.get('include_self_loops', False)}")
            
            self.graph_constructor = HexagonalLatticeConstructor(
                device=self.device,
                modes=self.config['modes'],
                cache_dir=self.cache_dir,
                data_dir=self.project_dir / 'data',
                **hexagonal_params
            )
        else:
            # Default to accessibility-based graphs
            graph_params = self.config['graph']
            logger.info("Graph construction parameters:")
            logger.info(f"Speed settings: {graph_params['speeds']}")
            logger.info(f"Travel time limits: {graph_params['max_travel_time']}")
            logger.info(f"Search radii: {graph_params['search_radius']}")
            logger.info(f"Decay parameters: {graph_params['beta']}")

            self.graph_constructor = SpatialGraphConstructor(
                device=self.device,
                modes=self.config['modes'],
                cache_dir=self.cache_dir,
                data_dir=self.project_dir / 'data',
                **graph_params
            )

        # Initialize analyzer
        self.analyzer = UrbanEmbeddingAnalyzer(
            output_dir=self.output_dir,
            city_name=self.config['city_name'],
            cmap=self.config['visualization']['cmap'],
            dpi=self.config['visualization']['dpi'],
            figsize=self.config['visualization']['figsize']
        )

        logger.info("Components initialized successfully")

    def _create_threshold_variant(self) -> None:
        """Create threshold-filtered variant of city if needed."""
        if 'threshold' in self.config or 'fsi_threshold' in self.config:
            # Handle both percentage thresholds and FSI thresholds
            if 'fsi_threshold' in self.config:
                threshold = self.config['fsi_threshold']
                base_city = self.config['city_name'].split('_fsi')[0]  # Get base name
                # Convert decimal to percentage string for filename (0.99 -> 99, 0.1 -> 10)
                if threshold < 1.0:
                    threshold_str = str(int(threshold * 100))
                else:
                    threshold_str = str(int(threshold))
                variant_name = f"{base_city}_fsi{threshold_str}"
            else:
                threshold = self.config['threshold']
                base_city = self.config['city_name'].split('_threshold')[0]  # Get base name
                variant_name = f"{base_city}_threshold{threshold}"
            
            variant_dir = self.data_dir / variant_name

            # Check if all required files exist
            required_files = [
                variant_dir / 'area_study_gdf.parquet',
                variant_dir / 'regions_8_gdf.parquet',
                variant_dir / 'regions_9_gdf.parquet',
                variant_dir / 'regions_10_gdf.parquet',
                variant_dir / 'building_density_res8_preprocessed.parquet',
                variant_dir / 'building_density_res9_preprocessed.parquet',
                variant_dir / 'building_density_res10_preprocessed.parquet'
            ]

            files_exist = all(file.exists() for file in required_files)

            if not files_exist:
                # Check if data exists in experiments directory (new modular approach)
                exp_data_dir = self.project_dir / 'experiments' / variant_name / 'data'
                exp_required_files = [
                    exp_data_dir / 'area_study_gdf.parquet',
                    exp_data_dir / 'regions_8_gdf.parquet',
                    exp_data_dir / 'regions_9_gdf.parquet',
                    exp_data_dir / 'regions_10_gdf.parquet',
                    exp_data_dir / 'building_density_res8_preprocessed.parquet',
                    exp_data_dir / 'building_density_res9_preprocessed.parquet',
                    exp_data_dir / 'building_density_res10_preprocessed.parquet'
                ]
                
                exp_files_exist = all(file.exists() for file in exp_required_files)
                
                if exp_files_exist:
                    logger.info(f"Found experiment data at {exp_data_dir}, using as variant")
                    # Update city name to use the variant
                    self.config['city_name'] = variant_name
                    return
                else:
                    if 'fsi_threshold' in self.config:
                        logger.warning(f"FSI threshold variant (FSI >= {threshold}) does not exist at {variant_dir} or {exp_data_dir}")
                        logger.warning("Please run the modular preprocessing_auxiliary_data scripts to create the variant")
                        raise FileNotFoundError(f"FSI variant directory not found: {variant_dir}")
                    else:
                        logger.warning(f"Threshold variant ({threshold}%) does not exist at {variant_dir} or {exp_data_dir}")
                        logger.warning("Threshold preprocessing_auxiliary_data not implemented for percentage thresholds")
                        raise FileNotFoundError(f"Threshold variant directory not found: {variant_dir}")
            else:
                logger.info(f"Using existing threshold variant: {variant_name}")

            # Update city name to threshold variant
            self.config['city_name'] = variant_name

    def load_data(self) -> Tuple[gpd.GeoDataFrame, Dict[int, gpd.GeoDataFrame], Dict[int, List[str]]]:
        """Load geographic data with proper density information."""
        logger.info("Loading geographic data...")

        # Load study area - check both standard and experiment locations
        area_file = self.data_dir / self.config['city_name'] / 'area_study_gdf.parquet'
        if not area_file.exists():
            # Try experiments directory
            exp_area_file = self.project_dir / 'experiments' / self.config['city_name'] / 'data' / 'area_study_gdf.parquet'
            if exp_area_file.exists():
                area_file = exp_area_file
            else:
                raise FileNotFoundError(f"Study area file not found at {area_file} or {exp_area_file}")
        
        area_gdf = gpd.read_parquet(area_file)
        logger.info(f"Study area CRS: {area_gdf.crs}")

        hex_indices_by_res = {}
        regions_by_res = {}

        for resolution in self.config['modes'].keys():
            regions_file = self.data_dir / self.config['city_name'] / f'regions_{resolution}_gdf.parquet'
            density_file = self.data_dir / self.config['city_name'] / f'building_density_res{resolution}_preprocessed.parquet'
            
            # Check if files exist, if not try experiments directory
            if not regions_file.exists():
                exp_regions_file = self.project_dir / 'experiments' / self.config['city_name'] / 'data' / f'regions_{resolution}_gdf.parquet'
                if exp_regions_file.exists():
                    regions_file = exp_regions_file
                    
            if not density_file.exists():
                exp_density_file = self.project_dir / 'experiments' / self.config['city_name'] / 'data' / f'building_density_res{resolution}_preprocessed.parquet'
                if exp_density_file.exists():
                    density_file = exp_density_file

            regions_gdf = gpd.read_parquet(regions_file)
            logger.info(f"Resolution {resolution} regions shape: {regions_gdf.shape}")

            try:
                if density_file.exists():
                    # Load density data
                    density_df = pd.read_parquet(density_file)

                    # Log density statistics
                    logger.info(f"Resolution {resolution} density statistics:")
                    logger.info(f"FSI range: [{density_df['FSI_24'].min():.2f}, {density_df['FSI_24'].max():.2f}]")
                    logger.info(f"Mean FSI: {density_df['FSI_24'].mean():.2f}")

                    # If regions_gdf already has these columns, drop them before joining
                    if 'FSI_24' in regions_gdf.columns:
                        regions_gdf = regions_gdf.drop(columns=['FSI_24'])
                    if 'in_study_area' in regions_gdf.columns:
                        regions_gdf = regions_gdf.drop(columns=['in_study_area'])

                    # Join density data with regions
                    regions_gdf = regions_gdf.join(
                        density_df[['FSI_24', 'in_study_area']],
                        how='left'
                    )

                    # Fill any NaN values
                    regions_gdf['FSI_24'] = regions_gdf['FSI_24'].fillna(0.0)
                    regions_gdf['in_study_area'] = regions_gdf['in_study_area'].fillna(False)
                else:
                    logger.warning(f"Density file not found: {density_file}")
                    regions_gdf['in_study_area'] = False
                    regions_gdf['FSI_24'] = 0.0
            except Exception as e:
                logger.error(f"Error loading density data for resolution {resolution}: {str(e)}")
                regions_gdf['in_study_area'] = False
                regions_gdf['FSI_24'] = 0.0

            regions_by_res[resolution] = regions_gdf
            hex_indices_by_res[resolution] = list(regions_gdf.index)

            logger.info(f"Loaded resolution {resolution}: {len(regions_gdf)} regions")

        return area_gdf, regions_by_res, hex_indices_by_res

    def load_features(self, hex_indices: List[str]) -> Dict[str, pd.DataFrame]:
        """Load and validate modality features."""
        logger.info("Loading modality features...")

        features = {}
        sources = {
            'gtfs': 'gtfs/embeddings_GTFS_10',
            'roadnetwork': 'road_network/embeddings_roadnetwork_10', 
            'aerial_alphaearth': 'aerial_alphaearth/embeddings_AlphaEarth/processed/embeddings_aerial_10_alphaearth',
            'poi': 'poi_hex2vec/embeddings_POI_hex2vec_10'
        }

        for modality, filename in sources.items():
            path = self.embeddings_dir / f'{filename}.parquet'
            if path.exists():
                df = pd.read_parquet(path)
                
                # Handle different index formats
                if 'h3_index' in df.columns and df.index.name != 'h3_index':
                    # Set h3_index as index if it's a column
                    df = df.set_index('h3_index')
                
                # For AlphaEarth, drop non-embedding columns
                if modality == 'aerial_alphaearth' and 'geometry' in df.columns:
                    # Keep only embedding columns (embed_0, embed_1, etc.)
                    embedding_cols = [col for col in df.columns if col.startswith('embed_')]
                    df = df[embedding_cols]
                
                df = df.reindex(hex_indices, fill_value=0)
                features[modality] = df

                # Log feature statistics
                logger.info(f"\nLoaded {modality} features:")
                logger.info(f"Shape: {df.shape}")
                logger.info(f"Range: [{df.values.min():.3f}, {df.values.max():.3f}]")
                logger.info(f"Mean: {df.values.mean():.3f}")
                logger.info(f"Std: {df.values.std():.3f}")
            else:
                logger.warning(f"Feature file not found: {path}")

        if not features:
            raise ValueError("No feature files were loaded successfully")

        return features

    def run(self) -> Dict[int, pd.DataFrame]:
        """Run the complete pipeline with enhanced logging and validation."""
        try:
            # Load data
            area_gdf, regions_by_res, hex_indices_by_res = self.load_data()
            raw_features = self.load_features(hex_indices_by_res[10])

            logger.info(f"Loaded raw features with keys: {list(raw_features.keys())}")

            # Load mappings
            mappings = self.feature_processor.load_cross_scale_mappings(
                city_name=self.config['city_name'],
                resolutions=sorted(self.config['modes'].keys())
            )

            # Process features
            features = self.feature_processor.fit_transform(
                {name: df.values for name, df in raw_features.items()},
                self.config['city_name']
            )

            # Log feature dimensions after processing_modalities
            for name, feat in features.items():
                logger.info(f"Processed {name} features shape: {feat.shape}")

            # Convert to tensors
            features = {
                name: torch.tensor(feat, dtype=torch.float32).to(self.device)
                for name, feat in features.items()
            }

            logger.info("Constructing graphs...")
            edge_features = self.graph_constructor.construct_graphs(
                self.project_dir,
                self.config['city_name'],
                hex_indices_by_res,
                regions_by_res
            )

            # Process edge features properly for model input
            edge_indices = {}
            edge_weights = {}

            for res, ef in edge_features.items():
                # Get edge data
                edge_indices[res] = ef.index
                edge_weights[res] = ef.accessibility

                # Ensure on correct device
                if edge_indices[res].device != self.device:
                    edge_indices[res] = edge_indices[res].to(self.device)
                if edge_weights[res].device != self.device:
                    edge_weights[res] = edge_weights[res].to(self.device)

                # Normalize weights
                edge_weights[res] = edge_weights[res] / (edge_weights[res].max() + 1e-8)

                # Log statistics
                logger.info(f"\nGraph statistics for resolution {res}:")
                logger.info(f"Number of edges: {edge_indices[res].shape[1]}")
                logger.info(f"Edge weight range: [{edge_weights[res].min().item():.3f}, {edge_weights[res].max().item():.3f}]")
                logger.info(f"Mean edge weight: {edge_weights[res].mean().item():.3f}")

                # Compute and log degree statistics
                num_nodes = len(hex_indices_by_res[res])
                degree_stats = self._analyze_graph_degrees(edge_indices[res], num_nodes)
                logger.info(f"Degree statistics:")
                logger.info(f"- Min degree: {degree_stats['min_degree']:.1f}")
                logger.info(f"- Max degree: {degree_stats['max_degree']:.1f}")
                logger.info(f"- Mean degree: {degree_stats['mean_degree']:.1f}")
                logger.info(f"- Median degree: {degree_stats['median_degree']:.1f}")
                logger.info(f"- Isolated nodes: {degree_stats['isolated_nodes']}")
                logger.info(f"- Average node degree: {2 * edge_indices[res].shape[1] / num_nodes:.1f}")

            # Initialize model trainer
            logger.info("Initializing model trainer...")
            self.model_trainer = UrbanModelTrainer(
                model_config={
                    'feature_dims': self.feature_processor.feature_dims,
                    **self.config['model']
                },
                loss_weights=self.config['training']['loss_weights'],
                city_name=self.config['city_name'],
                checkpoint_dir=self.cache_dir / 'checkpoints'
            )

            logger.info("Starting training...")
            embeddings_by_res = None

            # Skip WandB initialization for faster testing
            train_params = {
                'features_dict': features,
                'edge_indices': edge_indices,
                'edge_weights': edge_weights,
                'mappings': mappings,
                'num_epochs': self.config['training']['num_epochs'],
                'learning_rate': self.config['training']['learning_rate'],
                'warmup_epochs': self.config['training']['warmup_epochs'],
                'patience': self.config['training']['patience'],
                'gradient_clip': self.config['training']['gradient_clip']
            }

            try:
                embeddings, _ = self.model_trainer.train(**train_params)
                if embeddings is None:
                    raise ValueError("Training failed to produce embeddings")

                # Process results
                embeddings_by_res = {}
                embeddings_tensors = {}  # Store tensor versions for saving

                for res, emb in embeddings.items():
                    # Store tensor version for saving
                    embeddings_tensors[res] = emb

                    # Convert to numpy and create DataFrame
                    emb = emb.detach().cpu().numpy()
                    emb_df = pd.DataFrame(
                        emb,
                        index=hex_indices_by_res[res],
                        columns=[f'emb_{i}' for i in range(emb.shape[1])]
                    )
                    embeddings_by_res[res] = emb_df

                    # Log embedding statistics
                    logger.info(f"\nEmbedding statistics for resolution {res}:")
                    logger.info(f"Shape: {emb.shape}")
                    logger.info(f"Range: [{emb.min():.3f}, {emb.max():.3f}]")
                    logger.info(f"Mean: {emb.mean():.3f}")
                    logger.info(f"Std: {emb.std():.3f}")

                # Save and visualize results only if we have embeddings
                if embeddings_by_res:
                    try:
                        logger.info("Saving embeddings...")
                        saved_paths = self.analyzer.save_embeddings(
                            embeddings_tensors,  # Use tensor versions for saving
                            hex_indices_by_res,
                            self.config['modes']
                        )

                        logger.info("Creating cluster visualizations...")
                        self.analyzer.plot_clusters(
                            area_gdf,
                            regions_by_res,
                            embeddings_by_res,  # Use DataFrame versions for plotting
                            n_clusters=self.config['visualization']['n_clusters']
                        )
                    except Exception as viz_error:
                        logger.error(f"Visualization/saving failed: {str(viz_error)}")
                        logger.error("Visualization error details:", exc_info=True)
                        # Continue since we still want to return the embeddings

            except Exception as e:
                logger.error(f"Training failed with error: {str(e)}")
                logger.error("Training error details:", exc_info=True)
                return None

            return embeddings_by_res

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            logger.error("Pipeline error details:", exc_info=True)
            raise

    def _analyze_graph_degrees(self, edge_indices: torch.Tensor, num_nodes: int) -> Dict[str, float]:
        """Analyze degree statistics of the graph."""
        edges = edge_indices.cpu().numpy()
        unique_sources, source_counts = np.unique(edges[0], return_counts=True)
        unique_targets, target_counts = np.unique(edges[1], return_counts=True)

        all_nodes = np.concatenate([unique_sources, unique_targets])
        all_counts = np.concatenate([source_counts, target_counts])
        node_degrees = pd.Series(all_counts, index=all_nodes).groupby(level=0).sum()

        return {
            'min_degree': float(node_degrees.min()),
            'max_degree': float(node_degrees.max()),
            'mean_degree': float(node_degrees.mean()),
            'median_degree': float(node_degrees.median()),
            'isolated_nodes': num_nodes - len(node_degrees)
        }


    @staticmethod
    def create_default_config(
            city_name: str = "south_holland",
            threshold: Optional[int] = None
    ) -> dict:
        """
        Create default configuration with separate city and threshold.

        Args:
            city_name: Base city name (e.g. "south_holland")
            threshold: Optional threshold percentage (e.g. 50)
        """
        config_dict = {
            "city_name": city_name,
            "project_dir": r"D:\Projects\UrbanRepML",
        }

        if threshold is not None:
            config_dict["threshold"] = threshold

        # Add standard config
        config_dict.update({
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
                    'walk': 300,
                    'bike': 450,
                    'drive': 600
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
                "hidden_dim": 128,
                "output_dim": 32,
                "num_convs": 6
            },
            "training": {
                "learning_rate": 1e-5,
                "num_epochs": 10000,
                "warmup_epochs": 1000,
                "patience": 100,
                "gradient_clip": 1.0,
                "loss_weights": {
                    "reconstruction": 1,
                    "consistency": 3
                }
            },
            "visualization": {
                "n_clusters": {8: 8, 9: 8, 10: 8},
                "cmap": "Accent",
                "dpi": 600,
                "figsize": (12, 12)
            },
            "modes": {
                8: 'drive',
                9: 'bike',
                10: 'walk'
            },
            "wandb_project": "urban-embedding",
            "debug": True
        })

        return config_dict

    @staticmethod
    def create_south_holland_fsi01_config() -> dict:
        """
        Create configuration for South Holland with FSI threshold 0.1 and AlphaEarth embeddings.
        """
        config_dict = {
            "city_name": "south_holland",
            "project_dir": r"C:\Users\Bert Berkers\PycharmProjects\UrbanRepML",
            "fsi_threshold": 0.1,
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
                    'walk': 300,
                    'bike': 450,
                    'drive': 600
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
                "hidden_dim": 128,
                "output_dim": 32,
                "num_convs": 6
            },
            "training": {
                "learning_rate": 1e-5,
                "num_epochs": 10000,
                "warmup_epochs": 1000,
                "patience": 100,
                "gradient_clip": 1.0,
                "loss_weights": {
                    "reconstruction": 1,
                    "consistency": 3
                }
            },
            "visualization": {
                "n_clusters": {8: 8, 9: 8, 10: 8},
                "cmap": "Accent",
                "dpi": 600,
                "figsize": (12, 12)
            },
            "modes": {
                8: 'drive',
                9: 'bike',
                10: 'walk'
            },
            "wandb_project": "urban-embedding-south-holland-fsi01",
            "debug": True
        }

        return config_dict

    @staticmethod
    def create_hexagonal_lattice_config(
            city_name: str = "south_holland",
            neighbor_rings: int = 1,
            edge_weight: float = 1.0
    ) -> dict:
        """
        Create configuration for hexagonal lattice experiments.
        
        Args:
            city_name: Base city name (e.g. "south_holland")  
            neighbor_rings: Number of hexagonal neighbor rings to connect
            edge_weight: Uniform edge weight for all connections
        """
        config_dict = {
            "city_name": city_name,
            "project_dir": r"C:\Users\Bert Berkers\PycharmProjects\UrbanRepML",
            "graph_type": "hexagonal_lattice",
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
            "hexagonal": {
                "neighbor_rings": neighbor_rings,
                "edge_weight": edge_weight,
                "include_self_loops": False
            },
            "model": {
                "hidden_dim": 128,
                "output_dim": 32,
                "num_convs": 6
            },
            "training": {
                "learning_rate": 1e-4,
                "num_epochs": 1000,
                "warmup_epochs": 100,
                "patience": 100,
                "gradient_clip": 1.0,
                "loss_weights": {
                    "reconstruction": 1,
                    "consistency": 3
                }
            },
            "visualization": {
                "n_clusters": {8: 8, 9: 8, 10: 8},
                "cmap": "Accent", 
                "dpi": 600,
                "figsize": (12, 12)
            },
            "modes": {
                8: 'drive',
                9: 'bike', 
                10: 'walk'
            },
            "wandb_project": "urban-embedding-hexagonal",
            "debug": True
        }

        return config_dict

def main():
    """Main entry point using separate city and threshold variables."""
    try:
        logger.info("Starting Urban Embedding Pipeline")

        # Create config with separate variables
        config = UrbanEmbeddingPipeline.create_default_config(
            city_name="south_holland",
            threshold=50
        )

        # Log key configuration parameters
        logger.info("\nKey configuration parameters:")
        logger.info(f"City: {config['city_name']}")
        logger.info(f"Feature processing:")
        logger.info(f"- Variance threshold: {config['feature_processing']['pca']['variance_threshold']}")
        logger.info(f"- Min components: {config['feature_processing']['pca']['min_components']}")
        logger.info(f"- Max components: {config['feature_processing']['pca']['max_components']}")
        logger.info(f"\nGraph construction:")
        logger.info(f"- Search radii: {config['graph']['search_radius']}")
        logger.info(f"- Max travel times: {config['graph']['max_travel_time']}")
        logger.info(f"- Beta values: {config['graph']['beta']}")

        # Initialize and run pipeline
        logger.info("\nInitializing pipeline...")
        pipeline = UrbanEmbeddingPipeline(config)

        logger.info("Running pipeline...")
        embeddings = pipeline.run()
        return embeddings

    except Exception as e:
        logger.error("Pipeline failed:", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Pipeline failed:")
        raise

