#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Del Norte Exploratory Analysis - Main Entry Point

Complete pipeline for processing 2021 AlphaEarth data for Del Norte County:
1. Load TIFF tiles and convert to H3 resolution 9 hexagons
2. Perform clustering analysis with multiple methods (K-means, Hierarchical, GMM)
3. Generate SRAI-based visualizations with categorical color schemes
4. Save all results with proper naming convention (2021_res9)

Usage:
    python run_analysis.py                    # Run full pipeline
    python run_analysis.py --clustering-only  # Skip data loading
    python run_analysis.py --viz-only         # Skip to visualization
"""

import sys
import logging
import argparse
from pathlib import Path
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import our analysis modules
from scripts.load_alphaearth import AlphaEarthToH3Converter
from scripts.clustering import MultiMethodClusterer
from scripts.srai_visualizations import SRAIVisualizer


def setup_logging(config: dict) -> logging.Logger:
    """Setup logging for the analysis."""
    log_level = config['output']['log_level']
    log_file = config['output']['log_file']
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Setup logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info(f"Del Norte Exploratory Analysis Started - {datetime.now()}")
    logger.info("="*60)
    
    return logger


def check_data_availability(config: dict) -> bool:
    """Check if source TIFF data is available."""
    source_dir = Path(config['data']['source_dir'])
    pattern = config['data']['pattern']
    
    if not source_dir.exists():
        return False
    
    tiff_files = list(source_dir.glob(pattern))
    return len(tiff_files) > 0


def check_h3_data_exists(config: dict) -> bool:
    """Check if H3 processed data already exists."""
    output_file = config['data']['output_file']
    res = config['data']['h3_resolution']
    data_path = Path(f"data/h3_2021_res{res}/{output_file}")
    return data_path.exists()


def check_clustering_results_exist() -> bool:
    """Check if clustering results already exist."""
    results_dir = Path("results/clusters")
    if not results_dir.exists():
        return False
    
    res = 9  # Using resolution 9
    parquet_files = list(results_dir.glob(f"*_2021_res{res}_*.parquet"))
    return len(parquet_files) > 0


def run_data_loading(config: dict, logger: logging.Logger) -> bool:
    """Run the data loading pipeline."""
    logger.info("Starting AlphaEarth TIFF to H3 conversion...")
    
    # Check if data already exists
    if check_h3_data_exists(config):
        logger.info("H3 data already exists. Skipping data loading.")
        return True
    
    # Check source data availability
    if not check_data_availability(config):
        logger.error(f"Source TIFF data not found at {config['data']['source_dir']}")
        logger.error("Please check the path to your AlphaEarth_Cascadia folder")
        return False
    
    # Initialize converter
    converter = AlphaEarthToH3Converter(config)
    
    # Process all files
    hex_df = converter.process_all()
    
    if hex_df is not None and not hex_df.empty:
        # Save results
        res = config['data']['h3_resolution']
        output_path = Path(f"data/h3_2021_res{res}/{config['data']['output_file']}")
        converter.save_results(hex_df, output_path)
        logger.info(f"Successfully processed {len(hex_df)} H3 hexagons")
        return True
    else:
        logger.error("Failed to process TIFF data")
        return False


def run_clustering_analysis(config: dict, logger: logging.Logger) -> bool:
    """Run the clustering analysis pipeline."""
    logger.info("Starting clustering analysis...")
    
    # Check if clustering results already exist
    if check_clustering_results_exist():
        logger.info("Clustering results already exist. Skipping clustering analysis.")
        return True
    
    # Check if H3 data exists
    if not check_h3_data_exists(config):
        logger.error("H3 data not found. Please run data loading first.")
        return False
    
    # Initialize clusterer
    clusterer = MultiMethodClusterer(config)
    
    # Load H3 data
    res = config['data']['h3_resolution']
    data_path = Path(f"data/h3_2021_res{res}/{config['data']['output_file']}")
    df = clusterer.load_h3_data(data_path)
    
    # Run all clustering methods
    results = clusterer.run_all_clustering(df)
    
    if results:
        logger.info("Clustering analysis completed successfully")
        logger.info(f"Generated results for: {list(results.keys())}")
        return True
    else:
        logger.error("Failed to run clustering analysis")
        return False


def run_visualization(config: dict, logger: logging.Logger) -> bool:
    """Run the visualization pipeline."""
    logger.info("Starting visualization generation...")
    
    # Check if clustering results exist
    if not check_clustering_results_exist():
        logger.error("No clustering results found. Please run clustering analysis first.")
        return False
    
    # Initialize visualizer
    visualizer = SRAIVisualizer(config)
    
    # Generate all visualizations
    try:
        visualizer.visualize_all_results()
        logger.info("Visualization generation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        return False


def print_results_summary(config: dict, logger: logging.Logger):
    """Print a summary of generated results."""
    logger.info("="*60)
    logger.info("ANALYSIS COMPLETE - RESULTS SUMMARY")
    logger.info("="*60)
    
    # Check H3 data
    res = config['data']['h3_resolution']
    h3_path = Path(f"data/h3_2021_res{res}/{config['data']['output_file']}")
    if h3_path.exists():
        import pandas as pd
        df = pd.read_parquet(h3_path)
        logger.info(f"H3 Data: {len(df)} hexagons at resolution {res}")
        logger.info(f"         64-dimensional AlphaEarth embeddings preserved")
    
    # Check clustering results
    results_dir = Path("results/clusters")
    if results_dir.exists():
        res = 9  # Using resolution 9
        parquet_files = list(results_dir.glob(f"*_2021_res{res}_*.parquet"))
        json_files = list(results_dir.glob(f"*_2021_res{res}_*.json"))
        logger.info(f"Clustering: {len(parquet_files)} clustering configurations")
        logger.info(f"           {len(json_files)} metric files")
    
    # Check visualizations
    plots_dir = Path("plots")
    if plots_dir.exists():
        spatial_plots = list(plots_dir.glob("spatial/*.png"))
        dist_plots = list(plots_dir.glob("distributions/*.png"))
        comp_plots = list(plots_dir.glob("comparisons/*.png"))
        
        total_plots = len(spatial_plots) + len(dist_plots) + len(comp_plots)
        logger.info(f"Visualizations: {total_plots} plots generated")
        logger.info(f"               {len(spatial_plots)} spatial maps")
        logger.info(f"               {len(dist_plots)} distribution/feature plots")
        logger.info(f"               {len(comp_plots)} comparison plots")
    
    logger.info("="*60)
    logger.info("File Structure:")
    logger.info(f"  data/h3_2021_res{res}/           - Processed H3 data")
    logger.info("  results/clusters/            - Clustering results")
    logger.info("  results/stats/               - Analysis statistics")
    logger.info("  plots/spatial/               - Spatial maps (multiple colors)")
    logger.info("  plots/distributions/         - Statistical plots")
    logger.info("  plots/comparisons/           - Method comparisons")
    logger.info("="*60)


def main():
    """Main entry point for Del Norte exploratory analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Del Norte Exploratory Analysis")
    parser.add_argument("--data-only", action="store_true", 
                       help="Run only data loading step")
    parser.add_argument("--clustering-only", action="store_true", 
                       help="Run only clustering analysis step")
    parser.add_argument("--viz-only", action="store_true", 
                       help="Run only visualization step")
    parser.add_argument("--skip-data", action="store_true", 
                       help="Skip data loading step")
    parser.add_argument("--skip-clustering", action="store_true", 
                       help="Skip clustering step")
    parser.add_argument("--skip-viz", action="store_true", 
                       help="Skip visualization step")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Print configuration summary
    logger.info("Configuration Summary:")
    logger.info(f"  Year: {config['experiment']['year']}")
    logger.info(f"  H3 Resolution: {config['experiment']['h3_resolution']}")
    logger.info(f"  Dimensions: {config['experiment']['keep_dimensions']}")
    logger.info(f"  Source: {config['data']['source_dir']}")
    
    success = True
    
    try:
        # Step 1: Data Loading
        if not (args.clustering_only or args.viz_only or args.skip_data):
            success &= run_data_loading(config, logger)
            if not success:
                logger.error("Data loading failed. Stopping analysis.")
                return 1
        elif args.data_only:
            success &= run_data_loading(config, logger)
            print_results_summary(config, logger)
            return 0 if success else 1
        
        # Step 2: Clustering Analysis
        if not (args.data_only or args.viz_only or args.skip_clustering):
            success &= run_clustering_analysis(config, logger)
            if not success:
                logger.error("Clustering analysis failed. Stopping analysis.")
                return 1
        elif args.clustering_only:
            success &= run_clustering_analysis(config, logger)
            print_results_summary(config, logger)
            return 0 if success else 1
        
        # Step 3: Visualization
        if not (args.data_only or args.clustering_only or args.skip_viz):
            success &= run_visualization(config, logger)
            if not success:
                logger.error("Visualization generation failed.")
                return 1
        elif args.viz_only:
            success &= run_visualization(config, logger)
            print_results_summary(config, logger)
            return 0 if success else 1
        
        # Print final summary
        if success:
            print_results_summary(config, logger)
            logger.info("Del Norte Exploratory Analysis completed successfully!")
            return 0
        else:
            logger.error("Analysis completed with errors.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())