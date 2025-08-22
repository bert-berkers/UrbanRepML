#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Del Norte County Active Inference Experiment

Main experiment runner that orchestrates the complete pipeline:
1. Load AlphaEarth satellite embeddings
2. Process through active inference framework
3. Calculate information-theoretic metrics
4. Generate synthetic data for gaps
5. Analyze and visualize results
"""

import sys
import logging
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import custom modules
from scripts.del_norte.active_inference_processor import DelNorteActiveInferenceProcessor
from urban_embedding.information_gain import InformationMetrics, SpatialInformationCalculator

# Setup logging
log_dir = Path("experiments/del_norte_active_inference/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DelNorteExperimentRunner:
    """
    Orchestrates the complete Del Norte active inference experiment.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = "experiments/del_norte_active_inference/config.yaml"
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.experiment_dir = Path("experiments/del_norte_active_inference")
        self.results_dir = self.experiment_dir / "results"
        self.analysis_dir = self.experiment_dir / "analysis"
        
        # Create directories
        for dir_path in [self.results_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        self.processor = DelNorteActiveInferenceProcessor(config_path)
        
        # Track experiment metrics
        self.experiment_metrics = {
            'start_time': datetime.now().isoformat(),
            'config_used': config_path,
            'processing_results': {},
            'analysis_results': {},
            'total_runtime': 0.0
        }
        
        logger.info("Del Norte Active Inference Experiment initialized")
    
    def run_processing_pipeline(
        self, 
        data_paths: Optional[Dict[int, Path]] = None,
        years: Optional[List[int]] = None
    ) -> Dict:
        """
        Run the main processing pipeline for all years.
        
        Args:
            data_paths: Dictionary mapping year to data path
            years: List of years to process
            
        Returns:
            Dictionary with processing results
        """
        if years is None:
            years = self.config['alphaearth']['years']
        
        processing_results = {}
        
        for year in years:
            logger.info(f"{'='*60}")
            logger.info(f"Processing Year: {year}")
            logger.info(f"{'='*60}")
            
            # Determine data path for this year
            data_path = None
            if data_paths and year in data_paths:
                data_path = data_paths[year]
            
            try:
                # Run processing for this year
                self.processor.run(data_path, year)
                
                # Store results
                processing_results[year] = {
                    'success': True,
                    'stats': self.processor.processing_stats.copy(),
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Year {year} processing completed successfully")
                
            except Exception as e:
                logger.error(f"Error processing year {year}: {e}")
                processing_results[year] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        self.experiment_metrics['processing_results'] = processing_results
        return processing_results
    
    def analyze_results(self, years: Optional[List[int]] = None) -> Dict:
        """
        Analyze processed results across years.
        
        Args:
            years: Years to analyze
            
        Returns:
            Analysis results dictionary
        """
        if years is None:
            years = self.config['alphaearth']['years']
        
        logger.info("Analyzing experiment results...")
        
        analysis_results = {
            'temporal_analysis': {},
            'spatial_analysis': {},
            'information_analysis': {},
            'free_energy_analysis': {}
        }
        
        # Load all processed data
        yearly_data = {}
        for year in years:
            try:
                data_path = self.results_dir / f"del_norte_embeddings_{year}.parquet"
                if data_path.exists():
                    df = pd.read_parquet(data_path)
                    yearly_data[year] = df
                    logger.info(f"Loaded {len(df)} hexagons for year {year}")
                else:
                    logger.warning(f"No data found for year {year}")
            except Exception as e:
                logger.error(f"Error loading data for year {year}: {e}")
        
        if not yearly_data:
            logger.warning("No data available for analysis")
            return analysis_results
        
        # Temporal analysis
        analysis_results['temporal_analysis'] = self._analyze_temporal_patterns(yearly_data)
        
        # Spatial analysis
        analysis_results['spatial_analysis'] = self._analyze_spatial_patterns(yearly_data)
        
        # Information theory analysis
        analysis_results['information_analysis'] = self._analyze_information_metrics(yearly_data)
        
        # Free energy analysis
        analysis_results['free_energy_analysis'] = self._analyze_free_energy(yearly_data)
        
        # Save analysis results
        analysis_path = self.analysis_dir / "comprehensive_analysis.json"
        with open(analysis_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = self._make_json_serializable(analysis_results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Analysis results saved to {analysis_path}")
        
        return analysis_results
    
    def _analyze_temporal_patterns(self, yearly_data: Dict[int, pd.DataFrame]) -> Dict:
        """Analyze temporal patterns across years."""
        if len(yearly_data) < 2:
            return {"note": "Insufficient years for temporal analysis"}
        
        years = sorted(yearly_data.keys())
        temporal_stats = {}
        
        # Track metrics over time
        metrics = ['entropy', 'free_energy_total', 'information_gain', 'spatial_mutual_info']
        
        for metric in metrics:
            if metric in yearly_data[years[0]].columns:
                values_over_time = []
                for year in years:
                    if metric in yearly_data[year].columns:
                        mean_val = yearly_data[year][metric].mean()
                        values_over_time.append(mean_val)
                    else:
                        values_over_time.append(np.nan)
                
                temporal_stats[metric] = {
                    'values': values_over_time,
                    'trend': self._calculate_trend(values_over_time),
                    'stability': np.std(values_over_time) if len(values_over_time) > 1 else 0
                }
        
        return temporal_stats
    
    def _analyze_spatial_patterns(self, yearly_data: Dict[int, pd.DataFrame]) -> Dict:
        """Analyze spatial patterns and clustering."""
        spatial_stats = {}
        
        for year, df in yearly_data.items():
            if len(df) == 0:
                continue
            
            # Calculate spatial autocorrelation (simplified)
            if 'spatial_mutual_info' in df.columns:
                spatial_autocorr = df['spatial_mutual_info'].corr(df['entropy'])
            else:
                spatial_autocorr = 0.0
            
            # Calculate clustering metrics
            clustering_stats = self._calculate_spatial_clustering(df)
            
            spatial_stats[year] = {
                'total_hexagons': len(df),
                'spatial_autocorrelation': spatial_autocorr,
                'clustering': clustering_stats
            }
        
        return spatial_stats
    
    def _analyze_information_metrics(self, yearly_data: Dict[int, pd.DataFrame]) -> Dict:
        """Analyze information theory metrics."""
        info_stats = {}
        
        for year, df in yearly_data.items():
            if len(df) == 0:
                continue
            
            year_stats = {}
            
            # Entropy analysis
            if 'entropy' in df.columns:
                year_stats['entropy'] = {
                    'mean': df['entropy'].mean(),
                    'std': df['entropy'].std(),
                    'min': df['entropy'].min(),
                    'max': df['entropy'].max(),
                    'distribution': df['entropy'].describe().to_dict()
                }
            
            # Information gain analysis
            if 'information_gain' in df.columns:
                year_stats['information_gain'] = {
                    'total': df['information_gain'].sum(),
                    'mean': df['information_gain'].mean(),
                    'positive_cells': (df['information_gain'] > 0).sum(),
                    'negative_cells': (df['information_gain'] < 0).sum()
                }
            
            # Mutual information analysis
            if 'spatial_mutual_info' in df.columns:
                year_stats['spatial_mutual_info'] = {
                    'mean': df['spatial_mutual_info'].mean(),
                    'std': df['spatial_mutual_info'].std(),
                    'distribution': df['spatial_mutual_info'].describe().to_dict()
                }
            
            info_stats[year] = year_stats
        
        return info_stats
    
    def _analyze_free_energy(self, yearly_data: Dict[int, pd.DataFrame]) -> Dict:
        """Analyze free energy patterns."""
        fe_stats = {}
        
        for year, df in yearly_data.items():
            if len(df) == 0:
                continue
            
            year_stats = {}
            
            # Free energy components
            fe_cols = [col for col in df.columns if col.startswith('free_energy_')]
            
            for col in fe_cols:
                if col in df.columns:
                    year_stats[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
            
            # Expected free energy analysis
            if 'expected_free_energy' in df.columns:
                year_stats['expected_free_energy'] = {
                    'mean': df['expected_free_energy'].mean(),
                    'high_efe_cells': (df['expected_free_energy'] > df['expected_free_energy'].quantile(0.9)).sum(),
                    'distribution': df['expected_free_energy'].describe().to_dict()
                }
            
            fe_stats[year] = year_stats
        
        return fe_stats
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from time series."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array([v for v in values if not np.isnan(v)])
        
        if len(y) < 2:
            return "insufficient_data"
        
        slope = np.polyfit(x[:len(y)], y, 1)[0]
        
        if abs(slope) < 1e-6:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _calculate_spatial_clustering(self, df: pd.DataFrame) -> Dict:
        """Calculate spatial clustering metrics."""
        if len(df) < 3:
            return {"note": "Insufficient data for clustering"}
        
        # Use embedding dimensions for clustering analysis
        embed_cols = [col for col in df.columns if col.startswith('A')]
        
        if not embed_cols:
            return {"note": "No embedding columns found"}
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        try:
            X = df[embed_cols].values
            
            # Try different cluster numbers
            best_k = 2
            best_score = -1
            
            for k in range(2, min(10, len(df))):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            return {
                'optimal_clusters': best_k,
                'silhouette_score': best_score,
                'cluster_sizes': np.bincount(labels).tolist()
            }
            
        except Exception as e:
            return {"error": f"Clustering failed: {e}"}
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def generate_visualizations(self, years: Optional[List[int]] = None):
        """Generate visualizations of results."""
        if years is None:
            years = self.config['alphaearth']['years']
        
        logger.info("Generating visualizations...")
        
        # Create visualization directory
        viz_dir = self.analysis_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Load data
        yearly_data = {}
        for year in years:
            data_path = self.results_dir / f"del_norte_embeddings_{year}.parquet"
            if data_path.exists():
                yearly_data[year] = pd.read_parquet(data_path)
        
        if not yearly_data:
            logger.warning("No data available for visualization")
            return
        
        # Information theory metrics over time
        self._plot_temporal_trends(yearly_data, viz_dir)
        
        # Spatial distributions
        self._plot_spatial_distributions(yearly_data, viz_dir)
        
        # Free energy landscapes
        self._plot_free_energy_landscapes(yearly_data, viz_dir)
        
        # Correlation matrices
        self._plot_correlation_matrices(yearly_data, viz_dir)
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def _plot_temporal_trends(self, yearly_data: Dict, viz_dir: Path):
        """Plot temporal trends of key metrics."""
        years = sorted(yearly_data.keys())
        
        metrics = ['entropy', 'free_energy_total', 'information_gain', 'spatial_mutual_info']
        available_metrics = [m for m in metrics if m in yearly_data[years[0]].columns]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:4]):
            ax = axes[i]
            
            values = []
            for year in years:
                if metric in yearly_data[year].columns:
                    mean_val = yearly_data[year][metric].mean()
                    values.append(mean_val)
                else:
                    values.append(np.nan)
            
            ax.plot(years, values, 'o-', linewidth=2, markersize=6)
            ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "temporal_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_spatial_distributions(self, yearly_data: Dict, viz_dir: Path):
        """Plot spatial distributions of information metrics."""
        # Select a representative year
        year = sorted(yearly_data.keys())[0]
        df = yearly_data[year]
        
        if 'geometry' not in df.columns or len(df) == 0:
            return
        
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(df)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Entropy distribution
        if 'entropy' in gdf.columns:
            gdf.plot(column='entropy', ax=axes[0,0], legend=True, cmap='viridis')
            axes[0,0].set_title('Entropy Distribution')
        
        # Free energy distribution
        if 'free_energy_total' in gdf.columns:
            gdf.plot(column='free_energy_total', ax=axes[0,1], legend=True, cmap='plasma')
            axes[0,1].set_title('Free Energy Distribution')
        
        # Information gain
        if 'information_gain' in gdf.columns:
            gdf.plot(column='information_gain', ax=axes[1,0], legend=True, cmap='coolwarm')
            axes[1,0].set_title('Information Gain Distribution')
        
        # Spatial mutual information
        if 'spatial_mutual_info' in gdf.columns:
            gdf.plot(column='spatial_mutual_info', ax=axes[1,1], legend=True, cmap='inferno')
            axes[1,1].set_title('Spatial Mutual Information')
        
        for ax in axes.flatten():
            ax.set_axis_off()
        
        plt.tight_layout()
        plt.savefig(viz_dir / f"spatial_distributions_{year}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_free_energy_landscapes(self, yearly_data: Dict, viz_dir: Path):
        """Plot free energy components."""
        year = sorted(yearly_data.keys())[0]
        df = yearly_data[year]
        
        fe_cols = [col for col in df.columns if col.startswith('free_energy_')]
        
        if not fe_cols:
            return
        
        fig, axes = plt.subplots(1, len(fe_cols), figsize=(5*len(fe_cols), 5))
        if len(fe_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(fe_cols):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(col.replace('_', ' ').title())
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(viz_dir / f"free_energy_landscapes_{year}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrices(self, yearly_data: Dict, viz_dir: Path):
        """Plot correlation matrices between metrics."""
        year = sorted(yearly_data.keys())[0]
        df = yearly_data[year]
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        info_cols = [col for col in numeric_cols if any(
            keyword in col for keyword in ['entropy', 'free_energy', 'information', 'mutual']
        )]
        
        if len(info_cols) < 2:
            return
        
        corr_matrix = df[info_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, cbar_kws={"shrink": .8})
        plt.title('Information Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(viz_dir / f"correlation_matrix_{year}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_experiment(
        self,
        data_paths: Optional[Dict[int, Path]] = None,
        years: Optional[List[int]] = None
    ) -> Dict:
        """
        Run the complete experiment pipeline.
        
        Args:
            data_paths: Optional data paths for each year
            years: Years to process
            
        Returns:
            Complete experiment results
        """
        start_time = datetime.now()
        
        logger.info("="*80)
        logger.info("STARTING DEL NORTE ACTIVE INFERENCE EXPERIMENT")
        logger.info("="*80)
        
        try:
            # Phase 1: Processing
            logger.info("Phase 1: Data Processing")
            processing_results = self.run_processing_pipeline(data_paths, years)
            
            # Phase 2: Analysis
            logger.info("Phase 2: Results Analysis")
            analysis_results = self.analyze_results(years)
            
            # Phase 3: Visualization
            logger.info("Phase 3: Visualization Generation")
            self.generate_visualizations(years)
            
            # Final metrics
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds()
            
            self.experiment_metrics.update({
                'end_time': end_time.isoformat(),
                'total_runtime': runtime,
                'processing_results': processing_results,
                'analysis_results': analysis_results,
                'success': True
            })
            
            # Save final experiment report
            report_path = self.results_dir / "experiment_report.json"
            with open(report_path, 'w') as f:
                serializable_metrics = self._make_json_serializable(self.experiment_metrics)
                json.dump(serializable_metrics, f, indent=2)
            
            logger.info("="*80)
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info(f"Total runtime: {runtime:.2f} seconds")
            logger.info(f"Report saved to: {report_path}")
            logger.info("="*80)
            
            return self.experiment_metrics
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            self.experiment_metrics.update({
                'end_time': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            })
            
            return self.experiment_metrics


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run Del Norte Active Inference Experiment'
    )
    parser.add_argument(
        '--config',
        default='experiments/del_norte_active_inference/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        help='Years to process (default from config)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Directory containing AlphaEarth data files'
    )
    parser.add_argument(
        '--processing-only',
        action='store_true',
        help='Run processing only (skip analysis and visualization)'
    )
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = DelNorteExperimentRunner(args.config)
    
    # Prepare data paths if provided
    data_paths = None
    if args.data_dir and args.data_dir.exists():
        data_paths = {}
        years = args.years or runner.config['alphaearth']['years']
        for year in years:
            # Look for data files matching pattern
            pattern = f"DelNorte_AlphaEarth_{year}*.tif"
            matches = list(args.data_dir.glob(pattern))
            if matches:
                data_paths[year] = matches[0]
    
    # Run experiment
    if args.processing_only:
        runner.run_processing_pipeline(data_paths, args.years)
    else:
        runner.run_complete_experiment(data_paths, args.years)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())