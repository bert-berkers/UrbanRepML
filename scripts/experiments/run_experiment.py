#!/usr/bin/env python3
"""
Experiment orchestrator for UrbanRepML.
Runs the complete pipeline from region setup to model training.
"""

import sys
import subprocess
import argparse
from pathlib import Path
import json
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def run_command(cmd: list, description: str):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"[RUNNING] {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"[ERROR] Command failed after {elapsed:.1f}s")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Command failed: {description}")
    
    print(result.stdout)
    print(f"[SUCCESS] Completed in {elapsed:.1f}s")
    return result


def check_existing_data(path: Path, description: str) -> bool:
    """Check if data already exists."""
    if path.exists():
        print(f"[EXISTS] {description} already exists at {path}")
        return True
    return False


def run_experiment(args):
    """Run the complete experiment pipeline."""
    
    print("\n" + "="*70)
    print(f"URBANREPML EXPERIMENT: {args.experiment_name}")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"City: {args.city}")
    print(f"FSI Filtering: {'Percentile ' + str(args.fsi_percentile) + '%' if args.fsi_percentile else 'Absolute ' + str(args.fsi_threshold)}")
    print(f"Resolutions: {args.resolutions}")
    
    # Setup paths
    base_data_dir = Path(f"data/preprocessed [TODO SORT & CLEAN UP]/{args.city}_base")
    experiment_dir = Path(f"experiments/{args.experiment_name}")
    experiment_data_dir = experiment_dir / "data"
    experiment_graphs_dir = experiment_dir / "graphs"
    
    # Create experiment directory
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Setup regions (if needed)
    if not args.skip_regions:
        regions_marker = base_data_dir / "regions" / f"{args.city}_res8.parquet"
        if check_existing_data(regions_marker, "Region data") and not args.force:
            print("[SKIP] Region setup - data already exists")
        else:
            run_command([
                sys.executable, "scripts/preprocessing auxiliary data/setup_regions.py",
                "--city_name", args.city,
                "--output_dir", str(base_data_dir),
                "--resolutions", args.resolutions
            ], "Setting up H3 regions")
    
    # Step 2: Calculate density (if needed)
    if not args.skip_density:
        density_marker = base_data_dir / "density" / f"{args.city}_res8_density.parquet"
        if check_existing_data(density_marker, "Density data") and not args.force:
            print("[SKIP] Density calculation - data already exists")
        else:
            building_data = Path(args.building_data)
            if not building_data.exists():
                print(f"[ERROR] Building data not found: {building_data}")
                print("[INFO] Please provide the path to the building shapefile using --building_data")
                return False
            
            run_command([
                sys.executable, "scripts/preprocessing auxiliary data/setup_density.py",
                "--city_name", args.city,
                "--input_dir", str(base_data_dir),
                "--output_dir", str(base_data_dir),
                "--building_data", str(building_data),
                "--resolutions", args.resolutions
            ], "Calculating building density")
    
    # Step 3: Apply FSI filtering
    if not args.skip_filter:
        filter_marker = experiment_data_dir / "metadata.json"
        if check_existing_data(filter_marker, "FSI filtered data") and not args.force:
            print("[SKIP] FSI filtering - data already exists")
        else:
            filter_cmd = [
                sys.executable, "scripts/preprocessing auxiliary data/setup_fsi_filter.py",
                "--city_name", args.city,
                "--input_dir", str(base_data_dir),
                "--output_dir", str(experiment_data_dir),
                "--resolutions", args.resolutions,
                "--experiment_name", args.experiment_name
            ]
            
            if args.fsi_percentile:
                filter_cmd.extend(["--fsi_percentile", str(args.fsi_percentile)])
            elif args.fsi_threshold:
                filter_cmd.extend(["--fsi_threshold", str(args.fsi_threshold)])
            
            run_command(filter_cmd, f"Applying FSI filtering")
    
    # Step 4: Generate accessibility graphs
    if not args.skip_graphs:
        graphs_marker = experiment_graphs_dir / "hierarchical_mapping.pkl"
        if check_existing_data(graphs_marker, "Accessibility graphs") and not args.force:
            print("[SKIP] Graph generation - graphs already exist")
        else:
            run_command([
                sys.executable, "scripts/preprocessing auxiliary data/setup_hierarchical_graphs.py",
                "--data_dir", str(experiment_data_dir),
                "--output_dir", str(experiment_graphs_dir),
                "--city_name", args.experiment_name,
                "--resolutions", args.resolutions,
                "--fsi_threshold", str(args.graph_fsi_threshold),
                "--cutoff_time", str(args.cutoff_time),
                "--percentile_threshold", str(args.percentile_threshold)
            ], "Generating accessibility graphs")
    
    # Step 5: Run the UrbanEmbedding pipeline (if requested)
    if args.run_training:
        print("\n" + "="*60)
        print("[TRAINING] Running UrbanEmbedding pipeline")
        print("="*60)
        
        # Import the pipeline
        from urban_embedding import UrbanEmbeddingPipeline
        
        # Create configuration
        config = {
            'city_name': args.experiment_name,
            'project_dir': str(project_root),
            'fsi_threshold': None,  # Don't use pipeline's built-in FSI filtering
            'modes': {8: 'drive', 9: 'bike', 10: 'walk'},
            'feature_processing': {
                'pca': {
                    'variance_threshold': 0.95,
                    'min_components': {'aerial': 50, 'poi': 20, 'gtfs': 10, 'road': 10},
                    'max_components': 100
                }
            },
            'graph': {
                'speeds': {'walk': 1.4, 'bike': 4.17, 'drive': 11.11},
                'max_travel_time': {'walk': 900, 'bike': 900, 'drive': 900},
                'search_radius': {'walk': 1200, 'bike': 3000, 'drive': 10000},
                'beta': {'walk': 0.002, 'bike': 0.0012, 'drive': 0.0008}
            },
            'model': {
                'hidden_dim': getattr(args, 'hidden_dim', 128),
                'num_convs': getattr(args, 'num_convs', 4),
                'dropout': 0.1
            },
            'training': {
                'epochs': args.epochs,
                'batch_size': 1,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'loss_weights': {'reconstruction': 1, 'consistency': 3}
            },
            'visualization': {
                'n_clusters': {8: 8, 9: 8, 10: 8},
                'cmap': 'Accent',
                'dpi': 600,
                'figsize': (12, 12)
            },
            'wandb_project': f"urban-embedding-{args.experiment_name}",
            'debug': args.debug
        }
        
        # Save configuration
        config_path = experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[SAVE] Configuration saved to {config_path}")
        
        # Initialize and run pipeline
        try:
            pipeline = UrbanEmbeddingPipeline(config)
            embeddings = pipeline.run()
            print("[SUCCESS] Training completed!")
            return True
        except Exception as e:
            print(f"[ERROR] Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # Generate summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Status: {'Training Complete' if args.run_training else 'Data Prepared'}")
    print(f"\nOutputs:")
    print(f"  Data: {experiment_data_dir}")
    print(f"  Graphs: {experiment_graphs_dir}")
    if args.run_training:
        print(f"  Results: {experiment_dir / 'results'}")
    
    # Save metadata
    metadata = {
        'experiment_name': args.experiment_name,
        'city': args.city,
        'timestamp': datetime.now().isoformat(),
        'fsi_filtering': {
            'percentile': args.fsi_percentile,
            'threshold': args.fsi_threshold
        },
        'resolutions': args.resolutions,
        'training_completed': args.run_training,
        'epochs': args.epochs if args.run_training else None
    }
    
    metadata_path = experiment_dir / "experiment_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[SAVE] Experiment metadata: {metadata_path}")
    
    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run UrbanRepML experiment')
    
    # Basic configuration
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name for this experiment')
    parser.add_argument('--city', type=str, default='south_holland',
                        help='City/region name')
    parser.add_argument('--resolutions', type=str, default='8,9,10',
                        help='Comma-separated H3 resolutions')
    
    # FSI filtering
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fsi_percentile', type=float,
                       help='FSI percentile threshold (e.g., 95)')
    group.add_argument('--fsi_threshold', type=float,
                       help='Absolute FSI threshold (e.g., 0.1)')
    
    # Data paths
    parser.add_argument('--building_data', type=str,
                        default='data/preprocessed [TODO SORT & CLEAN UP]/density/PV28__00_Basis_Bouwblok.shp',
                        help='Path to building data shapefile')
    
    # Graph parameters
    parser.add_argument('--graph_fsi_threshold', type=float, default=0.1,
                        help='FSI threshold for graph active hexagons')
    parser.add_argument('--cutoff_time', type=int, default=300,
                        help='Travel time cutoff in seconds')
    parser.add_argument('--percentile_threshold', type=float, default=90,
                        help='Edge filtering percentile')
    
    # Training parameters
    parser.add_argument('--run_training', action='store_true',
                        help='Run the full training pipeline')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Model hidden dimension')
    parser.add_argument('--num_convs', type=int, default=4,
                        help='Number of GCN convolution layers')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    # Control flow
    parser.add_argument('--skip_regions', action='store_true',
                        help='Skip region setup')
    parser.add_argument('--skip_density', action='store_true',
                        help='Skip density calculation')
    parser.add_argument('--skip_filter', action='store_true',
                        help='Skip FSI filtering')
    parser.add_argument('--skip_graphs', action='store_true',
                        help='Skip graph generation')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration of existing data')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if not args.fsi_percentile and not args.fsi_threshold:
        print("[ERROR] Either --fsi_percentile or --fsi_threshold must be specified")
        return 1
    
    try:
        success = run_experiment(args)
        return 0 if success else 1
    except Exception as e:
        print(f"\n[FATAL ERROR] Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())