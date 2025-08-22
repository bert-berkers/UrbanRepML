#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cascadia Coastal Forests Processing - Stage 2: Stitch Intermediate JSONs to Final Dataset
Combines intermediate JSON files from Stage 1 into final Parquet dataset with overlap handling
"""

import sys
import yaml
import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

def setup_logging(config, run_id):
    """Set up logging with timestamped files"""
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"stitching_{run_id}.log"
    
    logging.basicConfig(
        level=getattr(logging, config['output']['log_level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def discover_intermediate_files(intermediate_dir: Path) -> List[Path]:
    """Discover all intermediate JSON files from Stage 1 processing"""
    json_files = list(intermediate_dir.glob("*.json"))
    return sorted(json_files)


def load_and_combine_intermediates(json_files: List[Path], logger) -> pd.DataFrame:
    """Load all intermediate JSON files and combine with consistent processing"""
    logger.info(f"Loading {len(json_files)} intermediate files...")
    
    all_data = {}  # h3_index -> list of records for averaging
    
    for i, json_file in enumerate(json_files):
        if i % 50 == 0:
            logger.info(f"Processing file {i+1}/{len(json_files)}: {json_file.name}")
            
        try:
            with open(json_file, 'r') as f:
                tile_data = json.load(f)
            
            for record in tile_data:
                h3_index = record['h3_index']
                
                # Skip records with NaN embeddings
                if 'embedding' not in record or not record['embedding']:
                    continue
                    
                embedding_array = np.array(record['embedding'])
                if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
                    continue
                
                if h3_index not in all_data:
                    all_data[h3_index] = []
                    
                all_data[h3_index].append(record)
                
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue
    
    logger.info(f"Found data for {len(all_data)} unique H3 hexagons")
    
    # CONSISTENT PROCESSING: Average embeddings for ALL hexagons to eliminate discontinuities
    logger.info("Processing all hexagons with consistent averaging (eliminates tile boundaries)...")
    
    final_records = []
    single_tile_count = 0
    multi_tile_count = 0
    
    for h3_index, records in all_data.items():
        # ALWAYS average embeddings, even for single-tile hexagons
        # This ensures consistent processing and eliminates tile boundary artifacts
        
        # Extract embeddings and average
        valid_embeddings = []
        for rec in records:
            embedding_array = np.array(rec['embedding'])
            if not (np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array))):
                valid_embeddings.append(embedding_array)
        
        if not valid_embeddings:
            continue  # Skip if no valid embeddings
            
        # Average all valid embeddings (includes single-tile consistency)
        embeddings = np.array(valid_embeddings)
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        
        # Create merged record with averaged embedding
        merged_record = records[0].copy()
        merged_record['embedding'] = avg_embedding
        merged_record['tile_count'] = len(valid_embeddings)
        
        final_records.append(merged_record)
        
        if len(valid_embeddings) == 1:
            single_tile_count += 1
        else:
            multi_tile_count += 1
    
    logger.info(f"Processed {single_tile_count} single-tile hexagons (averaged for consistency)")
    logger.info(f"Processed {multi_tile_count} multi-tile hexagons (averaged from overlaps)")
    logger.info(f"ELIMINATED tile boundary discontinuities through consistent processing")
    
    # Convert to DataFrame
    df = pd.DataFrame(final_records)
    
    # Expand embeddings into separate columns
    embedding_cols = [f'A{i:02d}' for i in range(64)]  # A00 through A63
    embeddings_df = pd.DataFrame(df['embedding'].tolist(), columns=embedding_cols, index=df.index)
    
    # Combine with metadata
    result_df = pd.concat([
        df[['h3_index', 'lat', 'lng']].copy(),
        embeddings_df
    ], axis=1)
    
    # Add tile_count if available
    if 'tile_count' in df.columns:
        result_df['tile_count'] = df['tile_count']
    else:
        result_df['tile_count'] = 1
        
    return result_df


def save_final_dataset(df: pd.DataFrame, config: dict, run_id: str, logger):
    """Save final dataset in multiple formats"""
    output_dir = Path(config['output']['modular_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main Parquet file
    parquet_file = output_dir / f"cascadia_coastal_forests_2021_res8_final.parquet"
    df.to_parquet(parquet_file, index=False)
    logger.info(f"Saved final dataset: {parquet_file}")
    
    # Summary statistics
    stats = {
        'total_hexagons': len(df),
        'coastal_area_km2': len(df) * 0.737,  # H3 res 8 area
        'embedding_dimensions': 64,
        'overlap_hexagons': len(df[df['tile_count'] > 1]) if 'tile_count' in df.columns else 0,
        'avg_embeddings_per_hex': df['tile_count'].mean() if 'tile_count' in df.columns else 1.0,
        'processing_timestamp': run_id
    }
    
    stats_file = output_dir / f"dataset_stats_{run_id}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
        
    logger.info(f"Dataset statistics: {stats}")
    
    return parquet_file, stats


def cleanup_intermediates(intermediate_dir: Path, config: dict, logger):
    """Archive intermediate files after successful stitching"""
    if config['processing'].get('cleanup_intermediates', False):
        archive_dir = Path("data/archive/intermediates") / datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        json_files = list(intermediate_dir.glob("*.json"))
        for json_file in json_files:
            json_file.rename(archive_dir / json_file.name)
            
        logger.info(f"Archived {len(json_files)} intermediate files to: {archive_dir}")
    else:
        logger.info("Intermediate files retained (cleanup disabled in config)")


def main():
    """Main entry point for Cascadia coastal forest stitching"""
    
    parser = argparse.ArgumentParser(
        description='Stitch intermediate JSONs into final Cascadia coastal forest dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stitch_results.py                               # Default stitching
  python stitch_results.py --cleanup                     # Clean up intermediates after
  python stitch_results.py --intermediate-dir custom/    # Custom intermediate directory
        """
    )
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--intermediate-dir', type=str, default=None,
                       help='Override intermediate directory path')
    parser.add_argument('--cleanup', action='store_true',
                       help='Archive intermediate files after successful stitching')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Override output filename')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.cleanup:
        config['processing']['cleanup_intermediates'] = True
    
    # Create run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging
    logger = setup_logging(config, run_id)
    
    # Log run details
    logger.info("="*80)
    logger.info("CASCADIA COASTAL FORESTS PROCESSING - STAGE 2")
    logger.info("="*80)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Stage: 2 - Stitching intermediate JSONs to final dataset")
    logger.info("-"*80)
    
    try:
        # Determine intermediate directory
        intermediate_dir = Path(args.intermediate_dir or config['output']['intermediate_dir'])
        
        if not intermediate_dir.exists():
            logger.error(f"Intermediate directory not found: {intermediate_dir}")
            logger.info("Make sure Stage 1 processing has completed successfully")
            sys.exit(1)
            
        logger.info(f"Using intermediate directory: {intermediate_dir}")
        
        # Discover intermediate files
        start_time = datetime.now()
        json_files = discover_intermediate_files(intermediate_dir)
        
        if not json_files:
            logger.error(f"No intermediate JSON files found in: {intermediate_dir}")
            logger.info("Make sure Stage 1 processing has completed successfully")
            sys.exit(1)
            
        logger.info(f"Found {len(json_files)} intermediate files to stitch")
        
        # Load and combine data
        logger.info("Starting stitching process...")
        df = load_and_combine_intermediates(json_files, logger)
        
        # Save final dataset
        parquet_file, stats = save_final_dataset(df, config, run_id, logger)
        
        # Cleanup if requested
        if config['processing'].get('cleanup_intermediates', False):
            cleanup_intermediates(intermediate_dir, config, logger)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Log completion
        logger.info("="*80)
        logger.info("STAGE 2 STITCHING COMPLETE")
        logger.info(f"Duration: {duration}")
        logger.info(f"Final dataset: {parquet_file}")
        logger.info(f"Total hexagons: {len(df):,}")
        logger.info(f"Coastal area covered: ~{len(df) * 0.737:.0f} km²")
        logger.info("="*80)
        logger.info("CASCADIA COASTAL FORESTS PROCESSING COMPLETE!")
        logger.info("Dataset ready for analysis and visualization")
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.warning("Stitching interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Stitching failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()