#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the modular TIFF to H3 processor with proper configuration and monitoring
"""

import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Add scripts to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from modular_tiff_processor import ModularTiffProcessor


def setup_logging(config):
    """Set up logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"modular_run_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config['output']['log_level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_directory_structure(config):
    """Create all necessary directories"""
    directories = [
        Path(config['output']['modular_dir']),
        Path(config['output']['intermediate_dir']),
        Path(config['output']['archive_dir']),
        Path(config['processing']['checkpoint_dir']),
        Path("logs"),
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    return directories


def main():
    """Main entry point for modular processor"""
    
    parser = argparse.ArgumentParser(description='Run modular TIFF to H3 processor')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--max-tiles', type=int, default=None,
                       help='Maximum number of tiles to process (for testing)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--clean-start', action='store_true',
                       help='Start fresh, ignoring checkpoints')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.max_tiles:
        config['experiment']['max_tiles'] = args.max_tiles
    
    if args.clean_start:
        config['processing']['resume_from_checkpoint'] = False
        
    # Set up logging
    logger = setup_logging(config)
    
    # Log run information
    logger.info("="*60)
    logger.info("MODULAR TIFF TO H3 PROCESSOR")
    logger.info("="*60)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Year: {config['experiment']['year']}")
    logger.info(f"H3 Resolution: {config['experiment']['h3_resolution']}")
    logger.info(f"Processing mode: {config['experiment'].get('processing_mode', 'modular')}")
    logger.info(f"Max tiles: {config['experiment']['max_tiles'] or 'ALL'}")
    logger.info(f"Resume from checkpoint: {config['processing']['resume_from_checkpoint']}")
    logger.info("-"*60)
    
    # Create directory structure
    directories = create_directory_structure(config)
    logger.info(f"Created directory structure: {', '.join(str(d) for d in directories)}")
    
    try:
        # Initialize processor
        logger.info("Initializing modular processor...")
        processor = ModularTiffProcessor(config)
        
        # Run processing
        logger.info("Starting optimized processing with spatial filtering and parallel workers...")
        start_time = datetime.now()
        
        result_df = processor.run(n_workers=args.workers)
        
        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("="*60)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Duration: {duration}")
        logger.info(f"Total hexagons: {len(result_df)}")
        logger.info(f"Output saved to: {config['output']['modular_dir']}")
        logger.info("="*60)
        
        # Archive this run
        archive_dir = Path(config['output']['archive_dir']) / f"run_{start_time.strftime('%Y%m%d_%H%M%S')}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Save run metadata
        run_metadata = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_hexagons': len(result_df),
            'config': config,
            'completed': True
        }
        
        import json
        with open(archive_dir / 'run_metadata.json', 'w') as f:
            json.dump(run_metadata, f, indent=2)
            
        logger.info(f"Run archived to: {archive_dir}")
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()