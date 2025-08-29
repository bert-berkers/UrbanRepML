#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cascadia Coastal Forests Processing - Stage 1: TIFF to Intermediate JSONs
Processes AlphaEarth TIFFs to per-tile H3 hexagon JSON files with parallel workers
"""

import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add scripts to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from modular_tiff_processor import ModularTiffProcessor


def setup_logging(config, run_id):
    """Set up logging with timestamped files"""
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"coastal_processing_{run_id}.log"
    
    logging.basicConfig(
        level=getattr(logging, config['output']['log_level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main entry point for Cascadia coastal forest processing"""
    
    parser = argparse.ArgumentParser(
        description='Process Cascadia coastal forest AlphaEarth data to H3 hexagons',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_coastal_processing.py --workers 6
  python run_coastal_processing.py --max-tiles 50 --workers 4  # Test run
  python run_coastal_processing.py --clean-start --workers 8  # Fresh start
        """
    )
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--workers', type=int, default=6,
                       help='Number of parallel workers (default: 6)')
    parser.add_argument('--max-tiles', type=int, default=None,
                       help='Maximum tiles to process (for testing)')
    parser.add_argument('--clean-start', action='store_true',
                       help='Ignore checkpoints, start fresh')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from checkpoint (default behavior)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments  
    if args.max_tiles:
        config['experiment']['max_tiles'] = args.max_tiles
    if args.clean_start:
        config['processing']['resume_from_checkpoint'] = False
    
    # Create run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging
    logger = setup_logging(config, run_id)
    
    # Log run details
    logger.info("="*80)
    logger.info("CASCADIA COASTAL FORESTS PROCESSING - STAGE 1")
    logger.info("="*80)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Max tiles: {config['experiment']['max_tiles'] or 'ALL'}")
    logger.info(f"Resume: {config['processing']['resume_from_checkpoint']}")
    logger.info(f"Output: Stage 1 - Intermediate JSON files per tile")
    logger.info("-"*80)
    
    try:
        # Initialize processor
        logger.info("Initializing coastal forest processor...")
        processor = ModularTiffProcessor(config)
        
        # Run Stage 1 processing  
        logger.info(f"Starting Stage 1: TIFF → Intermediate JSON processing...")
        logger.info(f"Using {args.workers} parallel workers for optimal throughput")
        
        start_time = datetime.now()
        
        # Process to intermediate files (no final stitching yet)
        processor.process_to_intermediate(n_workers=args.workers)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Log completion
        logger.info("="*80) 
        logger.info("STAGE 1 PROCESSING COMPLETE")
        logger.info(f"Duration: {duration}")
        logger.info(f"Next step: Run 'python stitch_results.py' to create final dataset")
        logger.info("="*80)
        
        # Archive this run
        archive_dir = Path("../data/archive") / f"stage1_{run_id}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Save run metadata
        run_metadata = {
            'stage': 1,
            'run_id': run_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(), 
            'duration_seconds': duration.total_seconds(),
            'workers': args.workers,
            'config': config,
            'status': 'completed'
        }
        
        import json
        with open(archive_dir / 'run_metadata.json', 'w') as f:
            json.dump(run_metadata, f, indent=2)
            
        logger.info(f"Run archived to: {archive_dir}")
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        logger.info("Progress is checkpointed - you can resume with --resume")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()