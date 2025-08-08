#!/usr/bin/env python3
"""
Cascadia AlphaEarth Multi-Resolution Experiment Orchestrator.

This script orchestrates the complete Cascadia experiment pipeline from 
Google Earth Engine export through GEO-INFER integration.

Usage:
    python run_cascadia_experiment.py --check_availability
    python run_cascadia_experiment.py --export_gee --years 2023 2024
    python run_cascadia_experiment.py --process_h3 --all_resolutions
    python run_cascadia_experiment.py --full_pipeline --years 2023
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
from typing import List, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(f'logs/orchestrator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CascadiaExperimentOrchestrator:
    """Orchestrate the complete Cascadia experiment pipeline."""
    
    def __init__(self, base_dir: str = "."):
        """
        Initialize orchestrator.
        
        Args:
            base_dir: Base directory of the experiment
        """
        self.base_dir = os.path.abspath(base_dir)
        self.scripts_dir = os.path.join(base_dir, "scripts")
        self.logs_dir = os.path.join(base_dir, "logs")
        
        # Ensure directories exist
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Pipeline steps
        self.pipeline_status = {
            'availability_check': False,
            'gee_export': False,
            'h3_processing': False,
            'gap_detection': False,
            'synthetic_generation': False,
            'geoinfer_preparation': False
        }
        
        logger.info("Initialized Cascadia Experiment Orchestrator")
        logger.info(f"Base directory: {self.base_dir}")
    
    def run_script(self, script_path: str, args: List[str] = None, 
                   cwd: str = None) -> bool:
        """
        Run a Python script with arguments.
        
        Args:
            script_path: Path to Python script
            args: Command line arguments
            cwd: Working directory
            
        Returns:
            True if successful
        """
        if args is None:
            args = []
        
        if cwd is None:
            cwd = self.base_dir
        
        cmd = [sys.executable, script_path] + args
        
        logger.info(f"Running: {' '.join(cmd)}")
        logger.info(f"Working directory: {cwd}")
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=cwd, 
                capture_output=True, 
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("Script completed successfully")
                if result.stdout:
                    logger.info("STDOUT:")
                    for line in result.stdout.strip().split('\n'):
                        logger.info(f"  {line}")
                return True
            else:
                logger.error(f"Script failed with return code {result.returncode}")
                if result.stderr:
                    logger.error("STDERR:")
                    for line in result.stderr.strip().split('\n'):
                        logger.error(f"  {line}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Script timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"Error running script: {e}")
            return False
    
    def check_availability(self, years: List[int] = None) -> bool:
        """
        Check AlphaEarth data availability.
        
        Args:
            years: Years to check
            
        Returns:
            True if successful
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Checking AlphaEarth Data Availability")
        logger.info("="*60)
        
        script_path = os.path.join(
            self.scripts_dir, "gee", "check_years_availability.py"
        )
        
        args = ["--save_report"]
        if years:
            args.extend(["--years"] + [str(y) for y in years])
        
        success = self.run_script(script_path, args)
        self.pipeline_status['availability_check'] = success
        
        return success
    
    def export_from_gee(self, years: List[int] = None, dry_run: bool = False) -> bool:
        """
        Export data from Google Earth Engine.
        
        Args:
            years: Years to export
            dry_run: Test run without actual export
            
        Returns:
            True if successful
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Exporting from Google Earth Engine")
        logger.info("="*60)
        
        script_path = os.path.join(
            self.scripts_dir, "gee", "export_cascadia_alphaearth.py"
        )
        
        args = []
        if years:
            if len(years) == 1:
                args.extend(["--year", str(years[0])])
            else:
                args.extend(["--years"] + [str(y) for y in years])
        else:
            args.append("--all_years")
        
        if dry_run:
            args.append("--dry_run")
        
        success = self.run_script(script_path, args)
        self.pipeline_status['gee_export'] = success
        
        if success and not dry_run:
            logger.info("\n" + "⚠️  IMPORTANT NEXT STEP:")
            logger.info("Monitor export progress at: https://code.earthengine.google.com/tasks")
            logger.info("Once exports complete, sync Google Drive to local storage")
            logger.info("Then proceed with H3 processing")
        
        return success
    
    def process_to_h3(self, years: List[int] = None, 
                     resolutions: List[int] = None) -> bool:
        """
        Process AlphaEarth tiles to H3 hexagons.
        
        Args:
            years: Years to process
            resolutions: H3 resolutions to generate
            
        Returns:
            True if successful
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Processing to H3 Multi-Resolution")
        logger.info("="*60)
        
        script_path = os.path.join(
            self.scripts_dir, "processing", "process_cascadia_multires.py"
        )
        
        args = []
        if years:
            if len(years) == 1:
                args.extend(["--year", str(years[0])])
            else:
                args.extend(["--years"] + [str(y) for y in years])
        
        if resolutions:
            if len(resolutions) == 1:
                args.extend(["--resolution", str(resolutions[0])])
            else:
                args.extend(["--resolutions"] + [str(r) for r in resolutions])
        else:
            args.append("--all_resolutions")
        
        success = self.run_script(script_path, args)
        self.pipeline_status['h3_processing'] = success
        
        return success
    
    def detect_gaps(self, years: List[int] = None, 
                   resolutions: List[int] = None) -> bool:
        """
        Detect gaps in the processed data.
        
        Args:
            years: Years to analyze
            resolutions: Resolutions to analyze
            
        Returns:
            True if successful
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Detecting Data Gaps")
        logger.info("="*60)
        
        script_path = os.path.join(
            self.scripts_dir, "actualization", "gap_detector.py"
        )
        
        args = ["--save_report"]
        if years:
            if len(years) == 1:
                args.extend(["--year", str(years[0])])
            else:
                args.extend(["--years"] + [str(y) for y in years])
        else:
            args.append("--all_years")
        
        if resolutions:
            if len(resolutions) == 1:
                args.extend(["--resolution", str(resolutions[0])])
            else:
                args.extend(["--resolutions"] + [str(r) for r in resolutions])
        else:
            args.append("--all_resolutions")
        
        success = self.run_script(script_path, args)
        self.pipeline_status['gap_detection'] = success
        
        return success
    
    def generate_synthetic(self, years: List[int] = None, 
                          resolutions: List[int] = None,
                          method: str = 'vae') -> bool:
        """
        Generate synthetic data for gaps.
        
        Args:
            years: Years to generate for
            resolutions: Resolutions to generate
            method: Generation method
            
        Returns:
            True if successful
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 5: Generating Synthetic Data")
        logger.info("="*60)
        
        script_path = os.path.join(
            self.scripts_dir, "actualization", "synthetic_generator.py"
        )
        
        args = ["--method", method, "--validate"]
        if years:
            if len(years) == 1:
                args.extend(["--year", str(years[0])])
            else:
                args.extend(["--years"] + [str(y) for y in years])
        
        if resolutions:
            if len(resolutions) == 1:
                args.extend(["--resolution", str(resolutions[0])])
            else:
                args.extend(["--resolutions"] + [str(r) for r in resolutions])
        
        success = self.run_script(script_path, args)
        self.pipeline_status['synthetic_generation'] = success
        
        return success
    
    def prepare_for_geoinfer(self, years: List[int] = None, 
                           include_synthetic: bool = False) -> bool:
        """
        Prepare data for GEO-INFER integration.
        
        Args:
            years: Years to prepare
            include_synthetic: Include synthetic data
            
        Returns:
            True if successful
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 6: Preparing for GEO-INFER Integration")
        logger.info("="*60)
        
        script_path = os.path.join(
            self.scripts_dir, "geoinfer", "prepare_for_geoinfer.py"
        )
        
        args = []
        if years:
            if len(years) == 1:
                args.extend(["--year", str(years[0])])
            else:
                args.extend(["--years"] + [str(y) for y in years])
        else:
            args.append("--all_years")
        
        if include_synthetic:
            args.append("--include_synthetic")
        
        success = self.run_script(script_path, args)
        self.pipeline_status['geoinfer_preparation'] = success
        
        return success
    
    def run_full_pipeline(self, years: List[int] = None, 
                         resolutions: List[int] = None,
                         include_synthetic: bool = True,
                         skip_gee: bool = False) -> bool:
        """
        Run the complete pipeline.
        
        Args:
            years: Years to process
            resolutions: H3 resolutions
            include_synthetic: Include synthetic data generation
            skip_gee: Skip Google Earth Engine export (data already local)
            
        Returns:
            True if all steps successful
        """
        logger.info("\n" + "#"*60)
        logger.info("RUNNING COMPLETE CASCADIA EXPERIMENT PIPELINE")
        logger.info(f"Years: {years if years else 'All (2017-2024)'}")
        logger.info(f"Resolutions: {resolutions if resolutions else 'All (5-11)'}")
        logger.info(f"Include synthetic: {include_synthetic}")
        logger.info(f"Skip GEE export: {skip_gee}")
        logger.info("#"*60)
        
        success = True
        
        # Step 1: Check availability
        if success:
            success = self.check_availability(years)
        
        # Step 2: GEE Export (optional)
        if success and not skip_gee:
            logger.info("\n⚠️  Starting GEE export - this will initiate tasks")
            logger.info("You will need to monitor progress and sync data manually")
            response = input("Continue with GEE export? [y/N]: ").lower().strip()
            if response == 'y':
                success = self.export_from_gee(years)
                if success:
                    logger.info("\n⏸️  PIPELINE PAUSED")
                    logger.info("Please complete the following steps:")
                    logger.info("1. Monitor exports at: https://code.earthengine.google.com/tasks")
                    logger.info("2. Wait for all exports to complete")
                    logger.info("3. Sync Google Drive to local storage")
                    logger.info("4. Run pipeline again with --skip_gee flag")
                    return True
            else:
                logger.info("Skipping GEE export")
        
        # Step 3: H3 Processing
        if success:
            success = self.process_to_h3(years, resolutions)
        
        # Step 4: Gap Detection
        if success:
            success = self.detect_gaps(years, resolutions)
        
        # Step 5: Synthetic Generation (optional)
        if success and include_synthetic:
            success = self.generate_synthetic(years, resolutions)
        
        # Step 6: GEO-INFER Preparation
        if success:
            success = self.prepare_for_geoinfer(years, include_synthetic)
        
        # Final report
        self.print_pipeline_summary()
        
        return success
    
    def print_pipeline_summary(self):
        """Print pipeline execution summary."""
        logger.info("\n" + "#"*60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("#"*60)
        
        for step, status in self.pipeline_status.items():
            status_str = "✅ COMPLETED" if status else "❌ FAILED/SKIPPED"
            logger.info(f"{step.replace('_', ' ').title():<25} {status_str}")
        
        successful_steps = sum(self.pipeline_status.values())
        total_steps = len(self.pipeline_status)
        
        logger.info(f"\nOverall Success: {successful_steps}/{total_steps} steps")
        
        if successful_steps == total_steps:
            logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("Ready for GEO-INFER integration and analysis")
        else:
            logger.info(f"\n⚠️  Pipeline partially completed ({successful_steps}/{total_steps})")
            logger.info("Check logs for error details")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Orchestrate Cascadia AlphaEarth experiment pipeline"
    )
    
    # Pipeline steps
    parser.add_argument('--check_availability', action='store_true',
                       help='Check AlphaEarth data availability')
    parser.add_argument('--export_gee', action='store_true',
                       help='Export data from Google Earth Engine')
    parser.add_argument('--process_h3', action='store_true',
                       help='Process tiles to H3 hexagons')
    parser.add_argument('--detect_gaps', action='store_true',
                       help='Detect gaps in data')
    parser.add_argument('--generate_synthetic', action='store_true',
                       help='Generate synthetic data')
    parser.add_argument('--prepare_geoinfer', action='store_true',
                       help='Prepare for GEO-INFER integration')
    parser.add_argument('--full_pipeline', action='store_true',
                       help='Run complete pipeline')
    
    # Parameters
    parser.add_argument('--years', nargs='+', type=int,
                       help='Years to process')
    parser.add_argument('--resolutions', nargs='+', type=int,
                       help='H3 resolutions to generate')
    parser.add_argument('--all_resolutions', action='store_true',
                       help='Use all resolutions (5-11)')
    parser.add_argument('--method', choices=['vae', 'gan', 'interpolation'],
                       default='vae', help='Synthetic generation method')
    parser.add_argument('--include_synthetic', action='store_true',
                       help='Include synthetic data in outputs')
    parser.add_argument('--skip_gee', action='store_true',
                       help='Skip Google Earth Engine export')
    parser.add_argument('--dry_run', action='store_true',
                       help='Test run without actual execution')
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = CascadiaExperimentOrchestrator()
    
    # Determine resolutions
    resolutions = None
    if args.all_resolutions:
        resolutions = [5, 6, 7, 8, 9, 10, 11]
    elif args.resolutions:
        resolutions = args.resolutions
    
    # Execute requested steps
    if args.full_pipeline:
        orchestrator.run_full_pipeline(
            years=args.years,
            resolutions=resolutions,
            include_synthetic=args.include_synthetic,
            skip_gee=args.skip_gee
        )
    else:
        # Individual steps
        if args.check_availability:
            orchestrator.check_availability(args.years)
        
        if args.export_gee:
            orchestrator.export_from_gee(args.years, args.dry_run)
        
        if args.process_h3:
            orchestrator.process_to_h3(args.years, resolutions)
        
        if args.detect_gaps:
            orchestrator.detect_gaps(args.years, resolutions)
        
        if args.generate_synthetic:
            orchestrator.generate_synthetic(args.years, resolutions, args.method)
        
        if args.prepare_geoinfer:
            orchestrator.prepare_for_geoinfer(args.years, args.include_synthetic)


if __name__ == "__main__":
    main()