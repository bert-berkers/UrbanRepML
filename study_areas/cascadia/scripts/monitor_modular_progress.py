#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monitor progress of the modular TIFF processor in real-time
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys


class ModularProgressMonitor:
    """Monitor and report on modular processing progress"""
    
    def __init__(self):
        self.checkpoint_file = Path("data/checkpoints/modular_progress.json")
        self.output_dir = Path("data/h3_2021_res8_modular")
        self.source_dir = Path("G:/My Drive/AlphaEarth_Cascadia")
        self.pattern = "Cascadia_AlphaEarth_2021_*.tif"
        
    def load_checkpoint(self):
        """Load current checkpoint data"""
        if not self.checkpoint_file.exists():
            return None
            
        with open(self.checkpoint_file, 'r') as f:
            return json.load(f)
    
    def count_total_tiles(self):
        """Count total TIFF tiles available"""
        return len(list(self.source_dir.glob(self.pattern)))
    
    def count_output_files(self):
        """Count generated H3 output files"""
        return len(list(self.output_dir.glob("*_h3_res8.json")))
    
    def calculate_progress(self, checkpoint):
        """Calculate detailed progress metrics"""
        if not checkpoint:
            return {
                'tiles_completed': 0,
                'tiles_total': self.count_total_tiles(),
                'percent_complete': 0,
                'subtiles_in_progress': {}
            }
        
        completed_tiles = checkpoint.get('completed_tiles', [])
        completed_subtiles = checkpoint.get('completed_subtiles', {})
        
        # Calculate subtile progress for in-progress tiles
        subtiles_per_tile = 144  # 12x12 grid of 256x256 subtiles
        
        tiles_in_progress = {}
        for tile_name, subtiles in completed_subtiles.items():
            if tile_name not in completed_tiles:
                n_completed = len(subtiles)
                percent = (n_completed / subtiles_per_tile) * 100
                tiles_in_progress[tile_name] = {
                    'completed': n_completed,
                    'total': subtiles_per_tile,
                    'percent': percent
                }
        
        total_tiles = self.count_total_tiles()
        
        # Calculate overall progress including partial tiles
        total_subtiles = total_tiles * subtiles_per_tile
        completed_subtiles_count = len(completed_tiles) * subtiles_per_tile
        
        for tile_name, subtiles in completed_subtiles.items():
            if tile_name not in completed_tiles:
                completed_subtiles_count += len(subtiles)
        
        overall_percent = (completed_subtiles_count / total_subtiles) * 100 if total_subtiles > 0 else 0
        
        return {
            'tiles_completed': len(completed_tiles),
            'tiles_total': total_tiles,
            'tiles_in_progress': tiles_in_progress,
            'overall_percent': overall_percent,
            'checkpoint_time': checkpoint.get('timestamp', 'Unknown')
        }
    
    def estimate_completion(self, progress):
        """Estimate time to completion based on progress"""
        if progress['tiles_completed'] == 0:
            return None
            
        # Simple estimation based on completed tiles
        # Assume 18 minutes per tile average
        minutes_per_tile = 18
        remaining_tiles = progress['tiles_total'] - progress['tiles_completed']
        
        # Account for in-progress tiles
        for tile_name, tile_progress in progress.get('tiles_in_progress', {}).items():
            remaining_tiles -= (tile_progress['percent'] / 100)
        
        remaining_minutes = remaining_tiles * minutes_per_tile
        
        return {
            'remaining_tiles': remaining_tiles,
            'estimated_hours': remaining_minutes / 60,
            'estimated_completion': datetime.now() + timedelta(minutes=remaining_minutes)
        }
    
    def display_progress(self, continuous=False):
        """Display progress information"""
        
        while True:
            # Clear screen for continuous mode
            if continuous:
                print("\033[2J\033[H")  # Clear screen and move cursor to top
            
            checkpoint = self.load_checkpoint()
            progress = self.calculate_progress(checkpoint)
            
            print("="*70)
            print("MODULAR PROCESSOR PROGRESS MONITOR")
            print("="*70)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-"*70)
            
            # Overall progress
            print(f"\n📊 OVERALL PROGRESS")
            print(f"   Tiles completed: {progress['tiles_completed']}/{progress['tiles_total']}")
            print(f"   Overall progress: {progress['overall_percent']:.1f}%")
            
            # Progress bar
            bar_length = 50
            filled = int(bar_length * progress['overall_percent'] / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"   [{bar}] {progress['overall_percent']:.1f}%")
            
            # In-progress tiles
            if progress.get('tiles_in_progress'):
                print(f"\n🔄 TILES IN PROGRESS ({len(progress['tiles_in_progress'])})")
                for tile_name, tile_prog in list(progress['tiles_in_progress'].items())[:5]:
                    tile_short = tile_name.replace('Cascadia_AlphaEarth_2021_', '')
                    print(f"   {tile_short}: {tile_prog['completed']}/{tile_prog['total']} subtiles ({tile_prog['percent']:.1f}%)")
                
                if len(progress['tiles_in_progress']) > 5:
                    print(f"   ... and {len(progress['tiles_in_progress']) - 5} more")
            
            # Time estimation
            estimation = self.estimate_completion(progress)
            if estimation:
                print(f"\n⏱️  TIME ESTIMATION")
                print(f"   Remaining work: ~{estimation['remaining_tiles']:.1f} tiles")
                print(f"   Estimated hours: {estimation['estimated_hours']:.1f} hours")
                print(f"   Expected completion: {estimation['estimated_completion'].strftime('%Y-%m-%d %H:%M')}")
            
            # Output files
            output_count = self.count_output_files()
            print(f"\n📁 OUTPUT FILES")
            print(f"   H3 JSON files created: {output_count}")
            print(f"   Output directory: {self.output_dir}")
            
            # Checkpoint info
            if checkpoint:
                print(f"\n💾 LAST CHECKPOINT")
                print(f"   Time: {progress.get('checkpoint_time', 'Unknown')}")
                print(f"   Checkpoint file: {self.checkpoint_file}")
            
            print("\n" + "="*70)
            
            if not continuous:
                break
            
            # Wait before next update
            time.sleep(30)  # Update every 30 seconds
    
    def show_summary(self):
        """Show a quick summary of progress"""
        checkpoint = self.load_checkpoint()
        progress = self.calculate_progress(checkpoint)
        
        print(f"Del Norte Processing: {progress['tiles_completed']}/{progress['tiles_total']} tiles ({progress['overall_percent']:.1f}%)")
        
        if progress.get('tiles_in_progress'):
            print(f"In progress: {len(progress['tiles_in_progress'])} tiles")
        
        estimation = self.estimate_completion(progress)
        if estimation:
            print(f"ETA: {estimation['estimated_completion'].strftime('%Y-%m-%d %H:%M')} (~{estimation['estimated_hours']:.1f} hours)")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor modular processor progress')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Continuously monitor progress')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='Show brief summary only')
    args = parser.parse_args()
    
    monitor = ModularProgressMonitor()
    
    if args.summary:
        monitor.show_summary()
    else:
        try:
            monitor.display_progress(continuous=args.continuous)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            sys.exit(0)


if __name__ == "__main__":
    main()