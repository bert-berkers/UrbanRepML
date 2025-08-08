#!/usr/bin/env python3
"""
Simple utility to display generated plots
"""
import os
import sys
from pathlib import Path

def show_plots(experiment_name: str):
    """List and optionally display plots for an experiment."""
    plots_dir = Path("results/plots") / experiment_name
    
    if not plots_dir.exists():
        print(f"No plots directory found for {experiment_name}")
        return
    
    png_files = list(plots_dir.glob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in {plots_dir}")
        return
    
    print(f"Generated plots for {experiment_name}:")
    print("=" * 50)
    
    for i, plot_file in enumerate(sorted(png_files), 1):
        file_size = plot_file.stat().st_size / (1024 * 1024)  # MB
        print(f"{i}. {plot_file.name}")
        print(f"   Size: {file_size:.1f} MB")
        print(f"   Path: {plot_file}")
        print()
    
    # Also show statistics if available
    stats_file = plots_dir / f"{experiment_name}_visualization_stats.json"
    if stats_file.exists():
        print("Visualization Statistics:")
        print("-" * 30)
        import json
        with open(stats_file) as f:
            stats = json.load(f)
            print(json.dumps(stats, indent=2))
    
    # Show file system command to open plots folder
    abs_path = plots_dir.resolve()
    print(f"\nTo view plots, open: {abs_path}")
    
    # Platform-specific command to open folder
    import platform
    if platform.system() == "Windows":
        print(f"Command: explorer \"{abs_path}\"")
    elif platform.system() == "Darwin":  # macOS
        print(f"Command: open \"{abs_path}\"")
    else:  # Linux
        print(f"Command: xdg-open \"{abs_path}\"")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_plots.py <experiment_name>")
        print("Example: python show_plots.py south_holland_fsi99")
        sys.exit(1)
    
    show_plots(sys.argv[1])