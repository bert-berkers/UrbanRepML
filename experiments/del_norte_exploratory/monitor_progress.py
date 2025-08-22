#!/usr/bin/env python
"""Monitor AlphaEarth processing progress."""

import time
from pathlib import Path
import json

# Check for output files
output_dir = Path("data/h3_2021_res10")
output_file = output_dir / "del_norte_2021_res10.parquet"
metadata_file = output_dir / "del_norte_2021_res10.json"

print("Monitoring AlphaEarth processing...")
print("=" * 50)

# Check if files exist
if output_file.exists():
    print(f"[DONE] Output file created: {output_file}")
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"[DONE] Metadata saved")
        print(f"  - H3 Resolution: {metadata['h3_resolution']}")
        print(f"  - Total hexagons: {metadata['n_hexagons']:,}")
        print(f"  - Dimensions: {metadata['n_dimensions']}")
        print(f"  - Source files: {metadata['source_files']}")
        print(f"  - Avg pixels/hex: {metadata['mean_pixels_per_hex']:.1f}")
    else:
        print("[WAIT] Waiting for metadata...")
else:
    print("[PROCESSING] Processing in progress...")
    print("  Expected output: data/h3_2021_res10/del_norte_2021_res10.parquet")
    print("  This will take approximately 25-30 minutes for 50 tiles")

# Check log file
log_file = Path("logs/del_norte_exploratory.log")
if log_file.exists():
    with open(log_file, 'r') as f:
        lines = f.readlines()
        # Get last 5 relevant lines
        relevant = [l for l in lines if 'batch' in l.lower() or 'hexagon' in l.lower() or 'saved' in l.lower()]
        if relevant:
            print("\nRecent log entries:")
            for line in relevant[-5:]:
                print(f"  {line.strip()}")