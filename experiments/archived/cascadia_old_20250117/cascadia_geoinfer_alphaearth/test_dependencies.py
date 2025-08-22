#!/usr/bin/env python3
"""
Test script to verify dependencies for Cascadia experiment.
"""

import sys
import os
from datetime import datetime

print("="*60)
print("CASCADIA EXPERIMENT DEPENDENCY CHECK")
print("="*60)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print()

# Test core dependencies
dependencies = [
    ('os', 'Built-in'),
    ('json', 'Built-in'),
    ('datetime', 'Built-in'),
    ('numpy', 'Scientific computing'),
    ('pandas', 'Data manipulation'),
    ('geopandas', 'Geospatial data'),
    ('h3', 'H3 hexagonal indexing'),
    ('shapely', 'Geometric operations'),
    ('rasterio', 'Raster data'),
    ('torch', 'Machine learning'),
    ('sklearn', 'Machine learning utilities'),
    ('matplotlib', 'Plotting'),
    ('seaborn', 'Statistical visualization'),
    ('tqdm', 'Progress bars'),
    ('yaml', 'Configuration parsing')
]

available = []
missing = []

for dep, description in dependencies:
    try:
        __import__(dep)
        available.append((dep, description))
        print(f"✅ {dep:<15} - {description}")
    except ImportError as e:
        missing.append((dep, description, str(e)))
        print(f"❌ {dep:<15} - {description} (MISSING: {e})")

print()
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Available dependencies: {len(available)}/{len(dependencies)}")
print(f"Missing dependencies: {len(missing)}")

if missing:
    print("\nMISSING DEPENDENCIES:")
    for dep, desc, error in missing:
        print(f"  {dep}: {error}")
    print("\nInstall missing dependencies with:")
    print("pip install " + " ".join([dep for dep, _, _ in missing]))
else:
    print("\n🎉 All core dependencies available!")
    print("Ready to run Cascadia experiment scripts")

print()
print("="*60)
print("DIRECTORY STRUCTURE CHECK")
print("="*60)

# Check directory structure
required_dirs = [
    'scripts/gee',
    'scripts/processing', 
    'scripts/actualization',
    'scripts/geoinfer',
    'data',
    'logs',
    'analysis'
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"✅ {dir_path}")
    else:
        print(f"❌ {dir_path} (missing)")
        os.makedirs(dir_path, exist_ok=True)
        print(f"   → Created: {dir_path}")

print("\n🏗️ Directory structure validated!")