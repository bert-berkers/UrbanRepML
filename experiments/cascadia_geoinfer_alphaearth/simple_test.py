#!/usr/bin/env python3
"""
Simple test script for Cascadia experiment.
"""

import sys
import os
from datetime import datetime

print("="*60)
print("CASCADIA EXPERIMENT SIMPLE TEST")
print("="*60)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print()

# Test basic imports
try:
    import numpy as np
    import pandas as pd
    print("OK - numpy and pandas available")
except ImportError as e:
    print(f"ERROR - numpy/pandas: {e}")

try:
    import json
    import yaml
    print("OK - JSON and YAML support available")
except ImportError as e:
    print(f"ERROR - JSON/YAML: {e}")

# Check if config file exists
if os.path.exists('config.yaml'):
    print("OK - config.yaml found")
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"OK - Config loaded with {len(config)} sections")
    except Exception as e:
        print(f"ERROR - Config loading: {e}")
else:
    print("WARNING - config.yaml not found")

# Check directory structure
dirs = ['scripts', 'data', 'logs', 'analysis']
for d in dirs:
    if os.path.exists(d):
        print(f"OK - {d} directory exists")
        if d == 'scripts':
            subdirs = os.listdir(d)
            print(f"  Subdirs: {subdirs}")
    else:
        print(f"WARNING - {d} directory missing")
        os.makedirs(d, exist_ok=True)
        print(f"  Created: {d}")

print()
print("Basic test complete!")
print("="*60)