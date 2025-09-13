#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test of the modular processor with just 2 tiles
"""

import subprocess
import sys

print("Testing modular processor with 2 tiles...")
print("-" * 60)

# Run with just 2 tiles to test
result = subprocess.run([
    sys.executable,
    "run_modular_processor.py",
    "--max-tiles", "2",
    "--clean-start"
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print("-" * 60)
print(f"Test {'PASSED' if result.returncode == 0 else 'FAILED'}")
print(f"Return code: {result.returncode}")