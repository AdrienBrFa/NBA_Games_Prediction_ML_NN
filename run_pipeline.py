"""
Main pipeline entrypoint for NBA game prediction.

DEPRECATED: This file is kept for backwards compatibility.
Please use the stage-specific scripts:
  - run_stage_a1.py  (historical features only)
  - run_stage_b1.py  (+ team statistics)

This script will redirect to Stage A1 by default.
"""

import sys
import subprocess

print("="*80)
print("⚠️  DEPRECATION NOTICE")
print("="*80)
print("run_pipeline.py is deprecated. Please use stage-specific scripts:")
print("  • python run_stage_a1.py  (historical features only)")
print("  • python run_stage_b1.py  (+ team statistics)")
print("\nRedirecting to Stage A1 in 3 seconds...")
print("="*80)

import time
time.sleep(3)

# Redirect to Stage A1
sys.exit(subprocess.call([sys.executable, "run_stage_a1.py"]))

