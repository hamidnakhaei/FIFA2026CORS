#!/usr/bin/env python3
"""Quick test of solver initialization"""
import sys
import traceback

try:
    from main import run_quick_test_optsim
    print("[OK] Imports successful")
    solver = run_quick_test_optsim()
    print("[OK] Quick test passed")
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()
    sys.exit(1)
