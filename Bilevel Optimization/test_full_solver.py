#!/usr/bin/env python3
"""Test full solver pipeline with debug output"""
import sys
import traceback

try:
    from main import run_optimization_optsim
    print("[OK] Imports successful\n")
    
    print("Running full optimization pipeline with short time limits for testing...")
    print("(Using 120s per iteration, max 3 iterations for quick feedback)\n")
    
    solution = run_optimization_optsim(
        time_limit=360,  # 360s total = 120s per iteration max
        max_iterations=3,
        verbose=True
    )
    
    if solution:
        print("\n" + "="*70)
        print("[SUCCESS] OPTIMIZATION SUCCESSFUL!")
        print("="*70)
        print(f"\nSolution details:")
        print(f"  - Iterations run: {solution.get('iterations', 'N/A')}")
        print(f"  - Final KPI penalty: {solution.get('kpi_penalty', 'N/A'):.4f}")
        print(f"  - Solve time: {solution.get('solve_time', 'N/A'):.1f}s")
        print(f"\nKPI history:")
        for entry in solution.get('kpi_history', []):
            print(f"    Iter {entry.get('iteration', -1):2d}: penalty={entry.get('penalty', 0):.4f}, "
                  f"accepted={'Yes' if entry.get('accepted') else 'No'}")
    else:
        print("\n[FAIL] No solution returned")
        
except Exception as e:
    print(f"\n[ERROR] Error during optimization: {e}")
    traceback.print_exc()
    sys.exit(1)
