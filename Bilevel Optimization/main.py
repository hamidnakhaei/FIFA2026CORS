"""
Main entry points for FIFA 2026 Bilevel Optimization
Provides convenient functions for common use cases
"""

from solver import FIFA2026Solver
from config import DATA_DIR


def run_optimization(time_limit=3600, mip_gap=0.01, verbose=True):
    """
    Run the complete optimization pipeline.
    
    Args:
        time_limit: Time limit in seconds (default 1 hour)
        mip_gap: MIP optimality gap tolerance (default 1%)
        verbose: Print progress output (default True)
    
    Returns:
        Solution dictionary
    
    Output Files (in 'output/' directory):
        - optimized_schedule.xlsx: New schedule with optimized times/venues
        - base_camp_assignments.xlsx: Base camp assignments for each team
    """
    solver = FIFA2026Solver(data_dir=DATA_DIR)
    solution = solver.run_full_pipeline(time_limit=time_limit, mip_gap=mip_gap)
    
    if solution:
        solver.print_solution_summary()
        solver.save_solution("solution_output.txt")
    
    return solution


def run_quick_test():
    """
    Quick test: load data and build model without solving.
    Useful for debugging and validation.
    """
    print("Running quick test (data load + model build only)...\n")
    
    solver = FIFA2026Solver(data_dir=DATA_DIR)
    solver.load_data()
    solver.build_parameters()
    solver.build_model()
    
    print("\n✓ Quick test complete. Model ready for solving.")
    
    return solver


def run_with_custom_weights(weights_dict, time_limit=3600, mip_gap=0.01):
    """
    Run optimization with custom KPI weights.
    
    Args:
        weights_dict: Dictionary mapping KPI IDs to weights
                     Example: {1.2: 2.0, 1.3: 1.0, ...}
        time_limit: Solver time limit
        mip_gap: MIP gap tolerance
    
    Returns:
        Solution dictionary
    """
    from config import KPI_WEIGHTS
    
    # Update weights
    for kpi_id, weight in weights_dict.items():
        if kpi_id in KPI_WEIGHTS:
            KPI_WEIGHTS[kpi_id] = weight
    
    print(f"Running optimization with custom weights: {weights_dict}\n")
    
    return run_optimization(time_limit=time_limit, mip_gap=mip_gap)


def compare_scenarios(scenario_configs):
    """
    Run multiple optimization scenarios and compare results.
    
    Args:
        scenario_configs: List of dicts, each containing:
                         {'name': str, 'weights': dict, 'time_limit': int, 'mip_gap': float}
    
    Returns:
        Dictionary of results for each scenario
    """
    results = {}
    
    for config in scenario_configs:
        scenario_name = config.get('name', 'Scenario')
        weights = config.get('weights', {})
        time_limit = config.get('time_limit', 3600)
        mip_gap = config.get('mip_gap', 0.01)
        
        print(f"\n{'='*70}")
        print(f"Running {scenario_name}...")
        print(f"{'='*70}\n")
        
        solution = run_with_custom_weights(weights, time_limit, mip_gap)
        results[scenario_name] = solution
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SCENARIO COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    for scenario_name, solution in results.items():
        if solution:
            print(f"{scenario_name}:")
            print(f"  Objective: {solution['objective']:.2f}")
            print(f"  Teams assigned: {solution['kpis']['num_teams_assigned']}")
            print(f"  Solver time: {solution['solver_time']:.1f}s")
        else:
            print(f"{scenario_name}: NO SOLUTION FOUND")
        print()
    
    return results


# Example scenarios for comparison
EXAMPLE_SCENARIOS = [
    {
        'name': 'Travel-Focused',
        'weights': {1.2: 2.0, 1.3: 0.5, 1.4: 0.5},
        'time_limit': 3600,
        'mip_gap': 0.01
    },
    {
        'name': 'Balanced',
        'weights': {1.2: 1.0, 1.3: 1.0, 1.4: 1.0},
        'time_limit': 3600,
        'mip_gap': 0.01
    },
    {
        'name': 'Fairness-Focused',
        'weights': {1.2: 2.0, 4.1: 2.0, 5.3: 2.0},
        'time_limit': 3600,
        'mip_gap': 0.01
    }
]


if __name__ == "__main__":
    import sys
    
    # Simple CLI interface
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "optimize":
            run_optimization()
        elif command == "test":
            run_quick_test()
        elif command == "compare":
            compare_scenarios(EXAMPLE_SCENARIOS)
        else:
            print("Usage: python main.py [command]")
            print("\nAvailable commands:")
            print("  optimize    Run full optimization (default)")
            print("  test        Quick test (load + build, no solve)")
            print("  compare     Compare multiple scenarios")
    else:
        # Default: run optimization
        run_optimization()
