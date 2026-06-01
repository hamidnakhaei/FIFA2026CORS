"""
Main entry points for FIFA 2026 Optimization + Simulation Framework

Uses the new Opt+Sim approach as described in Opt+Sim.tex:
1. Optimization phase: Schedule on base-camp-free KPIs + surrogate predictions
2. Simulation phase: Greedy base-camp selection
3. Guard mechanism: Accept only if simulated KPIs improve
"""

from config import DATA_DIR


# =================================================================
# OPTIMIZATION+SIMULATION FUNCTIONS (Primary Framework)
# =================================================================

def run_optimization_optsim(time_limit=3600, max_iterations=10, verbose=True):
    """
    Run the complete Optimization+Simulation pipeline.
    
    This is the PRIMARY framework as described in Opt+Sim.tex
    
    Algorithm:
    1. Optimize schedule on base-camp-free KPIs + surrogate predictions
    2. Simulate greedy base-camp selection process
    3. Accept only if simulated KPIs improve (sim-in-the-loop guard)
    4. Update surrogate model and repeat
    
    Args:
        time_limit: Total time limit in seconds (default 1 hour)
        max_iterations: Maximum number of optimization iterations (default 10)
        verbose: Print progress output (default True)
    
    Returns:
        Solution dictionary with:
        - schedule: Optimized match schedule
        - camp_assignment: Base camp assignments (from simulation)
        - kpi_penalty: Final KPI penalty value
        - iterations: Number of iterations run
        - kpi_history: Penalty trajectory across iterations
    
    Output Files:
        - optimized_schedule_optsim.xlsx
        - base_camp_assignments_optsim.xlsx
    """
    from solver_optsim import OptSimSolver
    
    solver = OptSimSolver(data_dir=DATA_DIR)
    solution = solver.run_full_pipeline(
        time_limit=time_limit,
        max_iterations=max_iterations
    )
    
    if solution:
        solver.print_solution_summary()
        solver.save_solution("solution_optsim.txt")
    
    return solution


def run_quick_test_optsim():
    """
    Quick test: load data and build Opt+Sim model without solving.
    Useful for debugging and validation.
    """
    print("Running quick test (Opt+Sim data load + model build only)...\n")
    
    from solver_optsim import OptSimSolver
    
    solver = OptSimSolver(data_dir=DATA_DIR)
    solver.load_data()
    solver.build_parameters()
    solver.build_model()
    solver.initialize_simulation()
    
    print("\n[OK] Quick test complete. Opt+Sim ready for optimization loop.")
    
    return solver


# =================================================================
# DEFAULT ENTRY POINT
# =================================================================

def run_optimization(time_limit=3600, max_iterations=10, **kwargs):
    """
    Run optimization with Opt+Sim framework (main entry point).
    
    Args:
        time_limit: Time limit in seconds
        max_iterations: Maximum iterations for Opt+Sim loop
        **kwargs: Additional framework-specific arguments
    
    Returns:
        Solution dictionary
    """
    return run_optimization_optsim(time_limit=time_limit, max_iterations=max_iterations, **kwargs)



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
