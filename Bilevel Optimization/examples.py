"""
Examples and usage patterns for FIFA 2026 Bilevel Optimization
Demonstrates different ways to use the modular framework
"""

from data_loader import load_data
from parameter_builder import build_parameters
from model_builder import build_model
from solver import FIFA2026Solver
from config import DATA_DIR


# ============================================================================
# EXAMPLE 1: Simplest usage - run full pipeline
# ============================================================================
def example_1_basic_optimization():
    """Most basic usage: load, build, solve, extract."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Full Pipeline")
    print("="*70)
    
    solver = FIFA2026Solver()
    solution = solver.run_full_pipeline(time_limit=3600, mip_gap=0.01)
    
    if solution:
        print(f"\nObjective value: {solution['objective']:.2f}")
        print(f"Teams assigned: {solution['kpis']['num_teams_assigned']}")


# ============================================================================
# EXAMPLE 2: Step-by-step control
# ============================================================================
def example_2_step_by_step():
    """Step-by-step control for debugging and inspection."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Step-by-Step Execution")
    print("="*70)
    
    # Step 1: Load data
    print("\n[1] Loading data...")
    data_loader = load_data(DATA_DIR)
    
    # Inspect data
    print(f"  Loaded {len(data_loader.matches)} matches")
    print(f"  Loaded {len(data_loader.teams)} teams")
    print(f"  Loaded {len(data_loader.base_camps)} base camps")
    
    # Step 2: Build parameters
    print("\n[2] Building parameters...")
    params = build_parameters(data_loader)
    print(f"  Built distance matrix")
    print(f"  Built jet-lag penalties")
    print(f"  Big-M value: {params['M_exclusivity']:.0f}")
    
    # Step 3: Build model
    print("\n[3] Building model...")
    model = build_model(data_loader, params)
    print(f"  Model has {model.NumVars} variables")
    print(f"  Model has {model.NumConstrs} constraints")
    
    # Step 4: Solve
    print("\n[4] Solving (first 60 seconds)...")
    model.Params.TimeLimit = 60
    model.optimize()
    
    if model.SolCount > 0:
        print(f"  Found solution with objective: {model.ObjVal:.2f}")
    else:
        print(f"  Time limit reached, no feasible solution yet")


# ============================================================================
# EXAMPLE 3: Data exploration
# ============================================================================
def example_3_data_exploration():
    """Explore and analyze the input data."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Data Exploration")
    print("="*70)
    
    data = load_data(DATA_DIR)
    
    # Get a specific team
    team_id = "BRA"
    print(f"\nTeam: {team_id}")
    
    matches = data.get_team_matches(team_id)
    print(f"  Matches: {len(matches)}")
    for i, match in enumerate(matches, 1):
        print(f"    Match {i}: {match['team_a_id']} vs {match['team_b_id']} "
              f"at {match['venue_id']} ({match['date']})")
    
    eligible_camps = data.get_team_eligible_camps(team_id)
    print(f"  Eligible camps: {len(eligible_camps)}")
    print(f"    IDs: {eligible_camps[:5]}...")  # Show first 5
    
    # Get group information
    group = data.get_team_info(team_id)['group']
    teams_in_group = data.get_group_teams(group)
    print(f"\n  Group {group}: {teams_in_group}")
    
    # Check base camp distribution
    print("\nBase Camp Geographic Distribution:")
    us_camps = len(data.get_camps_in_country("USA"))
    mex_camps = len(data.get_camps_in_country("MEX"))
    can_camps = len(data.get_camps_in_country("CAN"))
    print(f"  USA: {us_camps} facilities")
    print(f"  Mexico: {mex_camps} facilities")
    print(f"  Canada: {can_camps} facilities")


# ============================================================================
# EXAMPLE 4: Modify configuration
# ============================================================================
def example_4_custom_configuration():
    """Run optimization with custom settings."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Configuration")
    print("="*70)
    
    # Modify config values before loading
    from config import KPI_WEIGHTS, GUROBI_TIME_LIMIT, GUROBI_MIP_GAP
    
    # Adjust weights (travel-focused scenario)
    print("\nConfiguration:")
    print("  KPI 1.2 (travel dispersion): Weight = 2.0 (high)")
    print("  KPI 1.7 (US restrictions): Weight = 0.5 (low)")
    print("  Solver time limit: 1800 seconds (30 min)")
    print("  MIP gap: 0.5% (less strict)")
    
    solver = FIFA2026Solver()
    solver.load_data()
    solver.build_parameters()
    solver.build_model()
    
    # Custom solve parameters
    solver.solve(time_limit=1800, mip_gap=0.005)
    solution = solver.extract_solution()
    
    if solution:
        print(f"\nResult: Objective = {solution['objective']:.2f}")


# ============================================================================
# EXAMPLE 5: Access solution details
# ============================================================================
def example_5_solution_details():
    """Access and inspect detailed solution information."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Solution Details")
    print("="*70)
    
    solver = FIFA2026Solver()
    solver.load_data()
    solver.build_parameters()
    solver.build_model()
    solver.model.Params.TimeLimit = 60  # Quick test
    solver.solve()
    
    solution = solver.extract_solution()
    
    if solution:
        print("\nSchedule (sample):")
        for i, (match_id, hour, venue_id) in enumerate(list(solution['schedule'].keys())[:5]):
            print(f"  Match {match_id}: Slot {hour}, Stadium {venue_id}")
        
        print(f"\nBase Camp Assignments (sample):")
        for i, (team_id, camp_id) in enumerate(list(solution['base_camps'].items())[:5]):
            print(f"  {team_id}: Camp {camp_id}")
        
        print(f"\nKPIs:")
        for kpi_name, kpi_value in solution['kpis'].items():
            print(f"  {kpi_name}: {kpi_value}")


# ============================================================================
# EXAMPLE 6: Compare different solver configurations
# ============================================================================
def example_6_solver_comparison():
    """Run solver with different time limits and compare."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Solver Configuration Comparison")
    print("="*70)
    
    configs = [
        {"name": "Quick (30s)", "time_limit": 30, "mip_gap": 0.05},
        {"name": "Standard (60s)", "time_limit": 60, "mip_gap": 0.02},
        {"name": "Extended (120s)", "time_limit": 120, "mip_gap": 0.01},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nRunning {config['name']}...")
        
        solver = FIFA2026Solver()
        solver.load_data()
        solver.build_parameters()
        solver.build_model()
        solver.solve(time_limit=config['time_limit'], mip_gap=config['mip_gap'])
        solution = solver.extract_solution()
        
        results[config['name']] = solution
    
    print("\n" + "-"*70)
    print("COMPARISON RESULTS")
    print("-"*70)
    
    for config_name, solution in results.items():
        if solution:
            print(f"\n{config_name}:")
            print(f"  Objective: {solution['objective']:.2f}")
            print(f"  Teams assigned: {solution['kpis']['num_teams_assigned']}")
            print(f"  Solver time: {solution['solver_time']:.1f}s")
        else:
            print(f"\n{config_name}: NO SOLUTION")


# ============================================================================
# EXAMPLE 7: Using different model builders
# ============================================================================
def example_7_model_comparison():
    """Compare basic vs. advanced model builders."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Model Builder Comparison")
    print("="*70)
    
    from model_builder import ModelBuilder
    from model_builder_advanced import AdvancedModelBuilder
    
    data = load_data(DATA_DIR)
    params = build_parameters(data)
    
    # Basic model
    print("\nBasic Model:")
    basic_builder = ModelBuilder(data, params)
    basic_model = basic_builder.build()
    print(f"  Variables: {basic_model.NumVars}")
    print(f"  Constraints: {basic_model.NumConstrs}")
    
    # Advanced model
    print("\nAdvanced Model:")
    advanced_builder = AdvancedModelBuilder(data, params)
    advanced_model = advanced_builder.build()
    print(f"  Variables: {advanced_model.NumVars}")
    print(f"  Constraints: {advanced_model.NumConstrs}")


# ============================================================================
# EXAMPLE 8: Parameter inspection
# ============================================================================
def example_8_parameter_inspection():
    """Inspect precomputed parameters."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Parameter Inspection")
    print("="*70)
    
    data = load_data(DATA_DIR)
    params = build_parameters(data)
    
    # Sample distances
    print("\nSample Distances (camp → venue):")
    first_camp = list(params['dist'].keys())[0]
    first_camp_distances = params['dist'][first_camp]
    for venue_id in list(first_camp_distances.keys())[:3]:
        dist = first_camp_distances[venue_id]
        print(f"  Camp {first_camp} → Venue {venue_id}: {dist:.1f} km")
    
    # Sample jet-lag penalties
    print("\nSample Jet-Lag Penalties:")
    first_team = list(data.team_by_id.keys())[0]
    first_camp = list(data.eligible_camps_by_team[first_team])[0]
    first_venue = list(data.venue_by_id.keys())[0]
    
    phi_dict = params['Phi'][first_team][first_camp][first_venue]
    for hour in [6, 12, 18]:
        if hour in phi_dict:
            penalty = phi_dict[hour]
            print(f"  {first_team} at camp {first_camp}, "
                  f"match at {first_venue} at hour {hour}: {penalty:.2f} hrs")


# ============================================================================
# Main: Run all examples
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_basic_optimization,
            example_2_step_by_step,
            example_3_data_exploration,
            example_4_custom_configuration,
            example_5_solution_details,
            example_6_solver_comparison,
            example_7_model_comparison,
            example_8_parameter_inspection,
        ]
        
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Example {example_num} not found. Choose 1-{len(examples)}")
    else:
        print("\nAvailable examples:")
        print("  1. Basic optimization")
        print("  2. Step-by-step execution")
        print("  3. Data exploration")
        print("  4. Custom configuration")
        print("  5. Solution details")
        print("  6. Solver comparison")
        print("  7. Model builder comparison")
        print("  8. Parameter inspection")
        print("\nUsage: python examples.py [1-8]")
        
        # Run example 1 by default
        print("\nRunning Example 1 (basic optimization)...\n")
        example_1_basic_optimization()
