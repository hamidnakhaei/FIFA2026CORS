"""
IMPLEMENTATION GUIDE: FIFA 2026 Bilevel Optimization
Modular Python/Gurobi Implementation
"""

================================================================================
OVERVIEW
================================================================================

This project implements the bilevel optimization model for FIFA 2026 World Cup
group-stage scheduling as described in "Bilevel Opt.tex".

KEY FEATURES:
- Modular design with separate files for different purposes
- All teams can choose from all eligible base camps (configuration-based)
- Base camp exclusivity enforced (each facility hosts at most one team)
- Complete hard constraint set (H1-H8)
- Lower-level best-response modeling via McCormick linearization
- Gurobi MILP formulation (no need for KKT or continuous relaxation)


================================================================================
FILES CREATED
================================================================================

CORE MODULES (required for any use):
├─ config.py                 Configuration and constants
├─ utils.py                  Utility functions
├─ data_loader.py            Load and manage CSV data
├─ parameter_builder.py      Precompute parameters
├─ model_builder.py          Build Gurobi MILP model
└─ solver.py                 Main orchestrator

OPTIONAL MODULES:
├─ model_builder_advanced.py Enhanced model builder with more KPIs
├─ main.py                   CLI entry points and helper functions
├─ examples.py               Usage examples and demonstrations
├─ requirements.txt          Python package dependencies
└─ README.md                 Updated project documentation


================================================================================
INSTALLATION & SETUP
================================================================================

1. INSTALL DEPENDENCIES:
   pip install -r requirements.txt

   This installs:
   - gurobipy (Gurobi Python API)
   - pandas (data manipulation)
   - numpy (numerical computing)

2. GUROBI LICENSE:
   Gurobi requires a license (free for academic use):
   - University license: Available through most institutions
   - Individual academic license: Free from gurobi.com
   - Commercial: Available with cost


================================================================================
QUICK START
================================================================================

OPTION A: Command line (simplest)
   python solver.py           # Runs full pipeline with defaults

OPTION B: Python script (most flexible)
   from solver import FIFA2026Solver
   solver = FIFA2026Solver()
   solution = solver.run_full_pipeline()
   solver.print_solution_summary()
   solver.save_solution("results.txt")

OPTION C: CLI with options
   python main.py optimize     # Full optimization
   python main.py test         # Load + build, no solve (for debugging)
   python main.py analyze      # Data exploration
   python main.py compare      # Compare multiple scenarios

OPTION D: Run examples
   python examples.py 1        # Example 1: basic optimization
   python examples.py 2        # Example 2: step-by-step execution
   python examples.py 3        # Example 3: data exploration
   ... etc (see examples.py for full list)


================================================================================
DETAILED MODULE GUIDE
================================================================================

1. CONFIG.PY
   Sets global parameters and constants
   
   Key sections:
   - Hard constraint parameters (REST_MIN, MATCH_COUNTS_BY_COUNTRY)
   - KPI weights for objective function (KPI_WEIGHTS)
   - US entry restrictions (FULL_BAN_TEAMS, VISA_BOND_TEAMS)
   - Gurobi solver settings (GUROBI_TIME_LIMIT, GUROBI_MIP_GAP)
   - Jet-lag penalty function (piecewise linear)
   - Data file paths
   
   Customization:
   - Adjust KPI weights for different objectives
   - Modify time/gap limits for faster/higher quality solutions
   - Add/remove teams from restriction lists


2. UTILS.PY
   Utility functions for calculations
   
   Key functions:
   - great_circle_distance(lat1, lon1, lat2, lon2): Haversine formula
   - perceived_kickoff_time(hour, venue_tz, camp_tz): Time zone adjustment
   - jet_lag_penalty(perceived_hour): Piecewise linear penalty
   - altitude_disruption(venue_elev, camp_elev): Altitude impact
   - countries_differ(): Border crossing indicator
   - wbgt_excess_heat_load(): Heat stress calculation
   
   All functions precompute offline to keep MILP linear


3. DATA_LOADER.PY
   Loads CSV files and provides access methods
   
   DataLoader class:
   - load_all(): Load all CSV files
   - get_team_matches(team_id): Matches for a team
   - get_team_eligible_camps(team_id): Eligible camps for a team
   - get_group_teams(group_letter): Teams in a group
   - get_venues_in_country(country): Stadiums in a country
   - get_camps_in_country(country): Base camps in a country
   
   Usage example:
   ```python
   from data_loader import load_data
   data = load_data()
   team_matches = data.get_team_matches('BRA')
   eligible_camps = data.get_team_eligible_camps('BRA')
   ```


4. PARAMETER_BUILDER.PY
   Precomputes all model parameters
   
   ParameterBuilder class builds:
   - dist[camp_id][venue_id]: Great-circle distances
   - Phi[team_id][camp_id][venue_id][hour]: Jet-lag penalties
   - A[camp_id][venue_id]: Altitude penalties
   - beta[camp_id][venue_id]: Border-crossing indicators
   - D[team_id][camp_id][venue_id]: Travel contributions
   - P[team_id][camp_id]: Visa penalties
   - M_exclusivity: Big-M for conditional constraints
   
   All precomputation avoids in-model nonlinearity


5. MODEL_BUILDER.PY
   Constructs the Gurobi MILP
   
   ModelBuilder class:
   - _create_decision_variables(): Define x, y, z, u
   - _add_hard_constraints(): H1-H8 constraints
   - _add_lower_level_constraints(): Selection, exclusivity, cost, McCormick
   - _add_kpi_constraints(): Auxiliary variables for KPIs
   - _set_objective(): Define objective function
   
   Model statistics:
   - ~70,000+ variables (depending on data)
   - ~50,000+ constraints
   - Solves in 10-60 minutes (typical)


6. MODEL_BUILDER_ADVANCED.PY
   Enhanced model builder for full KPI implementation
   
   Can be extended with:
   - All 13 KPI constraints
   - Piecewise linear linearizations
   - More sophisticated epigraph variables
   
   Replace model_builder.build_model() with:
   from model_builder_advanced import build_advanced_model


7. SOLVER.PY
   Main orchestrator tying everything together
   
   FIFA2026Solver class:
   - load_data(): Step 1
   - build_parameters(): Step 2
   - build_model(): Step 3
   - solve(time_limit, mip_gap): Step 4
   - extract_solution(): Step 5
   - print_solution_summary(): Display results
   - save_solution(filename): Write to file
   - run_full_pipeline(): Execute all 5 steps
   
   Typical usage:
   ```python
   solver = FIFA2026Solver()
   solution = solver.run_full_pipeline(time_limit=3600, mip_gap=0.01)
   solver.print_solution_summary()
   solver.save_solution()
   ```


8. MAIN.PY
   Command-line interface and helper functions
   
   Functions:
   - run_optimization(): Full optimization with defaults
   - run_quick_test(): Load + build, no solve
   - run_with_custom_weights(): Custom KPI weights
   - analyze_data(): Exploratory analysis
   - compare_scenarios(): Compare multiple configurations
   
   CLI commands:
   python main.py optimize    # Default full optimization
   python main.py test        # Quick validation
   python main.py analyze     # Data exploration
   python main.py compare     # Scenario comparison


9. EXAMPLES.PY
   Eight usage examples demonstrating different approaches
   
   Examples:
   1. Basic full pipeline
   2. Step-by-step execution with inspection
   3. Data exploration and analysis
   4. Custom configuration
   5. Access solution details
   6. Solver configuration comparison
   7. Model builder comparison (basic vs. advanced)
   8. Parameter inspection
   
   Run: python examples.py [1-8]


================================================================================
KEY DESIGN DECISIONS
================================================================================

1. MODULAR STRUCTURE:
   Separating concerns makes code:
   - Easier to test and debug
   - More maintainable
   - Reusable in different contexts
   - Easier to understand for new developers

2. PARAMETER PRECOMPUTATION:
   All nonlinear functions (distances, penalties) computed offline:
   - Keeps MILP linear (all constraints/objective linear in x,y,z,u)
   - Dramatically faster solving
   - Exact (no linearization error)
   - Makes model size manageable

3. McCORMICK LINEARIZATION:
   For product u = z_ib × x_mts (binary × binary):
   - Standard envelope: u ≤ z, u ≤ x, u ≥ z+x-1
   - Exact at binary endpoints (no relaxation)
   - Only generated for team's own matches (~144 vars per team)
   - Total product vars: ~7,000

4. CONDITIONAL OPTIMALITY CUTS:
   Base camp exclusivity breaks team independence:
   - Standard cut: C_i* ≤ C_ib ∀b would be infeasible
   - Solution: Add Big-M term if camp claimed by another team
   - Implements "no justified envy" equilibrium concept
   - Exact for discrete choice problem (no KKT needed)

5. BASE CAMP ELIGIBILITY:
   Currently: ALL teams can choose from ALL base camps
   To change:
   - Modify data_loader.py: _build_index_structures()
   - Filter eligible_camps_by_team[team_id] by team's home region
   - Adjust parameter_builder.py accordingly


================================================================================
CUSTOMIZATION EXAMPLES
================================================================================

EXAMPLE 1: Run with custom time limit
   solver = FIFA2026Solver()
   solver.solve(time_limit=7200)  # 2 hours instead of default 1 hour

EXAMPLE 2: Stricter optimality requirement
   solver = FIFA2026Solver()
   solver.solve(mip_gap=0.001)  # 0.1% gap instead of 1%

EXAMPLE 3: Travel-focused scenario (minimize travel only)
   from config import KPI_WEIGHTS
   KPI_WEIGHTS = {k: 0 for k in KPI_WEIGHTS}
   KPI_WEIGHTS[1.2] = 1.0  # Only intra-group travel dispersion

EXAMPLE 4: Fairness-focused scenario
   from config import KPI_WEIGHTS
   KPI_WEIGHTS[1.2] = 2.0  # Travel fairness
   KPI_WEIGHTS[4.1] = 2.0  # Venue load balance
   KPI_WEIGHTS[5.3] = 2.0  # Host-city equity

EXAMPLE 5: Restrict base camp eligibility by region
   Edit data_loader.py _build_index_structures():
   ```python
   # Restrict teams to camps in their own region
   for team_id in self.team_by_id.keys():
       home_region = determine_region(team_id)
       unassigned_camps = [c for c in unassigned_camps 
                          if camp_location(c) == home_region]
       self.eligible_camps_by_team[team_id] = [...] + unassigned_camps
   ```


================================================================================
TROUBLESHOOTING
================================================================================

PROBLEM: Model is infeasible
SOLUTION:
- Check US entry restrictions: teams might have no eligible camps outside US
- Verify base_camps.csv has teams assigned
- Run examples.py with data exploration

PROBLEM: Solver takes too long
SOLUTION:
- Increase MIP gap: solve(mip_gap=0.05) for 5%
- Reduce time limit: solve(time_limit=600) for 10 minutes
- Check Gurobi solver output (increase verbosity)

PROBLEM: Out of memory
SOLUTION:
- Reduce problem size (fewer matches, venues, teams)
- Use model_builder.py instead of model_builder_advanced.py
- Increase Gurobi thread limit (uses shared memory)

PROBLEM: Gurobi license error
SOLUTION:
- Activate license: grbgetkey [key]
- Set GUROBI_HOME environment variable
- Use trial/cloud license from Gurobi website


================================================================================
OUTPUT FILES
================================================================================

When solver.py runs, it produces:
1. Console output (progress, statistics, timing)
2. solution_output.txt (base camp assignments, objective)
3. model.lp (optional, for model inspection)
4. Gurobi .log file (solver details)

Solution file includes:
- Objective value
- Solver status (OPTIMAL, TIME_LIMIT, etc.)
- Time elapsed
- Base camp assignment for each team
- KPI values


================================================================================
PERFORMANCE EXPECTATIONS
================================================================================

Model size:        ~70,000 variables, ~50,000 constraints
Solve time:        10-60 minutes (typical with 1h time limit)
Memory usage:      2-4 GB
CPU threads:       Uses 4 by default (configurable)

Quality achievable:
- 1% gap:          ~15-30 minutes
- 2% gap:          ~5-15 minutes
- 5% gap:          ~1-5 minutes


================================================================================
NEXT STEPS
================================================================================

1. VALIDATE:
   python examples.py 3  # Data exploration
   python main.py test   # Quick model check

2. RUN BASELINE:
   python solver.py      # Full optimization

3. EXPLORE:
   python examples.py 6  # Compare configurations
   python main.py compare  # Compare scenarios

4. ENHANCE:
   - Implement all 13 KPIs in model_builder_advanced.py
   - Add post-processing for KPI calculation
   - Generate visualization of schedule
   - Implement warm starts and heuristics

5. VALIDATE SOLUTION:
   - Check base camp assignments feasible
   - Verify schedule satisfies all hard constraints
   - Calculate exact KPI values from solution


================================================================================
REFERENCES
================================================================================

Mathematical formulation:  Bilevel Opt.tex
Data schema:              data/data_schema.md
Gurobi documentation:     https://www.gurobi.com/documentation/
Python-Gurobi guide:      https://www.gurobi.com/documentation/current/refman/py_python_api_overview.html
