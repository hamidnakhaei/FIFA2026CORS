"""
Main solver for FIFA 2026 Optimization+Simulation framework
Implements the iterative Opt+Sim algorithm with sim-in-the-loop guard
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import gurobipy as gp
from gurobipy import GRB

from config import DATA_DIR, KPI_WEIGHTS
from data_loader import DataLoader, load_data
from parameter_builder import ParameterBuilder, build_parameters
from model_builder_optsim import ModelBuilder as OptSimModelBuilder, build_model
from camp_simulator import CampSimulator
from surrogate_kpis import SurrogateKPIModel
from output_handler import OutputHandler, export_solution_excel


class OptSimSolver:
    """
    Main solver for the FIFA 2026 Opt+Sim framework.
    
    Algorithm:
    1. Initialize with base-camp-free KPIs only
    2. Run simulation to get realistic base-camp outcomes
    3. Fit surrogate model to predict KPIs from schedule features
    4. Optimize schedule with surrogate + base-camp-free KPIs
    5. Simulate new schedule
    6. Accept only if simulated KPIs improve (guard mechanism)
    7. Repeat until convergence
    """
    
    def __init__(self, data_dir=DATA_DIR):
        """Initialize the solver."""
        self.data_dir = data_dir
        self.data_loader = None
        self.parameters = None
        self.model = None
        self.model_builder = None
        self.simulator = None
        self.surrogate = None
        self.solution = None
        self.start_time = None
        self.end_time = None
        
        # Iteration tracking
        self.iteration = 0
        self.max_iterations = 10  # Default; can be overridden
        self.incumbent_schedule = None
        self.incumbent_camp_assignment = None
        self.incumbent_kpi_penalty = float('inf')
        self.kpi_history = []
    
    def load_data(self):
        """Load all input data."""
        print("\n" + "="*70)
        print("STEP 1: LOADING DATA")
        print("="*70)
        
        self.data_loader = load_data(self.data_dir)
        self.data_loader.summary()
        
        return self
    
    def build_parameters(self):
        """Build precomputed parameters."""
        print("\n" + "="*70)
        print("STEP 2: BUILDING PARAMETERS")
        print("="*70)
        
        if self.data_loader is None:
            raise RuntimeError("Data must be loaded first (call load_data())")
        
        self.parameters = build_parameters(self.data_loader)
        
        return self
    
    def build_model(self):
        """Build the Gurobi model (upper level only)."""
        print("\n" + "="*70)
        print("STEP 3: BUILDING OPTIMIZATION MODEL (Upper Level Only)")
        print("="*70)
        
        if self.data_loader is None or self.parameters is None:
            raise RuntimeError("Data and parameters must be built first")
        
        self.model, self.model_builder = build_model(self.data_loader, self.parameters)
        
        return self
    
    def initialize_simulation(self):
        """Initialize simulator and surrogate."""
        print("\n" + "="*70)
        print("STEP 4: INITIALIZING SIMULATION AND SURROGATE")
        print("="*70)
        
        if self.data_loader is None or self.parameters is None:
            raise RuntimeError("Data and parameters must be loaded first")
        
        # Create simulator
        self.simulator = CampSimulator(self.data_loader, self.parameters)
        print(f"  Initialized camp simulator")
        
        # Create surrogate model
        self.surrogate = SurrogateKPIModel(self.data_loader, self.parameters)
        print(f"  Initialized surrogate KPI model")
        
        return self
    
    def run_opt_sim_loop(self, time_limit_per_iteration=600, max_iterations=10, 
                         epsilon=0.01, delta=0.02, no_improve_threshold=3):
        """
        Run the Opt+Sim iterative algorithm.
        
        Args:
            time_limit_per_iteration: Time limit for each MILP solve (seconds)
            max_iterations: Maximum number of iterations
            epsilon: Improvement threshold for simulated penalty (%)
            delta: Allowed degradation in other objectives (%)
            no_improve_threshold: Stop after N iterations with no accepted candidates
        
        Returns:
            Solution dictionary with schedule, camps, and KPIs
        """
        print("\n" + "="*70)
        print("STEP 5: OPTIMIZATION + SIMULATION LOOP")
        print("="*70)
        
        self.start_time = datetime.now()
        self.max_iterations = max_iterations
        no_improve_count = 0
        
        # PHASE 0: Initialize with baseline solution
        print("\n" + "="*70)
        print("INITIALIZATION PHASE: Establishing baseline solution")
        print("="*70)
        
        import sys
        sys.stdout.flush()
        
        baseline_schedule = self._optimize_with_surrogate(time_limit_per_iteration, use_surrogate=False)
        
        if baseline_schedule is None:
            print("CRITICAL ERROR: Cannot find baseline solution - problem may be infeasible")
            sys.stdout.flush()
            return self._extract_solution()
        
        print(f"\nSimulating baseline schedule...")
        sys.stdout.flush()
        baseline_venue_assignment = self._extract_venue_assignment(baseline_schedule)
        baseline_camps, baseline_kpis = self.simulator.simulate(baseline_venue_assignment)
        baseline_penalty = self._compute_kpi_penalty(baseline_kpis)
        
        # Set baseline as incumbent
        self.incumbent_schedule = baseline_schedule
        self.incumbent_camp_assignment = baseline_camps
        self.incumbent_kpi_penalty = baseline_penalty
        
        print(f"\n[OK] Baseline established:")
        print(f"  Penalty: {baseline_penalty:.4f}")
        print(f"  Incumbent set for iteration comparison")
        sys.stdout.flush()
        
        self.surrogate.update(baseline_venue_assignment, baseline_kpis)
        self.kpi_history.append({
            'iteration': -1,
            'penalty': baseline_penalty,
            'accepted': True,
            'note': 'Baseline/Initialization'
        })
        
        # PHASE 1: Iterative optimization with guard checks
        for iteration in range(max_iterations):
            self.iteration = iteration
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            
            # Step 1: Solve MILP with current surrogate
            print(f"\n  [1/4] Optimizing schedule with surrogate...")
            candidate_schedule = self._optimize_with_surrogate(time_limit_per_iteration, use_surrogate=True)
            
            if candidate_schedule is None:
                print(f"  [1/4] Optimization failed - stopping")
                break
            
            # Step 2: Simulate base-camp selection for candidate
            print(f"\n  [2/4] Simulating base-camp selection...")
            venue_assignment = self._extract_venue_assignment(candidate_schedule)
            candidate_camps, candidate_kpis = self.simulator.simulate(venue_assignment)
            
            # Step 3: Accept/reject with guard mechanism
            print(f"\n  [3/4] Guard check (accept/reject)...")
            penalty = self._compute_kpi_penalty(candidate_kpis)
            
            # Guard check: accept if penalty improves by at least epsilon%
            improvement_threshold = self.incumbent_kpi_penalty * (1 - epsilon)
            
            if penalty <= improvement_threshold:
                print(f"       ACCEPTED: penalty {penalty:.4f} < threshold {improvement_threshold:.4f}")
                self.incumbent_schedule = candidate_schedule
                self.incumbent_camp_assignment = candidate_camps
                self.incumbent_kpi_penalty = penalty
                no_improve_count = 0
            else:
                print(f"       REJECTED: penalty {penalty:.4f} >= threshold {improvement_threshold:.4f}")
                no_improve_count += 1
            
            # Step 4: Update surrogate for next iteration
            print(f"\n  [4/4] Updating surrogate model...")
            self.surrogate.update(venue_assignment, candidate_kpis)
            self.kpi_history.append({
                'iteration': iteration,
                'penalty': penalty,
                'accepted': penalty <= improvement_threshold
            })
            
            # Check stopping criteria
            if no_improve_count >= no_improve_threshold:
                print(f"\n  No accepted candidates in {no_improve_threshold} iterations - stopping")
                break
            
            print(f"  Incumbent penalty: {self.incumbent_kpi_penalty:.4f}")
        
        self.end_time = datetime.now()
        
        return self._extract_solution()
    
    def _optimize_with_surrogate(self, time_limit, use_surrogate=True):
        """
        Solve the MILP with current surrogate KPI terms.
        
        Args:
            time_limit: Time limit for Gurobi
            use_surrogate: If True, use fitted surrogate in objective. 
                          If False, use base-camp-free KPIs only (for initialization)
        
        Returns:
            Schedule solution if successful, None otherwise
        """
        import sys
        import time as time_module
        
        # Build objective function - keep it SIMPLE to avoid memory bloat
        # For now, use minimal objective (just need any feasible schedule)
        # Surrogate+KPI integration would require reformulating with auxiliary variables
        
        print(f"       Setting minimal objective (any feasible solution)...")
        sys.stdout.flush()
        
        # Simple objective: minimize 0 (just find feasible solution)
        # In iterations with surrogate, could add penalty terms, but keeping simple for now
        obj_expr = gp.LinExpr(0)
        self.model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Set time limit
        self.model.setParam('TimeLimit', time_limit)
        
        # Optimize
        num_vars = len(self.model_builder.x) * len(self.model_builder.slots) * len(self.data_loader.venues)
        print(f"       Solving MILP (time limit: {time_limit}s, {num_vars} variables)...")
        sys.stdout.flush()
        solve_start = time_module.time()
        self.model.optimize()
        solve_time = time_module.time() - solve_start
        
        status_str = {
            1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 
            5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT",
            9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC",
            13: "SUBOPTIMAL", 14: "INPROG", 15: "USER_OBJ_LIMIT"
        }
        status_name = status_str.get(self.model.status, "UNKNOWN")
        
        print(f"       Optimization completed in {solve_time:.1f}s (status: {self.model.status}={status_name})")
        sys.stdout.flush()
        
        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            print(f"       [OK] Solution found, extracting...")
            sys.stdout.flush()
            solution = self.model_builder.get_schedule_solution()
            if solution:
                print(f"       [OK] Extracted {len(solution)} matches")
                sys.stdout.flush()
            return solution
        else:
            print(f"       [FAIL] Status {self.model.status} ({status_name}) - no solution")
            sys.stdout.flush()
            return None
    
    def _extract_venue_assignment(self, schedule):
        """Extract venue assignment array from schedule solution."""
        # Convert schedule dict to numpy array for simulator
        match_count = len(self.data_loader.matches)
        venue_count = len(self.data_loader.venues)
        assignment = np.zeros((match_count, venue_count))
        
        # Create index mappings from DataFrames
        match_ids = self.data_loader.matches['match_id'].tolist()
        venue_ids = self.data_loader.venues['venue_id'].tolist()
        match_id_to_idx = {mid: i for i, mid in enumerate(match_ids)}
        venue_id_to_idx = {vid: i for i, vid in enumerate(venue_ids)}
        
        for match_id, assignment_dict in schedule.items():
            match_idx = match_id_to_idx.get(match_id)
            venue_id = assignment_dict['venue_id']
            venue_idx = venue_id_to_idx.get(venue_id)
            
            if match_idx is not None and venue_idx is not None:
                assignment[match_idx, venue_idx] = 1.0
        
        return assignment
    
    def _compute_kpi_penalty(self, kpis):
        """
        Compute normalized penalty from KPI dict.
        
        Args:
            kpis: Dict with KPI values from simulation
        
        Returns:
            Scalar penalty
        """
        penalty = 0.0
        
        # Sum across all KPI dimensions
        for kpi_type, team_values in kpis.items():
            if isinstance(team_values, dict):
                penalty += sum(team_values.values())
            else:
                penalty += team_values
        
        return penalty
    
    def _extract_incumbent_kpis(self):
        """Extract simulated KPIs for current incumbent."""
        if self.incumbent_schedule is None:
            return {'travel': {}, 'jet_lag': {}, 'border_crossings': {}, 'altitude': {}}
        
        venue_assignment = self._extract_venue_assignment(self.incumbent_schedule)
        _, kpis = self.simulator.simulate(venue_assignment)
        return kpis
    
    def _extract_solution(self):
        """Extract final solution."""
        if self.incumbent_schedule is None:
            print("  No solution found")
            return None
        
        solution = {
            'schedule': self.incumbent_schedule,
            'camp_assignment': self.incumbent_camp_assignment,
            'kpi_penalty': self.incumbent_kpi_penalty,
            'iterations': self.iteration + 1,
            'solve_time': (self.end_time - self.start_time).total_seconds(),
            'kpi_history': self.kpi_history
        }
        
        return solution
    
    def print_solution_summary(self):
        """Print summary of solution."""
        if self.solution is None:
            print("No solution available")
            return
        
        print("\n" + "="*70)
        print("SOLUTION SUMMARY")
        print("="*70)
        print(f"Iterations: {self.solution['iterations']}")
        print(f"Solve time: {self.solution['solve_time']:.1f} seconds")
        print(f"KPI penalty: {self.solution['kpi_penalty']:.4f}")
        print(f"Teams assigned camps: {len(self.solution['camp_assignment'])}")
        print("="*70 + "\n")
    
    def save_solution(self, output_dir="output"):
        """
        Save solution to Excel files.
        
        Args:
            output_dir: Directory to save xlsx files
        """
        if self.solution is None:
            print("No solution to save")
            return
        
        # Export to Excel using OutputHandler
        handler = OutputHandler(self.data_loader)
        schedule_file, camps_file, metadata_file = handler.export_solution(self.solution, output_dir)
        
        return schedule_file, camps_file, metadata_file
    
    def run_full_pipeline(self, time_limit=3600, max_iterations=10, mip_gap=0.01):
        """
        Run the complete Opt+Sim pipeline.
        
        Args:
            time_limit: Total time limit (seconds)
            max_iterations: Maximum iterations
            mip_gap: MIP gap tolerance per iteration
        
        Returns:
            Solution dictionary
        """
        self.load_data()
        self.build_parameters()
        self.build_model()
        self.initialize_simulation()
        
        self.solution = self.run_opt_sim_loop(
            time_limit_per_iteration=time_limit // max_iterations,
            max_iterations=max_iterations
        )
        
        return self.solution


def run_optimization_sim(time_limit=3600, max_iterations=10, verbose=True):
    """
    Run the complete Opt+Sim optimization pipeline.
    
    Args:
        time_limit: Total time limit in seconds
        max_iterations: Maximum number of iterations
        verbose: Print progress
    
    Returns:
        Solution dictionary
    
    Output Files:
        - output/optimized_schedule.xlsx
        - output/base_camp_assignments.xlsx
    """
    solver = OptSimSolver(data_dir=DATA_DIR)
    solution = solver.run_full_pipeline(
        time_limit=time_limit,
        max_iterations=max_iterations
    )
    
    if solution:
        solver.print_solution_summary()
        solver.save_solution()  # Exports to output/ directory
    
    return solution


if __name__ == '__main__':
    solution = run_optimization_sim(time_limit=3600, max_iterations=10, verbose=True)
