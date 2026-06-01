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

from config import (
    DATA_DIR, KPI_WEIGHTS, FULL_BAN_TEAMS, VISA_BOND_TEAMS,
    VISA_BOND_PENALTY, TEAM_PRIORITY_ORDER, OPTSIM_EPSILON, OPTSIM_DELTA,
    OPTSIM_NO_IMPROVE_THRESHOLD,
)

# Map the four base-camp KPI families to their KPI_WEIGHTS keys.
# The guard uses Lambda(K) = sum_k lambda_k * K_k / Kbar_k (Section 9.4):
# weighted AND normalised, so raw travel (thousands of km) does not swamp
# border-crossings (0-3) or altitude (single digits).
KPI_FAMILY_TO_WEIGHT_KEY = {
    'travel': 1.2,            # Intra-group travel dispersion
    'jet_lag': 1.3,           # Circadian shift
    'border_crossings': 1.4,  # Geographic dispersion / border crossings
    'altitude': 2.4,          # Altitude disruption
}
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

        # Per-KPI-family normalisation divisors (Kbar_k). Set from the
        # baseline simulation so the weighted penalty is scale-free.
        self.kpi_normalizers = None
    
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
        
        # Create simulator with the qualification/seeding priority order
        # (Section 9.2: greedy "first come, first served" assignment) and
        # the US entry restriction lists (Section 5.2).
        self.simulator = CampSimulator(
            self.data_loader, self.parameters,
            team_priority_order=TEAM_PRIORITY_ORDER,
            full_ban_teams=FULL_BAN_TEAMS,
            visa_bond_teams=VISA_BOND_TEAMS,
            visa_bond_penalty=VISA_BOND_PENALTY,
        )
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
        # Establish normalisers from the baseline BEFORE scoring, so the
        # weighted penalty Lambda(K) is scale-free from iteration one.
        self._set_normalizers_from_baseline(baseline_kpis)
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

            # No-good cut (Section 9.4): forbid re-finding this exact venue
            # assignment, so the next solve must explore a different schedule.
            # Without this, a deterministic objective would return the same
            # optimum every iteration and the loop would stall after one accept.
            self._add_no_good_cut(candidate_schedule, iteration)

            # Check stopping criteria
            if no_improve_count >= no_improve_threshold:
                print(f"\n  No accepted candidates in {no_improve_threshold} iterations - stopping")
                break
            
            print(f"  Incumbent penalty: {self.incumbent_kpi_penalty:.4f}")
        
        self.end_time = datetime.now()
        
        return self._extract_solution()

    def _add_no_good_cut(self, schedule, iteration):
        """
        Add a no-good cut excluding the venue assignment of `schedule`.

        For the set S1 of (match, venue) pairs that are 1 in this solution,
        require sum_{(m,v) in S1} a[m][v] <= |S1| - 1, which makes at least
        one match move to a different venue in any future solution.
        """
        a = self.model_builder.a
        expr = gp.LinExpr()
        count = 0
        for match_id, info in schedule.items():
            venue_id = info.get('venue_id')
            if match_id in a and venue_id in a[match_id]:
                expr.add(a[match_id][venue_id], 1.0)
                count += 1
        if count > 0:
            self.model.addConstr(expr <= count - 1, f"no_good_cut_{iteration}")
            self.model.update()
    
    def _build_surrogate_objective(self, use_surrogate):
        """
        Build the upper-level objective as a LINEAR Gurobi expression over
        the venue-assignment variables a[match][venue].

        This is the piece that was previously missing: the fitted surrogate
        (Section 9.3) is turned into actual optimisation pressure rather than
        a constant-zero objective.

        Construction. Every base-camp KPI is driven, to first order, by how
        far each team's match venues are from the camps that team could
        realistically use. For team i and venue s define the per-venue cost

            c[i][s] = (min over eligible camps b of 2*dist(b,s)) + visa term,

        i.e. the cheapest round-trip that placing a match of i at s could
        incur once i picks its best eligible camp. This is a constant
        (precomputed) and is LINEAR in a[m][s]. The surrogate objective is

            sum_i  W_i * sum_{m in M_i} sum_s c[i][s] * a[m][s]

        where W_i aggregates the fitted per-KPI surrogate weights and the
        normalised KPI_WEIGHTS, so the objective tracks the simulated penalty
        the guard actually measures.

        When use_surrogate is False (baseline/initialisation) we use unit KPI
        weights so the baseline still minimises travel-like cost rather than
        returning an arbitrary feasible point.
        """
        dist = self.parameters['dist'] if isinstance(self.parameters, dict) \
            else self.parameters.params['dist']

        # US camps + per-team eligible camps (respecting the full ban) -------
        us_camps = set(self.data_loader.get_camps_in_country("USA"))
        all_camps = list(self.data_loader.camp_by_id.keys())

        # Aggregate surrogate weight per KPI family. Combine the fitted
        # surrogate magnitude with the analyst KPI_WEIGHTS so the in-MILP
        # objective and the guard penalty are commensurate.
        fam_weight = {}
        for fam, wkey in KPI_FAMILY_TO_WEIGHT_KEY.items():
            lam = KPI_WEIGHTS.get(wkey, 1.0)
            if use_surrogate and self.surrogate is not None and self.surrogate.is_fitted:
                # L1 magnitude of the fitted feature weights = how strongly
                # this KPI responds to schedule features.
                mag = float(np.sum(np.abs(self.surrogate.kpi_weights[fam])))
            else:
                mag = 1.0
            norm = 1.0
            if self.kpi_normalizers:
                norm = max(self.kpi_normalizers.get(fam, 1.0), 1e-9)
            fam_weight[fam] = lam * mag / norm

        # Precompute per-team, per-venue cost c[i][s] -----------------------
        venue_ids = [v['venue_id'] for _, v in self.data_loader.venues.iterrows()]
        team_venue_cost = {}
        for team_id in self.data_loader.team_by_id.keys():
            eligible = list(self.data_loader.eligible_camps_by_team.get(team_id, all_camps))
            if team_id in FULL_BAN_TEAMS:
                eligible = [c for c in eligible if c not in us_camps]
            team_venue_cost[team_id] = {}
            for venue_id in venue_ids:
                best = float('inf')
                for camp_id in eligible:
                    d = dist.get(camp_id, {}).get(venue_id, None)
                    if d is None:
                        continue
                    cost = 2.0 * d
                    # Soft visa-bond friction for US venues (Section 5.2):
                    # a bond team sited away from the US still must travel to
                    # any US match, so US venues carry the bond penalty.
                    if team_id in VISA_BOND_TEAMS and venue_id in \
                            self.data_loader.get_venues_in_country("USA"):
                        cost += VISA_BOND_PENALTY
                    best = min(best, cost)
                team_venue_cost[team_id][venue_id] = 0.0 if best == float('inf') else best

        # Assemble the linear objective over a[match][venue] ----------------
        obj = gp.LinExpr()
        a = self.model_builder.a
        for team_id in self.data_loader.team_by_id.keys():
            # One scalar weight per team: the travel/border/altitude families
            # all scale with venue distance, so we fold them into a single
            # multiplier. (Jet-lag is slot-driven and is left to the guard.)
            W_i = (fam_weight['travel'] + fam_weight['border_crossings']
                   + fam_weight['altitude'])
            if W_i <= 0:
                W_i = 1.0
            matches = self.data_loader.get_team_matches(team_id)
            for mdict in matches:
                match_id = mdict['match_id']
                if match_id not in a:
                    continue
                for venue_id in venue_ids:
                    coef = W_i * team_venue_cost[team_id].get(venue_id, 0.0)
                    if coef != 0.0:
                        obj.add(a[match_id][venue_id], coef)
        return obj

    def _optimize_with_surrogate(self, time_limit, use_surrogate=True):
        """
        Solve the MILP with the current surrogate KPI objective.

        Args:
            time_limit: Time limit for Gurobi
            use_surrogate: If True, use the fitted surrogate weights in the
                          objective. If False, use unit weights (baseline).

        Returns:
            Schedule solution if successful, None otherwise
        """
        import sys
        import time as time_module

        # Build the LINEAR surrogate objective over a[match][venue] and set it.
        print(f"       Building surrogate objective "
              f"(use_surrogate={use_surrogate})...")
        sys.stdout.flush()
        obj_expr = self._build_surrogate_objective(use_surrogate)
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
    
    def _set_normalizers_from_baseline(self, baseline_kpis):
        """
        Set per-KPI-family normalisation divisors Kbar_k from the baseline
        simulation. Using the baseline's own family totals makes the weighted
        penalty Lambda(K) scale-free, so no single KPI (e.g. travel in km)
        dominates the guard's accept/reject decision.
        """
        normalizers = {}
        for fam in KPI_FAMILY_TO_WEIGHT_KEY.keys():
            team_values = baseline_kpis.get(fam, {})
            total = sum(team_values.values()) if isinstance(team_values, dict) else float(team_values)
            # Guard against zero/near-zero baselines.
            normalizers[fam] = total if abs(total) > 1e-9 else 1.0
        self.kpi_normalizers = normalizers

    def _compute_kpi_penalty(self, kpis):
        """
        Compute the weighted, normalised penalty Lambda(K) (Section 9.4):

            Lambda(K) = sum_k  lambda_k * (K_k / Kbar_k)

        where K_k is the total of KPI family k over all teams, lambda_k is the
        analyst weight from KPI_WEIGHTS, and Kbar_k is the baseline normaliser.
        This replaces the previous raw, unweighted sum in which travel (km)
        swamped border-crossings (0-3) and altitude.

        Args:
            kpis: Dict with KPI values from simulation

        Returns:
            Scalar penalty
        """
        penalty = 0.0
        for fam, wkey in KPI_FAMILY_TO_WEIGHT_KEY.items():
            team_values = kpis.get(fam, {})
            total = sum(team_values.values()) if isinstance(team_values, dict) else float(team_values)
            lam = KPI_WEIGHTS.get(wkey, 1.0)
            norm = 1.0
            if self.kpi_normalizers:
                norm = max(self.kpi_normalizers.get(fam, 1.0), 1e-9)
            penalty += lam * (total / norm)
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