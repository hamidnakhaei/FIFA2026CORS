"""
Main solver for FIFA 2026 Bilevel Optimization
Orchestrates data loading, model building, solving, and result extraction
"""

import os
import sys
from datetime import datetime
import gurobipy as gp
from gurobipy import GRB

from config import DATA_DIR
from data_loader import DataLoader, load_data
from parameter_builder import ParameterBuilder, build_parameters
from model_builder import ModelBuilder, build_model


class FIFA2026Solver:
    """Main solver orchestrator for the bilevel FIFA 2026 optimization."""
    
    def __init__(self, data_dir=DATA_DIR):
        """Initialize the solver."""
        self.data_dir = data_dir
        self.data_loader = None
        self.parameters = None
        self.model = None
        self.solution = None
        self.start_time = None
        self.end_time = None
    
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
        """Build the Gurobi model."""
        print("\n" + "="*70)
        print("STEP 3: BUILDING OPTIMIZATION MODEL")
        print("="*70)
        
        if self.data_loader is None or self.parameters is None:
            raise RuntimeError("Data and parameters must be built first")
        
        self.model = build_model(self.data_loader, self.parameters)
        
        # Write LP file for inspection (optional)
        print("\nModel statistics:")
        print(f"  Variables: {self.model.NumVars}")
        print(f"  Constraints: {self.model.NumConstrs}")
        print(f"  Nonzeros: {self.model.NumNZs}")
        
        return self
    
    def solve(self, time_limit=None, mip_gap=None):
        """
        Solve the optimization problem.
        
        Args:
            time_limit: Override time limit (seconds)
            mip_gap: Override MIP gap tolerance
        """
        print("\n" + "="*70)
        print("STEP 4: SOLVING OPTIMIZATION PROBLEM")
        print("="*70)
        
        if self.model is None:
            raise RuntimeError("Model must be built first (call build_model())")
        
        # Set optional parameters
        if time_limit is not None:
            self.model.setParam('TimeLimit', time_limit)
        if mip_gap is not None:
            self.model.setParam('MIPGap', mip_gap)
        
        print(f"\nSolver starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.start_time = datetime.now()
        
        # Solve
        self.model.optimize()
        
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        print(f"\nSolver finished at {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        # Print solution status
        if self.model.status == GRB.OPTIMAL:
            print(f"\n✓ Optimal solution found!")
            print(f"  Objective value: {self.model.ObjVal:.2f}")
        elif self.model.status == GRB.TIME_LIMIT:
            print(f"\n⚠ Time limit reached")
            if self.model.SolCount > 0:
                print(f"  Best solution found: {self.model.ObjVal:.2f}")
                print(f"  MIP gap: {self.model.MIPGap:.2%}")
            else:
                print(f"  No feasible solution found")
        elif self.model.status == GRB.INFEASIBLE:
            print(f"\n✗ Model is infeasible!")
            # Compute IIS if infeasible
            self.model.computeIIS()
            print(f"  Infeasible constraints: {self.model.IISConstr.shape[0]}")
        else:
            print(f"\n? Solver status: {self.model.status}")
        
        return self
    
    def extract_solution(self):
        """Extract solution into structured format."""
        print("\n" + "="*70)
        print("STEP 5: EXTRACTING SOLUTION")
        print("="*70)
        
        if self.model is None or self.model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            print("⚠ No feasible solution to extract")
            return None
        
        solution = {
            'schedule': {},
            'base_camps': {},
            'objective': self.model.ObjVal,
            'status': self.model.status,
            'solver_time': (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }
        
        # Extract schedule (x variables)
        print("\nExtracting schedule...")
        schedule_count = 0
        for match_id, hours_dict in getattr(self.model, 'x', {}).items():
            for hour, venues_dict in hours_dict.items():
                for venue_id, var in venues_dict.items():
                    if var.X > 0.5:  # Binary variable, so > 0.5 means 1
                        solution['schedule'][(match_id, hour, venue_id)] = 1
                        schedule_count += 1
        print(f"  Extracted {schedule_count} scheduled matches")
        
        # Extract base camps (z variables)
        print("Extracting base camp selections...")
        camp_count = 0
        for team_id, camps_dict in getattr(self.model, 'z', {}).items():
            for camp_id, var in camps_dict.items():
                if var.X > 0.5:  # Binary variable
                    solution['base_camps'][team_id] = camp_id
                    camp_count += 1
        print(f"  Extracted {camp_count} team base camp assignments")
        
        # Compute KPIs
        print("Computing KPIs...")
        solution['kpis'] = self._compute_kpis(solution)
        
        self.solution = solution
        return solution
    
    def _compute_kpis(self, solution):
        """Compute KPIs from the solution."""
        kpis = {}
        
        # Total travel cost
        total_travel = 0
        for team_id, camp_id in solution['base_camps'].items():
            # Simplified: add realized travel cost from model if available
            pass
        
        kpis['total_team_travel_cost'] = total_travel
        kpis['num_teams_assigned'] = len(solution['base_camps'])
        kpis['num_matches_scheduled'] = len(solution['schedule'])
        
        return kpis
    
    def print_solution_summary(self):
        """Print a summary of the solution."""
        if self.solution is None:
            print("No solution to summarize")
            return
        
        print("\n" + "="*70)
        print("SOLUTION SUMMARY")
        print("="*70)
        print(f"Objective value: {self.solution['objective']:.2f}")
        print(f"Solver status: {self.solution['status']}")
        print(f"Solver time: {self.solution['solver_time']:.1f} seconds")
        print(f"\nSchedule: {self.solution['kpis']['num_matches_scheduled']} matches")
        print(f"Base camps: {self.solution['kpis']['num_teams_assigned']} teams assigned")
        print(f"Total team travel cost: {self.solution['kpis']['total_team_travel_cost']:.2f} km")
        print("="*70 + "\n")
    
    def save_solution(self, output_file="solution.txt"):
        """Save solution to file."""
        if self.solution is None:
            print("No solution to save")
            return
        
        with open(output_file, 'w') as f:
            f.write("FIFA 2026 BILEVEL OPTIMIZATION SOLUTION\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Objective value: {self.solution['objective']:.2f}\n")
            f.write(f"Solver status: {self.solution['status']}\n")
            f.write(f"Solver time: {self.solution['solver_time']:.1f} seconds\n\n")
            
            f.write("BASE CAMP ASSIGNMENTS\n")
            f.write("-"*70 + "\n")
            for team_id, camp_id in sorted(self.solution['base_camps'].items()):
                f.write(f"{team_id}: Camp {camp_id}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"Solution saved to {output_file}")
    
    def run_full_pipeline(self, time_limit=None, mip_gap=None):
        """Run the complete pipeline: load, build, solve, extract."""
        try:
            self.load_data()
            self.build_parameters()
            self.build_model()
            self.solve(time_limit=time_limit, mip_gap=mip_gap)
            self.extract_solution()
            self.print_solution_summary()
            return self.solution
        except Exception as e:
            print(f"\n✗ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("FIFA 2026 WORLD CUP BILEVEL OPTIMIZATION SOLVER")
    print("Team UofT Champions | CORS 2026 OR Challenge")
    print("="*70)
    
    # Create solver
    solver = FIFA2026Solver(data_dir=DATA_DIR)
    
    # Run full pipeline
    solution = solver.run_full_pipeline(
        time_limit=3600,  # 1 hour
        mip_gap=0.01      # 1% optimality gap
    )
    
    # Save solution
    if solution is not None:
        solver.save_solution("solution.txt")
    
    return solver


if __name__ == "__main__":
    solver = main()
