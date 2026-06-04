"""
Main algorithm orchestrator for FIFA 2026 Two-Step Iterative Optimization.
Implements the alternating minimization algorithm with stochastic exploration.
"""

import pandas as pd
from typing import Dict
from data_loader import DataLoader
from kpis import KPICalculator
from schedule_optimizer import ScheduleOptimizer
from base_camp_optimizer import BaseCampOptimizer, load_base_camp_assignment_from_data


class TwoStepOptimizer:
    """
    Two-step iterative optimizer for FIFA 2026 World Cup Group-Stage.

    Algorithm:
    - Step A: Hold base camps fixed, solve MILP for schedule
    - Step B: Hold schedule fixed, optimize base camps with Metropolis exploration
    - Repeat Steps A and B for max_iterations or until convergence
    """

    def __init__(
        self,
        data_dir: str = "data",
        solver_name: str = "glpk"):
        self.data_dir = data_dir
        self.solver_name = solver_name

        # Load data
        self.loader = DataLoader(data_dir)
        self.data = self.loader.load_all()
        params = self.loader.get_parameters()

        # Initialize
        self.kpi_calc = KPICalculator(self.loader, params)
        self.best_objective = float("inf")
        self.best_solution = None
        self.iteration_history = []

    def get_initial_base_camp_assignment(self) -> Dict:
        """Load initial base camp assignment from CSV data (confirmed assignments)."""
        base_camps = self.loader.get_base_camps()
        assignment = load_base_camp_assignment_from_data(base_camps)
        return assignment

    def run_step_a(self, time_limit: int = 300) -> Dict:
        """
        Step A: Optimize schedule while holding base camps fixed.
        Uses full KPI objective (all 13 KPIs) in the MILP solver.

        Args:
            time_limit: Solver time limit in seconds

        Returns:
            Dictionary with schedule and full KPI objective value
        """

        # Solve schedule optimization
        optimizer = ScheduleOptimizer(self.loader, self.kpi_calc)
        optimizer.build_model()
        result = optimizer.solve(time_limit=time_limit, solver_name=self.solver_name)
        schedule = result["schedule"]
        objective_full = result["objective"]

        print(f"  ✓ Solver status: {result['status']}")
        print(f"  ✓ Full KPI objective (all 13 KPIs): {objective_full:.2f}")
        print(f"  ✓ Matches scheduled: {len(schedule)}")

        return {"schedule": schedule, "objective_full": objective_full, "schedule_df": result["schedule_df"]}
    

    def run_step_b(
        self,
        schedule: Dict) -> Dict:
        """
        Step B: Optimize base camps while holding schedule fixed.
        Uses full KPI objective (all 13 KPIs) to evaluate base camp changes.
        Args:
            schedule: Fixed schedule from Step A
        Returns:
            Dictionary with optimized base camp assignment and objective
        """
        optimizer = BaseCampOptimizer(self.loader, self.kpi_calc, schedule)
        result = optimizer.optimize()
        print(f"  ✓ Base camp optimization completed") 
        print(f"  ✓ Solver status: {result['status']}")
        return result["assignment"]


    def run(
        self) -> Dict:
        """
        Run the two-step optimization algorithm.
        Args:
            max_iterations: Maximum number of iterations
        Returns:
            Dictionary with final solution and history
        """

        # Step A: Optimize schedule
        print(f"\n---- Step A: Optimize Schedule ---")
        step_a_result = self.run_step_a()
        new_schedule = step_a_result["schedule"]
        new_schedule_df = step_a_result["schedule_df"]
        objective_a = step_a_result["objective_full"]

        # Step B: Optimize base camps
        print(f"\n---- Step B: Optimize Base Camps ---")
        new_base_camp = self.run_step_b(new_schedule)
        
        return new_schedule, new_base_camp, new_schedule_df


def main():
    """Main entry point."""
    # Initialize optimizer
    optimizer = TwoStepOptimizer(data_dir="data", solver_name="gurobi")
    # Run optimization
    result = optimizer.run()
    csv_results_1 = pd.DataFrame(result[2], columns=['match', 'time', 'stadium'])
    csv_results_1.to_csv("schedule_results.csv", index=False)
    
    # Convert base camp assignment dict to DataFrame
    base_camp_dict = result[1]
    csv_results_2 = pd.DataFrame([
        {"team_id": team_id, "base_camp_id": base_camp_id}
        for team_id, base_camp_id in base_camp_dict.items()
    ])
    csv_results_2.to_csv("base_camp_results.csv", index=False)
    return result


if __name__ == "__main__":
    result = main()
