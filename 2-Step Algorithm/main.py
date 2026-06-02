"""
Main algorithm orchestrator for FIFA 2026 Two-Step Iterative Optimization.
Implements the alternating minimization algorithm with stochastic exploration.
"""


from typing import Dict, Tuple
import pickle
from datetime import datetime

from data_loader import DataLoader
from kpis import KPICalculator
from schedule_optimizer import ScheduleOptimizer
from base_camp_optimizer import BaseCampOptimizer, generate_initial_assignment, load_base_camp_assignment_from_data


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
        solver_name: str = "glpk",
        verbose: bool = True,
    ):
        self.data_dir = data_dir
        self.solver_name = solver_name
        self.verbose = verbose

        # Load data
        self.loader = DataLoader(data_dir)
        self.data = self.loader.load_all()

        # Initialize
        self.kpi_calc = None
        self.best_objective = float("inf")
        self.best_solution = None
        self.iteration_history = []

    def log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(message)

    def get_initial_base_camp_assignment(self) -> Dict:
        """Load initial base camp assignment from CSV data (confirmed assignments)."""
        base_camps = self.loader.get_base_camps()
        assignment = load_base_camp_assignment_from_data(base_camps)
        
        # Log assignments loaded
        if assignment:
            self.log(f"✓ Loaded {len(assignment)} base camp assignments from data")
        else:
            self.log("⚠ No confirmed base camp assignments found in data, using fallback")
            assignment = generate_initial_assignment(self.loader.get_teams())
        
        return assignment

    def run_step_a(self, base_camp_assignment: Dict, time_limit: int = 300) -> Dict:
        """
        Step A: Optimize schedule while holding base camps fixed.
        Uses full KPI objective (all 13 KPIs) in the MILP solver.

        Args:
            base_camp_assignment: Fixed base camp assignment
            time_limit: Solver time limit in seconds

        Returns:
            Dictionary with schedule and full KPI objective value
        """
        self.log("\n[Step A] Optimizing schedule (base camps fixed)...")

        # Initialize KPI calculator if not already done
        if self.kpi_calc is None:
            params = self.loader.get_parameters()
            self.kpi_calc = KPICalculator(self.loader, params)

        # Solve schedule optimization
        optimizer = ScheduleOptimizer(self.loader, self.kpi_calc, base_camp_assignment)
        optimizer.build_model()

        try:
            result = optimizer.solve(time_limit=time_limit, solver_name=self.solver_name)
            schedule = result["schedule"]
            objective_full = result["objective"]

            self.log(f"  ✓ Solver status: {result['status']}")
            self.log(f"  ✓ Full KPI objective (all 13 KPIs): {objective_full:.2f}")
            self.log(f"  ✓ Matches scheduled: {len(schedule)}")

            return {"schedule": schedule, "objective_full": objective_full}
        except Exception as e:
            self.log(f"  ⚠ Solver error: {e}")
            # Return dummy schedule if solver fails
            dummy_schedule = {i: (i % 24, f"S{i % 16}") for i in range(1, 73)}
            return {"schedule": dummy_schedule, "objective_full": float("inf")}

    def run_step_b(
        self,
        schedule: Dict,
        initial_base_camp: Dict,
        max_iterations: int = 500,
        temperature_schedule: str = "exponential",
        num_changes_per_iteration: int = 2,
    ) -> Dict:
        """
        Step B: Optimize base camps while holding schedule fixed.
        Uses full KPI objective (all 13 KPIs) to evaluate base camp changes.

        Args:
            schedule: Fixed schedule from Step A
            initial_base_camp: Initial base camp assignment
            max_iterations: SA iterations
            temperature_schedule: Temperature schedule type
            num_changes_per_iteration: Number of teams' base camps to change per iteration

        Returns:
            Dictionary with optimized base camp assignment and objective
        """
        self.log("\n[Step B] Optimizing base camps (schedule fixed)...")

        optimizer = BaseCampOptimizer(self.loader, self.kpi_calc, schedule)
        result = optimizer.optimize(
            initial_assignment=initial_base_camp,
            max_iterations=max_iterations,
            temperature_schedule=temperature_schedule,
            num_changes_per_iteration=num_changes_per_iteration,
        )

        self.log(f"  ✓ Best cost: {result['best_cost']:.2f}")
        self.log(f"  ✓ Acceptance rate: {result['acceptance_rate']:.2%}")

        return {
            "base_camp_assignment": result["best_assignment"],
            "objective": result["best_cost"],
            "acceptance_rate": result["acceptance_rate"],
        }

    def run_iteration(
        self,
        iteration_num: int,
        base_camp_assignment: Dict,
        schedule: Dict = None,
        step_a_time_limit: int = 300,
        step_b_iterations: int = 500,
        step_b_num_changes: int = 2,
    ) -> Tuple[Dict, Dict]:
        """
        Run a single iteration of the two-step algorithm.

        Args:
            iteration_num: Iteration number (0-indexed)
            base_camp_assignment: Current base camp assignment
            schedule: Current schedule (None for iteration 0)
            step_a_time_limit: Time limit for Step A solver
            step_b_iterations: Number of SA iterations for Step B
            step_b_num_changes: Number of base camps to change per Step B iteration

        Returns:
            Tuple of (updated_schedule, updated_base_camp_assignment)
        """
        self.log(f"\n{'='*60}")
        self.log(f"ITERATION {iteration_num + 1}")
        self.log(f"{'='*60}")

        # Step A: Optimize schedule
        step_a_result = self.run_step_a(base_camp_assignment, time_limit=step_a_time_limit)
        new_schedule = step_a_result["schedule"]
        objective_a = step_a_result["objective_full"]

        # Step B: Optimize base camps
        step_b_result = self.run_step_b(
            new_schedule,
            base_camp_assignment,
            max_iterations=step_b_iterations,
            num_changes_per_iteration=step_b_num_changes,
        )
        new_base_camp = step_b_result["base_camp_assignment"]
        objective_b = step_b_result["objective"]

        # Record iteration history
        iteration_info = {
            "iteration": iteration_num + 1,
            "objective_a": objective_a,
            "objective_b": objective_b,
            "timestamp": datetime.now().isoformat(),
        }
        self.iteration_history.append(iteration_info)

        # Update best solution (compare full KPI objectives)
        if objective_b < self.best_objective:
            self.best_objective = objective_b
            self.best_solution = {
                "schedule": new_schedule,
                "base_camp_assignment": new_base_camp,
                "objective": objective_b,
                "iteration": iteration_num + 1,
            }
            self.log(f"\n  ★ NEW BEST SOLUTION: {objective_b:.2f}")

        self.log(
            f"\nIteration {iteration_num + 1} Summary:"
            f"\n  Step A objective: {objective_a:.2f}"
            f"\n  Step B objective: {objective_b:.2f}"
            f"\n  Best so far: {self.best_objective:.2f}"
        )

        return new_schedule, new_base_camp

    def run(
        self,
        max_iterations: int = 3,
        step_a_time_limit: int = 300,
        step_b_iterations: int = 500,
        step_b_num_changes: int = 2,
        convergence_tol: float = 0.01,
    ) -> Dict:
        """
        Run the two-step optimization algorithm.

        Args:
            max_iterations: Maximum number of iterations
            step_a_time_limit: Time limit for each Step A solve (seconds)
            step_b_iterations: Number of SA iterations per Step B
            step_b_num_changes: Number of base camps to change per Step B iteration (default: 2)
            convergence_tol: Objective change threshold for early stopping

        Returns:
            Dictionary with final solution and history
        """
        self.log(
            f"\n{'='*60}\n"
            f"FIFA 2026 TWO-STEP OPTIMIZATION ALGORITHM\n"
            f"{'='*60}\n"
        )
        self.log(f"Configuration:")
        self.log(f"  Max iterations: {max_iterations}")
        self.log(f"  Step A time limit: {step_a_time_limit}s")
        self.log(f"  Step B SA iterations: {step_b_iterations}")
        self.log(f"  Step B base camps per iteration: {step_b_num_changes}")
        self.log(f"  Convergence tolerance: {convergence_tol}")

        # Initialize
        base_camp_assignment = self.get_initial_base_camp_assignment()
        schedule = None
        prev_objective = float("inf")

        # Main loop
        for iteration_num in range(max_iterations):
            schedule, base_camp_assignment = self.run_iteration(
                iteration_num,
                base_camp_assignment,
                schedule,
                step_a_time_limit,
                step_b_iterations,
                step_b_num_changes,
            )

            # Check convergence
            obj_change = abs(prev_objective - self.best_objective)
            if obj_change < convergence_tol and iteration_num > 0:
                self.log(f"\n✓ Converged! Objective change: {obj_change:.6f} < {convergence_tol}")
                break

            prev_objective = self.best_objective

        # Final summary
        self.log(f"\n{'='*60}")
        self.log(f"OPTIMIZATION COMPLETE")
        self.log(f"{'='*60}")
        self.log(f"\nFinal Results:")
        self.log(f"  Total iterations: {len(self.iteration_history)}")
        self.log(f"  Best objective: {self.best_objective:.2f}")
        self.log(f"  Matches scheduled: {len(self.best_solution['schedule'])}")
        self.log(f"  Teams assigned: {len(self.best_solution['base_camp_assignment'])}")

        return {
            "best_solution": self.best_solution,
            "iteration_history": self.iteration_history,
            "best_objective": self.best_objective,
            "success": self.best_solution is not None,
        }

    def save_solution(self, filename: str):
        """Save best solution to pickle file."""
        if self.best_solution is not None:
            with open(filename, "wb") as f:
                pickle.dump(self.best_solution, f)
            self.log(f"✓ Solution saved to {filename}")
        else:
            self.log("⚠ No solution to save")

    def load_solution(self, filename: str):
        """Load solution from pickle file."""
        with open(filename, "rb") as f:
            self.best_solution = pickle.load(f)
        self.log(f"✓ Solution loaded from {filename}")


def main():
    """Main entry point."""
    # Initialize optimizer
    optimizer = TwoStepOptimizer(
        data_dir="data",
        solver_name="gurobi",  # Using Gurobi
        verbose=True,
    )

    # Run optimization
    result = optimizer.run(
        max_iterations=3,
        step_a_time_limit=60,  # 60s for testing; use 300+ for production
        step_b_iterations=50,
        step_b_num_changes=5,  # Change 5 base camps per iteration (default)
        convergence_tol=0.01,
    )

    # Save results
    optimizer.save_solution("best_solution.pkl")

    # Print summary
    if result["success"]:
        print("\n✓ Optimization successful!")
        print(f"  Best objective: {result['best_objective']:.2f}")
    else:
        print("⚠ Optimization failed or produced no solution")

    return result


if __name__ == "__main__":
    result = main()
