"""
Main algorithm orchestrator for FIFA 2026 Two-Step Iterative Optimization.
Implements the alternating minimization algorithm with stochastic exploration.
"""

import pandas as pd
from typing import Dict
from data_loader import DataLoader
from schedule_optimizer import ScheduleOptimizer
from base_camp_optimizer import BaseCampOptimizer


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

        # Initialize
        self.best_objective = float("inf")
        self.best_solution = None
        self.iteration_history = []

    def run_step_a(self, time_limit: int = 600) -> Dict:
        """
        Step A: Optimize schedule
        Args:
            time_limit: Solver time limit in seconds

        Returns:
            Dictionary with schedule and full KPI objective value
        """

        # Solve schedule optimization
        optimizer = ScheduleOptimizer(self.loader)
        optimizer.build_model()
        result = optimizer.solve(time_limit=time_limit, solver_name=self.solver_name)
        schedule = result["schedule"]
        objective_full = result["objective"]

        print(f"  ✓ Schedule optimization completed")
        print(f"  ✓ Solver status: {result['status']}")
        return {"schedule": schedule, "objective_full": objective_full}
    
    def run_step_b(
        self,
        schedule: Dict) -> Dict:
        """
        Step B: Optimize base camps while holding schedule fixed.
        Args:
            schedule: Fixed schedule from Step A
        Returns:
            Dictionary with optimized base camp assignment and objective
        """
        optimizer = BaseCampOptimizer(self.loader, schedule)
        result = optimizer.optimize()
        print(f"  ✓ Base camp optimization completed") 
        print(f"  ✓ Solver status: {result['status']}")
        return result["assignment"]

    def export_results(
        self,
        schedule: Dict,
        base_camp_assignment: Dict,
        fifa_base_camp_opt: Dict,
        output_dir: str = ".") -> Dict:
        """
        Export optimization results in the same format as imported files.

        Step A (Schedule): Exported as matches.csv with columns
                          [match_id, group, round, team_a_id, team_b_id, 
                           venue_id, date, kickoff_local]

        Step B (Base Camps): Exported as base_camps.csv with columns
                             [base_camp_id, team_id, training_site, city, 
                              country, lat, lon, utc_offset_june]

        Args:
            schedule: Schedule dict from Step A
            base_camp_assignment: Base camp assignment dict (team_id -> base_camp_id)
            fifa_schedule: FIFA schedule dict
            output_dir: Output directory for CSV files

        Returns:
            Dictionary with paths to exported files
        """
        # Get original data for reference
        matches_original = self.loader.get_matches()
        base_camps_original = self.loader.get_base_camps()

        # Export Step A: Schedule results in matches.csv format
        schedule_results = []
        for match_id, venue_id_assigned in schedule.items():
            # Find original match data
            match_row = matches_original[matches_original['match_id'] == match_id].iloc[0]
            
            schedule_results.append({
                'match_id': match_id,
                'group': match_row['group'],
                'team_a_id': match_row['team_a_id'],
                'team_b_id': match_row['team_b_id'],
                'venue_id': venue_id_assigned[-1],
                'date': venue_id_assigned[0][0],
                'kickoff_local': venue_id_assigned[0][1]
            })
        
        schedule_export_df = pd.DataFrame(schedule_results)
        schedule_export_path = f"{output_dir}/schedule_results.csv"
        schedule_export_df.to_csv(schedule_export_path, index=False)
        print(f"✓ Schedule exported to {schedule_export_path}")

        # Export Step B: Base camp assignment in base_camps.csv format
        base_camp_results = []
        for team_id, base_camp_id in base_camp_assignment.items():
            # Find base camp details from original data
            base_camp_row = base_camps_original[
                base_camps_original['base_camp_id'] == base_camp_id
            ].iloc[0]
            
            base_camp_results.append({
                'base_camp_id': base_camp_id,
                'team_id': team_id,
                'training_site': base_camp_row['training_site'],
                'city': base_camp_row['city'],
                'country': base_camp_row['country'],
                'lat': base_camp_row['lat'],
                'lon': base_camp_row['lon'],
                'utc_offset_june': base_camp_row['utc_offset_june']
            })
        
        base_camp_export_df = pd.DataFrame(base_camp_results)
        base_camp_export_path = f"{output_dir}/base_camp_results.csv"
        base_camp_export_df.to_csv(base_camp_export_path, index=False)
        print(f"✓ Base camps exported to {base_camp_export_path}")


        # Export Step B: Base camp assignment in base_camps.csv format (FIFA)
        fifa_base_camp_results = []
        for team_id, base_camp_id in fifa_base_camp_opt.items():
            # Find base camp details from original data
            base_camp_row = base_camps_original[
                base_camps_original['base_camp_id'] == base_camp_id
            ].iloc[0]
            
            fifa_base_camp_results.append({
                'base_camp_id': base_camp_id,
                'team_id': team_id,
                'training_site': base_camp_row['training_site'],
                'city': base_camp_row['city'],
                'country': base_camp_row['country'],
                'lat': base_camp_row['lat'],
                'lon': base_camp_row['lon'],
                'utc_offset_june': base_camp_row['utc_offset_june']
            })
        
        fifa_base_camp_export_df = pd.DataFrame(fifa_base_camp_results)
        fifa_base_camp_export_path = f"{output_dir}/fifa_base_camp_results.csv"
        fifa_base_camp_export_df.to_csv(fifa_base_camp_export_path, index=False)
        print(f"✓ FIFA Base camps exported to {fifa_base_camp_export_path}")

        return {
            'schedule_path': schedule_export_path,
            'base_camp_path': base_camp_export_path,
            'base_camp_df': base_camp_export_df,
            'fifa_base_camp_df': fifa_base_camp_export_df
        }

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
        
        # Step B: Optimize base camps
        print(f"\n---- Step B: Optimize Base Camps ---")
        new_base_camp = self.run_step_b(new_schedule)

        # run step b with FIFA schedule
        fifa_schedule_csv = self.loader.get_matches()  
        fifa_schedule = {
            row['match_id']: ((row['date'], row['kickoff_local']), row['venue_id'])
            for _, row in fifa_schedule_csv.iterrows()
        }
        fifa_base_camp_opt = self.run_step_b(fifa_schedule)
        
        
        return {
            'schedule': new_schedule,
            'base_camp_assignment': new_base_camp,
            'fifa_base_camp_opt': fifa_base_camp_opt
        }


def main():
    """Main entry point."""
    # Initialize optimizer
    optimizer = TwoStepOptimizer(data_dir="data", solver_name="gurobi")
    
    # Run optimization
    result = optimizer.run()
    
    # Export results in proper CSV format
    export_result = optimizer.export_results(
        schedule=result['schedule'],
        base_camp_assignment=result['base_camp_assignment'],
        fifa_base_camp_opt=result['fifa_base_camp_opt'],  # Using FIFA base camp optimization results
        output_dir="2-Step Algorithm/outputs"
    )

    return result, export_result


if __name__ == "__main__":
    result = main()
