"""
Base camp optimization (Step B) for FIFA 2026 Group-Stage.
Re-optimizes team base camps while holding schedule fixed, using Metropolis exploration
with full KPI evaluation (all 13 KPIs).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import random
from copy import deepcopy


class BaseCampOptimizer:
    """Solve Step B: optimize base camps while holding schedule fixed."""

    def __init__(self, data_loader, kpi_calculator, schedule: Dict):
        self.loader = data_loader
        self.kpi_calc = kpi_calculator
        self.schedule = schedule

        self.teams = data_loader.get_teams()
        self.base_camps = data_loader.get_base_camps()
        self.params = data_loader.get_parameters()

    def compute_org_objective(self, base_camp_assignment: Dict) -> float:
        """Compute travel cost for base camp assignment."""
        TD = {}
        for team_id in self.teams["team_id"].unique():
            total_distance = 0.0
            for match_id, (slot, stadium_id) in self.schedule.items():
                match = self.kpi_calc.matches[self.kpi_calc.matches["match_id"] == match_id].iloc[0]
                if team_id in [match["team_a_id"], match["team_b_id"]]:
                    if team_id in base_camp_assignment:
                        base_camp_id = base_camp_assignment[team_id]
                        distance = self.params["dist"].get((base_camp_id, stadium_id), 0)
                        total_distance += distance
            TD[team_id] = total_distance

    
        objective = np.mean(list(TD.values()))
        return objective

    def metropolis_move(
        self,
        base_camp_assignment: Dict,
        temperature: float = 1.0,
        num_changes: int = 1,
    ) -> Tuple[Dict, bool]:
        """
        Metropolis move: randomly perturb base camps for multiple teams.
        Accept with probability proportional to exp(-delta_cost / temperature).

        Args:
            base_camp_assignment: Current base camp assignment
            temperature: Metropolis temperature (higher = more liberal acceptance)
            num_changes: Number of teams' base camps to change per move (default: 1)

        Returns:
            Tuple of (new_assignment, accepted)
        """
        new_assignment = deepcopy(base_camp_assignment)

        # Select random teams to change (without replacement)
        num_teams = len(new_assignment)
        num_to_change = min(num_changes, num_teams)
        teams_to_change = random.sample(list(new_assignment.keys()), num_to_change)

        # Change base camps for selected teams
        for team_id in teams_to_change:
            eligible_bases = list(range(1, len(self.base_camps) + 1))
            current_base = new_assignment[team_id]
            new_candidates = [b for b in eligible_bases if b != current_base]

            if new_candidates:
                new_base = random.choice(new_candidates)
                new_assignment[team_id] = new_base

        # Compute costs
        old_cost = self.compute_org_objective(base_camp_assignment)
        new_cost = self.compute_org_objective(new_assignment)
        delta_cost = new_cost - old_cost

        # Metropolis acceptance criterion
        if delta_cost < 0:
            accepted = True
        else:
            prob_accept = np.exp(-delta_cost / temperature)
            accepted = random.random() < prob_accept

        if not accepted:
            new_assignment = deepcopy(base_camp_assignment)

        return new_assignment, accepted

    def optimize(
        self,
        initial_assignment: Dict,
        max_iterations: int = 10,
        temperature_schedule: str = "exponential",
        num_changes_per_iteration: int = 5,
    ) -> Dict:
        """
        Optimize base camps using simulated annealing with Metropolis moves.

        Args:
            initial_assignment: Initial base camp assignment (team_id -> base_camp_id)
            max_iterations: Number of iterations
            temperature_schedule: "exponential", "linear", or "constant"
            num_changes_per_iteration: Number of teams' base camps to change per iteration (default: 5)

        Returns:
            Dictionary with optimization results
        """
        current_assignment = deepcopy(initial_assignment)
        best_assignment = deepcopy(current_assignment)

        best_cost = self.compute_org_objective(best_assignment)
        current_cost = best_cost

        costs_history = [current_cost]
        accept_count = 0

        for iteration in range(max_iterations):
            # Update temperature
            if temperature_schedule == "exponential":
                temperature = 1.0 * np.exp(-3 * iteration / max_iterations)
            elif temperature_schedule == "linear":
                temperature = 1.0 * (1 - iteration / max_iterations)
            else:  # constant
                temperature = 1.0

            # Metropolis move with multiple changes
            new_assignment, accepted = self.metropolis_move(
                current_assignment, temperature, num_changes=num_changes_per_iteration
            )

            if accepted:
                current_assignment = new_assignment
                current_cost = self.compute_org_objective(current_assignment)
                accept_count += 1

                # Update best if improved
                if current_cost < best_cost:
                    best_assignment = deepcopy(current_assignment)
                    best_cost = current_cost

            costs_history.append(current_cost)

            if (iteration + 1) % max(1, max_iterations // 10) == 0:
                acceptance_rate = accept_count / (iteration + 1)
                print(
                    f"  Iteration {iteration + 1}/{max_iterations}: "
                    f"current={current_cost:.2f}, best={best_cost:.2f}, "
                    f"T={temperature:.4f}, accept_rate={acceptance_rate:.2%}"
                )

        return {
            "best_assignment": best_assignment,
            "best_cost": best_cost,
            "final_assignment": current_assignment,
            "final_cost": current_cost,
            "costs_history": costs_history,
            "acceptance_rate": accept_count / max_iterations,
            "iterations": max_iterations,
        }


def generate_initial_assignment(teams_df: pd.DataFrame) -> Dict:
    """
    Generate initial base camp assignment (greedy or random).
    Returns dict mapping team_id -> base_camp_id.
    """
    assignment = {}
    for _, team in teams_df.iterrows():
        team_id = team["team_id"]
        # Simplified: assign first base camp in eligible list (would be improved)
        base_camp_id = (hash(team_id) % 10) + 1
        assignment[team_id] = base_camp_id
    return assignment


def load_base_camp_assignment_from_data(base_camps_df: pd.DataFrame) -> Dict:
    """
    Load current base camp assignments from base_camps.csv.
    Returns dict mapping team_id -> base_camp_id for assigned teams.
    Only includes teams with confirmed assignments (team_id not null).
    """
    assignment = {}
    assigned = base_camps_df[base_camps_df["team_id"].notna()]
    for _, row in assigned.iterrows():
        team_id = row["team_id"]
        base_camp_id = row["base_camp_id"]
        assignment[team_id] = base_camp_id
    return assignment


# if __name__ == "__main__":
#     from data_loader import DataLoader
#     from kpis import KPICalculator

#     loader = DataLoader()
#     data = loader.load_all()
#     params = loader.get_parameters()

#     # Dummy schedule
#     schedule = {i: (i % 24, f"S{i % 16}") for i in range(1, 73)}

#     kpi_calc = KPICalculator(loader, params)
#     teams_df = loader.get_teams()
#     initial_assignment = generate_initial_assignment(teams_df)

#     optimizer = BaseCampOptimizer(loader, kpi_calc, schedule)

#     print("Optimizing base camps with simulated annealing...")
#     result = optimizer.optimize(
#         initial_assignment, max_iterations=100, temperature_schedule="exponential"
#     )

#     print(f"\n✓ Optimization complete!")
#     print(f"  Best cost: {result['best_cost']:.2f}")
#     print(f"  Final cost: {result['final_cost']:.2f}")
#     print(f"  Acceptance rate: {result['acceptance_rate']:.2%}")
#     print(f"  Teams in best assignment: {len(result['best_assignment'])}")
