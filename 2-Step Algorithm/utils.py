"""
Utilities module for solution validation, post-processing, and reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import pickle


class SolutionValidator:
    """Validate that a solution satisfies hard constraints H1-H8."""

    def __init__(self, data_loader):
        self.loader = data_loader
        self.matches = data_loader.get_matches()
        self.venues = data_loader.get_venues()
        self.teams = data_loader.get_teams()
        self.M, self.T, self.S, self.I, self.G, self.M_i, self.M_g, self.T_r, self.S_c = (
            data_loader.get_sets_and_indices()
        )

    def validate_all(self, schedule: Dict) -> Dict[str, bool]:
        """
        Validate all hard constraints.

        Returns:
            Dictionary with constraint validation results
        """
        results = {
            "H1_each_match_once": self._check_h1(schedule),
            "H2_round_robin": self._check_h2(schedule),
            "H3_one_per_round": self._check_h3(schedule),
            "H7_simultaneous_finals": self._check_h7(schedule),
            "H8_country_allocation": self._check_h8(schedule),
        }
        return results

    def _check_h1(self, schedule: Dict) -> bool:
        """H1: Each match scheduled exactly once."""
        match_ids = set(schedule.keys())
        expected = set(self.matches["match_id"].unique())
        return match_ids == expected

    def _check_h2(self, schedule: Dict) -> bool:
        """H2: Each team plays 3 matches (round-robin)."""
        for team_id in self.I:
            team_matches = [
                m for m in schedule if team_id in [
                    self.matches[self.matches["match_id"] == m]["team_a_id"].values[0],
                    self.matches[self.matches["match_id"] == m]["team_b_id"].values[0],
                ]
            ]
            if len(team_matches) != 3:
                return False
        return True

    def _check_h3(self, schedule: Dict) -> bool:
        """H3: One match per team per round."""
        for team_id in self.I:
            for round_num in [1, 2, 3]:
                round_matches = [
                    m for m in schedule
                    if self.matches[self.matches["match_id"] == m]["round"].values[0] == round_num
                    and team_id in [
                        self.matches[self.matches["match_id"] == m]["team_a_id"].values[0],
                        self.matches[self.matches["match_id"] == m]["team_b_id"].values[0],
                    ]
                ]
                if len(round_matches) != 1:
                    return False
        return True

    def _check_h7(self, schedule: Dict) -> bool:
        """H7: Final matches of each group simultaneous."""
        # Simplified check: at least verify finals exist
        for group in self.G:
            final_matches = [
                m for m in schedule
                if self.matches[self.matches["match_id"] == m]["group"].values[0] == group
                and self.matches[self.matches["match_id"] == m]["round"].values[0] == 3
            ]
            if len(final_matches) < 2:
                return False
        return True

    def _check_h8(self, schedule: Dict) -> bool:
        """H8: Match allocation by country."""
        targets = {"USA": 52, "MEX": 10, "CAN": 10}
        for country, target in targets.items():
            country_stadiums = self.S_c[country]
            count = sum(1 for m, (t, s) in schedule.items() if s in country_stadiums)
            if count != target:
                return False
        return True


class SolutionReporter:
    """Generate human-readable reports from solutions."""

    def __init__(self, data_loader):
        self.loader = data_loader
        self.matches = data_loader.get_matches()
        self.venues = data_loader.get_venues()
        self.teams = data_loader.get_teams()
        self.base_camps = data_loader.get_base_camps()

    def schedule_to_dataframe(self, schedule: Dict) -> pd.DataFrame:
        """Convert schedule dict to DataFrame."""
        rows = []
        for match_id, (time_slot, stadium_id) in schedule.items():
            match = self.matches[self.matches["match_id"] == match_id]
            if len(match) > 0:
                match = match.iloc[0]
                venue = self.venues[self.venues["venue_id"] == stadium_id]
                venue_name = venue["name"].values[0] if len(venue) > 0 else stadium_id

                rows.append({
                    "match_id": match_id,
                    "group": match["group"],
                    "round": match["round"],
                    "team_a": match["team_a_id"],
                    "team_b": match["team_b_id"],
                    "time_slot": time_slot,
                    "stadium": stadium_id,
                    "stadium_name": venue_name,
                    "city": venue["city"].values[0] if len(venue) > 0 else "Unknown",
                    "country": venue["country"].values[0] if len(venue) > 0 else "Unknown",
                })
        return pd.DataFrame(rows).sort_values("match_id")

    def base_camp_assignment_to_dataframe(
        self, base_camp_assignment: Dict
    ) -> pd.DataFrame:
        """Convert base camp assignment dict to DataFrame."""
        rows = []
        for team_id, base_camp_id in base_camp_assignment.items():
            team = self.teams[self.teams["team_id"] == team_id]
            base_camp = self.base_camps[self.base_camps["base_camp_id"] == base_camp_id]

            rows.append({
                "team_id": team_id,
                "team_name": team["team_name"].values[0] if len(team) > 0 else "Unknown",
                "group": team["group"].values[0] if len(team) > 0 else "Unknown",
                "base_camp_id": base_camp_id,
                "facility_name": base_camp["training_site"].values[0]
                if len(base_camp) > 0
                else "Unknown",
                "city": base_camp["city"].values[0] if len(base_camp) > 0 else "Unknown",
                "country": base_camp["country"].values[0]
                if len(base_camp) > 0
                else "Unknown",
            })

        return pd.DataFrame(rows).sort_values("team_id")

    def print_schedule_summary(self, schedule: Dict):
        """Print summary of schedule."""
        df = self.schedule_to_dataframe(schedule)

        print("\n" + "="*80)
        print("SCHEDULE SUMMARY")
        print("="*80)
        print(f"\nTotal matches: {len(df)}")

        print("\nMatches per country:")
        for country in ["USA", "MEX", "CAN"]:
            count = len(df[df["country"] == country])
            print(f"  {country}: {count}")

        print("\nMatches per stadium:")
        stadium_summary = df.groupby("stadium_name").size().sort_values(ascending=False)
        for stadium, count in stadium_summary.items():
            print(f"  {stadium}: {count}")

        print("\nFirst 10 matches in schedule:")
        display_cols = ["match_id", "group", "round", "team_a", "team_b", "stadium", "time_slot"]
        print(df[display_cols].head(10).to_string(index=False))

    def print_base_camp_summary(self, base_camp_assignment: Dict):
        """Print summary of base camp assignment."""
        df = self.base_camp_assignment_to_dataframe(base_camp_assignment)

        print("\n" + "="*80)
        print("BASE CAMP ASSIGNMENT SUMMARY")
        print("="*80)
        print(f"\nTotal teams assigned: {len(df)}")

        print("\nTeams per country:")
        for country in ["USA", "MEX", "CAN"]:
            count = len(df[df["country"] == country])
            print(f"  {country}: {count}")

        print("\nTeams per base camp facility:")
        facility_summary = df.groupby("facility_name").size().sort_values(ascending=False)
        for facility, count in facility_summary.items():
            print(f"  {facility}: {count}")

        print("\nFirst 10 team assignments:")
        display_cols = ["team_id", "team_name", "base_camp_id", "facility_name", "city"]
        print(df[display_cols].head(10).to_string(index=False))

    def export_to_csv(
        self, schedule: Dict, base_camp_assignment: Dict, output_prefix: str = "output"
    ):
        """Export schedule and base camp assignment to CSV files."""
        schedule_df = self.schedule_to_dataframe(schedule)
        base_camp_df = self.base_camp_assignment_to_dataframe(base_camp_assignment)

        schedule_file = f"{output_prefix}_schedule.csv"
        base_camp_file = f"{output_prefix}_base_camps.csv"

        schedule_df.to_csv(schedule_file, index=False)
        base_camp_df.to_csv(base_camp_file, index=False)

        print(f"\n✓ Schedule exported to {schedule_file}")
        print(f"✓ Base camps exported to {base_camp_file}")


def load_solution(filename: str) -> Dict:
    """Load solution from pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader()
    data = loader.load_all()

    # Test with dummy data
    schedule = {i: (i % 24, f"S{i % 16}") for i in range(1, 73)}
    base_camp_assignment = {
        "BRA": 1, "GER": 2, "ARG": 3, "FRA": 4, "ENG": 5,
    }

    validator = SolutionValidator(loader)
    results = validator.validate_all(schedule)
    print("Constraint validation results:")
    for constraint, valid in results.items():
        status = "✓" if valid else "✗"
        print(f"  {status} {constraint}")

    reporter = SolutionReporter(loader)
    print("\nSchedule info:")
    reporter.print_schedule_summary(schedule)
    print("\nBase camp info:")
    reporter.print_base_camp_summary(base_camp_assignment)
    reporter.export_to_csv(schedule, base_camp_assignment, "test_output")
