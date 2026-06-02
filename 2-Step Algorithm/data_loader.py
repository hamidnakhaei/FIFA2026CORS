"""
Data loader and preprocessing module for FIFA 2026 Group-Stage Optimization.
Loads and processes data from CSV files in the data/ folder.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Set
from config import config_params


class DataLoader:
    """Load and preprocess FIFA 2026 data."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._data = {}
        self.config_params = config_params()  # Load parameters from config

    def load_all(self) -> Dict:
        """Load all CSV files and return as dictionary."""
        self._data = {
            "matches": self._load_csv("matches.csv"),
            "venues": self._load_csv("venues.csv"),
            "teams": self._load_csv("teams.csv"),
            "base_camps": self._load_csv("base_camps.csv"),
            "weather": self._load_csv("weather.csv"),
            "broadcast_markets": self._load_csv("broadcast_markets.csv"),
        }
        return self._data

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load a single CSV file."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_csv(path)

    def get_matches(self) -> pd.DataFrame:
        """Get matches data with processed indices."""
        if "matches" not in self._data:
            self.load_all()
        matches = self._data["matches"].copy()
        matches["datetime"] = pd.to_datetime(
            matches["date"] + " " + matches["kickoff_local"]
        )
        return matches

    def get_venues(self) -> pd.DataFrame:
        """Get venues data."""
        if "venues" not in self._data:
            self.load_all()
        return self._data["venues"].copy()

    def get_teams(self) -> pd.DataFrame:
        """Get teams data."""
        if "teams" not in self._data:
            self.load_all()
        return self._data["teams"].copy()

    def get_base_camps(self) -> pd.DataFrame:
        """Get base camps data."""
        if "base_camps" not in self._data:
            self.load_all()
        return self._data["base_camps"].copy()

    def get_weather(self) -> pd.DataFrame:
        """Get weather data."""
        if "weather" not in self._data:
            self.load_all()
        weather = self._data["weather"].copy()
        weather["datetime"] = pd.to_datetime(weather["datetime"])
        return weather

    def get_broadcast_markets(self) -> pd.DataFrame:
        """Get broadcast markets data."""
        if "broadcast_markets" not in self._data:
            self.load_all()
        broadcast = self._data["broadcast_markets"].copy()
        broadcast["primetime_start_local"] = pd.to_datetime(
            broadcast["primetime_start_local"], format="%H:%M"
        ).dt.time
        broadcast["primetime_end_local"] = pd.to_datetime(
            broadcast["primetime_end_local"], format="%H:%M"
        ).dt.time
        return broadcast

    def get_sets_and_indices(
        self,
    ) -> Tuple[
        Set[int], Set[int], Set[str], Set[str], Set[str], Dict, Dict, Dict, Dict
    ]:
        """
        Extract sets and indices for the optimization model.

        Returns:
            Tuple containing:
            - M: set of match IDs
            - T: set of time slot IDs
            - S: set of stadium IDs
            - I: set of team IDs
            - G: set of group IDs
            - M_i: dict mapping team_id to its 3 matches
            - M_g: dict mapping group to its 6 matches
            - T_r: dict mapping round to time slots
            - S_c: dict mapping country to stadiums
        """
        matches = self.get_matches()
        venues = self.get_venues()
        teams = self.get_teams()

        M = set(matches["match_id"].unique())
        S = set(venues["venue_id"].unique())
        I = set(teams["team_id"].unique())
        G = set(teams["group"].unique())
        T = set(range(len(matches)))  # Each match has a potential time slot

        # Map teams to their matches
        M_i = {}
        for team_id in I:
            team_matches = matches[
                (matches["team_a_id"] == team_id) | (matches["team_b_id"] == team_id)
            ]["match_id"].tolist()
            M_i[team_id] = set(team_matches)

        # Map groups to their matches
        M_g = {}
        for group in G:
            group_matches = matches[matches["group"] == group]["match_id"].tolist()
            M_g[group] = set(group_matches)

        # Map rounds to time slots (simplified: assume 3 rounds with 24 matches each)
        matches_per_round = self.config_params.MATCH_COUNT // 3  # 72 / 3 = 24
        T_r = {
            1: set(range(0, matches_per_round)),
            2: set(range(matches_per_round, 2 * matches_per_round)),
            3: set(range(2 * matches_per_round, self.config_params.MATCH_COUNT))
        }

        # Map countries to stadiums
        S_c = {}
        for country in ["USA", "MEX", "CAN"]:
            country_stadiums = venues[venues["country"] == country]["venue_id"].tolist()
            S_c[country] = set(country_stadiums)

        return M, T, S, I, G, M_i, M_g, T_r, S_c

    def get_parameters(self) -> Dict:
        """
        Extract parameters for the optimization model.

        Returns:
            Dictionary of parameters including distances, time zones, elevations, etc.
        """
        from geopy.distance import geodesic

        venues = self.get_venues()
        base_camps = self.get_base_camps()
        teams = self.get_teams()
        matches = self.get_matches()
        weather = self.get_weather()
        broadcast = self.get_broadcast_markets()

        # Distance matrix: base_camp to venue
        dist = {}
        for _, camp in base_camps.iterrows():
            camp_loc = (camp["lat"], camp["lon"])
            for _, venue in venues.iterrows():
                venue_loc = (venue["lat"], venue["lon"])
                distance_km = geodesic(camp_loc, venue_loc).kilometers
                dist[(camp["base_camp_id"], venue["venue_id"])] = distance_km

        # Time zone offsets
        tzone_stadium = dict(zip(venues["venue_id"], venues["utc_offset_june"]))
        tzone_basecamp = dict(
            zip(base_camps["base_camp_id"], base_camps["utc_offset_june"])
        )

        # Elevations
        elev_stadium = dict(zip(venues["venue_id"], venues.get("elevation", [0] * len(venues))))
        elev_basecamp = dict(
            zip(base_camps["base_camp_id"], base_camps.get("elevation", [0] * len(base_camps)))
        )

        # Stadium clusters
        cluster = dict(zip(venues["venue_id"], venues["zone"]))

        # Team FIFA ratings
        team_rating = dict(zip(teams["team_id"], teams["fifa_ranking"]))

        # Match values based on team ratings
        match_value = {}
        for _, match in matches.iterrows():
            team_a_rating = team_rating.get(match["team_a_id"], 100)
            team_b_rating = team_rating.get(match["team_b_id"], 100)
            mu = (1 / team_a_rating + 1 / team_b_rating) / 2
            match_value[match["match_id"]] = mu

        # Broadcast quality by time slot
        broadcast_quality = {}
        for i, _ in matches.iterrows():
            broadcast_quality[i] = 0.5  # Simplified; would integrate prime-time scoring

        return {
            "dist": dist,
            "tzone_stadium": tzone_stadium,
            "tzone_basecamp": tzone_basecamp,
            "elev_stadium": elev_stadium,
            "elev_basecamp": elev_basecamp,
            "cluster": cluster,
            "team_rating": team_rating,
            "match_value": match_value,
            "broadcast_quality": broadcast_quality,
            "weather": weather,
            "broadcast_markets": broadcast,
            "N_c": self.config_params.COUNTRY_MATCH_ALLOCATION,
            "R_min": self.config_params.MIN_REST_HOURS,
            "match_duration": self.config_params.MATCH_DURATION_HOURS,
            "us_visa_ban_teams": self.config_params.US_VISA_BAN_TEAMS,
            "us_visa_bond_teams": self.config_params.US_VISA_BOND_TEAMS,
            "weights": self.config_params.KPI_WEIGHTS,
        }


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    data = loader.load_all()

    print("✓ Matches loaded:", len(data["matches"]))
    print("✓ Venues loaded:", len(data["venues"]))
    print("✓ Teams loaded:", len(data["teams"]))
    print("✓ Base camps loaded:", len(data["base_camps"]))
    print("✓ Weather records loaded:", len(data["weather"]))
    print("✓ Broadcast markets loaded:", len(data["broadcast_markets"]))

    M, T, S, I, G, M_i, M_g, T_r, S_c = loader.get_sets_and_indices()
    print(f"\nSets: |M|={len(M)}, |T|={len(T)}, |S|={len(S)}, |I|={len(I)}, |G|={len(G)}")
    print(f"Countries: USA={len(S_c['USA'])}, MEX={len(S_c['MEX'])}, CAN={len(S_c['CAN'])}")

    params = loader.get_parameters()
    print(f"\n✓ Parameters computed: {len(params)} parameter sets")
