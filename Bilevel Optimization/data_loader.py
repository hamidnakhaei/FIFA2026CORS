"""
Data loader for FIFA 2026 Bilevel Optimization
Loads and manages all CSV data files
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """Loads and manages all input data for the optimization model."""
    
    def __init__(self, data_dir):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to data directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        
        # DataFrames
        self.matches = None
        self.teams = None
        self.venues = None
        self.base_camps = None
        self.weather = None
        self.broadcast_markets = None
        
        # Index structures
        self.team_by_id = {}
        self.teams_by_group = {}
        self.eligible_camps_by_team = {}
        self.camp_by_id = {}
        self.venue_by_id = {}
        
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all CSV files."""
        print("\nLoading data files...")
        
        # Load main data files
        self.matches = pd.read_csv(self.data_dir / "matches.csv")
        self.teams = pd.read_csv(self.data_dir / "teams.csv")
        self.venues = pd.read_csv(self.data_dir / "venues.csv")
        self.base_camps = pd.read_csv(self.data_dir / "base_camps.csv")
        
        # Load optional data files if they exist
        weather_file = self.data_dir / "weather.csv"
        if weather_file.exists():
            self.weather = pd.read_csv(weather_file)
        
        broadcast_file = self.data_dir / "broadcast_markets.csv"
        if broadcast_file.exists():
            self.broadcast_markets = pd.read_csv(broadcast_file)
        
        # Build index structures
        self._build_indexes()
    
    def _build_indexes(self):
        """Build lookup structures for efficient data access."""
        # Team index
        for _, row in self.teams.iterrows():
            team_id = row['team_id']
            self.team_by_id[team_id] = row
            
            # Group index
            group = row['group']
            if group not in self.teams_by_group:
                self.teams_by_group[group] = []
            self.teams_by_group[group].append(team_id)
        
        # Base camp index
        for _, row in self.base_camps.iterrows():
            camp_id = row['base_camp_id']
            self.camp_by_id[camp_id] = row
            
            # Eligible camps per team
            team_id = row['team_id']
            if pd.notna(team_id):  # Only if team is assigned
                if team_id not in self.eligible_camps_by_team:
                    self.eligible_camps_by_team[team_id] = []
                self.eligible_camps_by_team[team_id].append(camp_id)
        
        # All teams can choose from all camps
        for team_id in self.team_by_id.keys():        
            self.eligible_camps_by_team[team_id] = list(self.camp_by_id.keys())
        
        # Venue index
        for _, row in self.venues.iterrows():
            venue_id = row['venue_id']
            self.venue_by_id[venue_id] = row
    
    def get_camp_info(self, camp_id):
        """
        Get information about a base camp.
        
        Args:
            camp_id: Base camp ID
        
        Returns:
            Series containing camp information
        """
        if camp_id in self.camp_by_id:
            return self.camp_by_id[camp_id]
        return None
    
    def get_venue_info(self, venue_id):
        """
        Get information about a venue.
        
        Args:
            venue_id: Venue ID
        
        Returns:
            Series containing venue information
        """
        if venue_id in self.venue_by_id:
            return self.venue_by_id[venue_id]
        return None
    
    def get_team_info(self, team_id):
        """
        Get information about a team.
        
        Args:
            team_id: Team ID
        
        Returns:
            Series containing team information
        """
        if team_id in self.team_by_id:
            return self.team_by_id[team_id]
        return None
    
    def get_camps_in_country(self, country):
        """
        Get all base camps in a specific country.
        
        Args:
            country: Country name (e.g., 'USA', 'MEX', 'CAN')
        
        Returns:
            List of base camp IDs in the specified country
        """
        camps = self.base_camps[self.base_camps['country'] == country]['base_camp_id'].tolist()
        return camps
    
    def get_team_matches(self, team_id):
        """
        Get all matches for a specific team.
        
        Args:
            team_id: Team ID
        
        Returns:
            List of match dictionaries where the team plays (as either team_a or team_b)
        """
        team_matches = self.matches[
            (self.matches['team_a_id'] == team_id) | (self.matches['team_b_id'] == team_id)
        ]
        return [row.to_dict() for _, row in team_matches.iterrows()]
    
    def summary(self):
        """Print a summary of loaded data."""
        print("\n" + "="*70)
        print("DATA SUMMARY")
        print("="*70)
        print(f"Matches: {len(self.matches)}")
        print(f"Teams: {len(self.teams)}")
        print(f"Venues: {len(self.venues)}")
        print(f"Base camps: {len(self.base_camps)}")
        
        if self.weather is not None:
            print(f"Weather records: {len(self.weather)}")
        
        if self.broadcast_markets is not None:
            print(f"Broadcast markets: {len(self.broadcast_markets)}")
        
        print(f"\nGroups: {sorted(self.teams_by_group.keys())}")
        
        for group in sorted(self.teams_by_group.keys()):
            teams = self.teams_by_group[group]
            print(f"  Group {group}: {len(teams)} teams")
        
        print("="*70 + "\n")


def load_data(data_dir):
    """
    Load all data files from a directory.
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        DataLoader instance with all data loaded
    """
    return DataLoader(data_dir)
