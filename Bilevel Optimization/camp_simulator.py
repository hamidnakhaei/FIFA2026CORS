"""
Deterministic greedy base-camp simulation for FIFA 2026 Opt+Sim framework
Implements sequential assignment of teams to base camps
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class CampSimulator:
    """
    Simulates base-camp selection using deterministic greedy assignment.
    
    Teams are assigned in priority order (seeding/qualification). Each team
    selects its travel-minimizing camp from those still available.
    Exclusivity is enforced by removing selected camps from availability.
    """
    
    def __init__(self, data_loader, parameters, team_priority_order=None):
        """
        Initialize the simulator.
        
        Args:
            data_loader: DataLoader instance with all data
            parameters: ParameterBuilder with precomputed constants
            team_priority_order: List of team_ids in selection order.
                                If None, uses default ordering (alphabetical by team_id)
        """
        self.data = data_loader
        self.params = parameters
        
        # Create index mappings from DataFrames
        match_ids = data_loader.matches['match_id'].tolist()
        venue_ids = data_loader.venues['venue_id'].tolist()
        self.match_id_to_idx = {mid: i for i, mid in enumerate(match_ids)}
        self.venue_id_to_idx = {vid: i for i, vid in enumerate(venue_ids)}
        self.venue_idx_to_id = {i: vid for i, vid in enumerate(venue_ids)}
        
        # Team priority order for sequential assignment
        all_teams = sorted(data_loader.team_by_id.keys())
        self.team_priority = team_priority_order if team_priority_order else all_teams
        
        # Eligible camps per team (filtered by hard constraints like US bans)
        self.eligible_camps_by_team = self._build_eligible_camps()
    
    def _build_eligible_camps(self) -> Dict[str, List[str]]:
        """
        Build eligible camp list for each team, enforcing hard constraints
        like US entry bans.
        
        Returns:
            Dict mapping team_id -> list of eligible camp_ids
        """
        eligible = {}
        
        # Get US base camps from the data
        us_camps = self.data.base_camps[self.data.base_camps['country'] == 'USA']['base_camp_id'].tolist()
        
        # Check for US ban list (if exists in parameters)
        us_ban_teams = self.params.get('us_ban_teams', []) if isinstance(self.params, dict) else []
        
        for team_id in self.data.team_by_id.keys():
            # Start with all camps eligible for this team
            camps = list(self.data.eligible_camps_by_team.get(team_id, []))
            
            # Remove US camps if team is under full US entry ban
            if team_id in us_ban_teams:
                camps = [c for c in camps if c not in us_camps]
            
            eligible[team_id] = camps
        
        return eligible
    
    def simulate(self, venue_assignment: np.ndarray) -> Tuple[Dict[str, str], Dict]:
        """
        Run greedy camp selection simulation for a given schedule.
        
        Args:
            venue_assignment: Array a_ms indexed by [match_idx, venue_idx].
                            Used to compute each team's three match venues.
        
        Returns:
            Tuple of:
            - camp_assignment: Dict {team_id -> selected camp_id}
            - kpi_values: Dict with realized base-camp-dependent KPIs:
                         {'travel': {...}, 'jet_lag': {...}, 'border_crossings': {...}, 'altitude': {...}}
        """
        # Initialize
        camp_assignment = {}
        available_camps = set(self.data.camp_by_id.keys())  # All camps initially available
        
        # Process teams in priority order
        for team_id in self.team_priority:
            # Get eligible camps that are still available
            eligible = self.eligible_camps_by_team[team_id]
            available_eligible = [c for c in eligible if c in available_camps]
            
            if not available_eligible:
                raise ValueError(
                    f"Team {team_id} has no available eligible camps. "
                    f"Check data and facility allocation."
                )
            
            # Find best camp by travel cost
            best_camp = self._find_best_camp(team_id, available_eligible, venue_assignment)
            
            # Assign and remove from availability
            camp_assignment[team_id] = best_camp
            available_camps.remove(best_camp)
        
        # Compute realized KPI values at simulated camps
        kpi_values = self._compute_realized_kpis(camp_assignment, venue_assignment)
        
        return camp_assignment, kpi_values
    
    def _find_best_camp(self, team_id: str, eligible_camps: List[str],
                        venue_assignment: np.ndarray) -> str:
        """
        Find the travel-minimizing camp for a team given venue assignment.
        
        Args:
            team_id: Team identifier
            eligible_camps: List of camps available for selection
            venue_assignment: Array a_ms indexed by [match_idx, venue_idx]
        
        Returns:
            Best camp_id
        """
        # Get team's matches
        team_matches = self.data.get_team_matches(team_id)
        
        # Map match_ids to indices and get venues
        team_venues = []
        for match_dict in team_matches:
            match_id = match_dict['match_id']
            # Find which venue(s) this match is assigned to
            match_idx = self.match_id_to_idx.get(match_id)
            if match_idx is not None:
                for venue_idx, assigned in enumerate(venue_assignment[match_idx, :]):
                    if assigned > 0.5:  # Account for floating point
                        venue_id = self.venue_idx_to_id.get(venue_idx)
                        if venue_id:
                            team_venues.append(venue_id)
                        break
        
        best_camp = None
        best_cost = float('inf')
        
        for camp_id in eligible_camps:
            # Compute round-trip travel cost for this camp
            cost = 0.0
            
            # Get distance matrix from parameters
            dist_matrix = self.params.get('dist', {})
            
            for venue_id in team_venues:
                dist = dist_matrix.get(camp_id, {}).get(venue_id, 0)
                cost += 2 * dist  # Round-trip
            
            # Add US visa-bond penalty if applicable (if it exists in params)
            if 'visa_penalties' in self.params:
                visa_penalties = self.params['visa_penalties']
                if team_id in visa_penalties and camp_id in visa_penalties[team_id]:
                    cost += visa_penalties[team_id].get(camp_id, 0)
            
            if cost < best_cost:
                best_cost = cost
                best_camp = camp_id
        
        return best_camp
    
    def _compute_realized_kpis(self, camp_assignment: Dict[str, str],
                                venue_assignment: np.ndarray) -> Dict:
        """
        Compute realized base-camp-dependent KPI values at simulated camps.
        
        Args:
            camp_assignment: Dict {team_id -> camp_id}
            venue_assignment: Array a_ms
        
        Returns:
            Dict with KPI values:
            {
                'travel': {team_id -> cost},
                'jet_lag': {team_id -> penalty},
                'border_crossings': {team_id -> count},
                'altitude': {team_id -> penalty}
            }
        """
        kpis = {
            'travel': {},
            'jet_lag': {},
            'border_crossings': {},
            'altitude': {}
        }
        
        # Get distance matrix
        dist_matrix = self.params.get('dist', {})
        
        for team_id, camp_id in camp_assignment.items():
            # Get team's match venues
            team_matches = self.data.get_team_matches(team_id)
            team_venues = []
            
            for match_dict in team_matches:
                match_id = match_dict['match_id']
                match_idx = self.match_id_to_idx.get(match_id)
                
                if match_idx is not None:
                    # Find venue for this match
                    for venue_idx, assigned in enumerate(venue_assignment[match_idx, :]):
                        if assigned > 0.5:
                            venue_id = self.venue_idx_to_id.get(venue_idx)
                            if venue_id:
                                team_venues.append(venue_id)
                            break
            
            # KPI 1.2: Travel cost (round-trip from camp to each venue)
            travel_cost = 0.0
            for venue_id in team_venues:
                dist = dist_matrix.get(camp_id, {}).get(venue_id, 0)
                travel_cost += 2 * dist
            kpis['travel'][team_id] = travel_cost
            
            # KPI 1.3: Jet-lag penalty (simplified - use precomputed if available)
            jet_lag_penalties = self.params.get('Phi', {})
            jet_lag_cost = 0.0
            for venue_id in team_venues:
                if team_id in jet_lag_penalties:
                    if camp_id in jet_lag_penalties[team_id]:
                        if venue_id in jet_lag_penalties[team_id][camp_id]:
                            # Average across slots
                            slot_penalties = jet_lag_penalties[team_id][camp_id][venue_id]
                            jet_lag_cost += np.mean(list(slot_penalties.values())) if isinstance(slot_penalties, dict) else 0
            kpis['jet_lag'][team_id] = jet_lag_cost
            
            # KPI 1.4: Border crossings (count venues in different country than camp)
            camp_data = self.data.base_camps[self.data.base_camps['base_camp_id'] == camp_id]
            camp_country = camp_data['country'].values[0] if len(camp_data) > 0 else None
            
            border_crossings = 0
            for venue_id in team_venues:
                venue_data = self.data.venues[self.data.venues['venue_id'] == venue_id]
                venue_country = venue_data['country'].values[0] if len(venue_data) > 0 else None
                if camp_country and venue_country and camp_country != venue_country:
                    border_crossings += 1
            kpis['border_crossings'][team_id] = border_crossings
            
            # KPI 2.4: Altitude disruption (simplified - sum of excess altitude differences)
            camp_elev_data = self.data.base_camps[self.data.base_camps['base_camp_id'] == camp_id]
            
            # Handle missing elevation column gracefully
            if 'elevation_m' in self.data.base_camps.columns and len(camp_elev_data) > 0:
                camp_elev = camp_elev_data['elevation_m'].values[0]
            else:
                camp_elev = 0  # Default if column doesn't exist
            
            altitude_penalty = 0.0
            for venue_id in team_venues:
                venue_elev_data = self.data.venues[self.data.venues['venue_id'] == venue_id]
                if 'elevation_m' in self.data.venues.columns and len(venue_elev_data) > 0:
                    venue_elev = venue_elev_data['elevation_m'].values[0]
                    elev_diff = abs(venue_elev - camp_elev)
                    # Penalty = max(0, diff - 500) / 1000
                    altitude_penalty += max(0, elev_diff - 500) / 1000.0
            kpis['altitude'][team_id] = altitude_penalty
        
        return kpis
    
    def _get_match_assignment(self, match_idx: int,
                             venue_assignment: np.ndarray) -> List[int]:
        """
        Get the assigned venue for a specific match.
        
        Args:
            match_idx: Index of the match
            venue_assignment: Array a_ms
        
        Returns:
            List of venue indices where match is assigned
        """
        assigned = []
        for venue_idx in range(venue_assignment.shape[1]):
            if venue_assignment[match_idx, venue_idx] > 0.5:
                assigned.append(venue_idx)
        return assigned
