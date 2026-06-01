"""
Parameter builder for FIFA 2026 Bilevel Optimization
Precomputes all parameters (distances, penalties, etc.) needed by the model
"""

import numpy as np
from utils import (
    great_circle_distance, perceived_kickoff_time, jet_lag_penalty,
    countries_differ, wbgt_excess_heat_load
)
from config import VISA_BOND_PENALTY, FULL_BAN_TEAMS, VISA_BOND_TEAMS


class ParameterBuilder:
    """Builds all precomputed parameters for the model."""
    
    def __init__(self, data_loader):
        """
        Initialize parameter builder.
        
        Args:
            data_loader: DataLoader instance with all data loaded
        """
        self.data = data_loader
        self.params = {}
    
    def build_all(self):
        """Build all parameters."""
        self.build_distances()
        self.build_jet_lag_penalties()
        self.build_border_crossing_indicators()
        self.build_travel_contributions()
        self.build_visa_penalties()
        self.build_big_m_values()
        return self.params
    
    def build_distances(self):
        """
        Precompute great-circle distances between all camps and venues.
        dist[camp_id][venue_id] = distance in km
        """
        dist = {}
        
        for _, camp in self.data.base_camps.iterrows():
            camp_id = camp['base_camp_id']
            camp_lat, camp_lon = camp['lat'], camp['lon']
            
            dist[camp_id] = {}
            for _, venue in self.data.venues.iterrows():
                venue_id = venue['venue_id']
                venue_lat, venue_lon = venue['lat'], venue['lon']
                
                distance = great_circle_distance(camp_lat, camp_lon, venue_lat, venue_lon)
                dist[camp_id][venue_id] = distance
        
        self.params['dist'] = dist
        print(f"  Built distance matrix: {len(dist)} camps × {len(self.data.venues)} venues")
        return dist
    
    def build_jet_lag_penalties(self):
        """
        Precompute jet-lag penalties.
        Phi[team_id][camp_id][venue_id][slot_hour] = penalty in hours
        """
        Phi = {}
        
        for team_id in self.data.team_by_id.keys():
            Phi[team_id] = {}
            
            for camp_id in self.data.eligible_camps_by_team[team_id]:
                camp_info = self.data.get_camp_info(camp_id)
                camp_tz = camp_info['utc_offset_june']
                
                Phi[team_id][camp_id] = {}
                
                for _, venue in self.data.venues.iterrows():
                    venue_id = venue['venue_id']
                    venue_tz = venue['utc_offset_june']
                    
                    Phi[team_id][camp_id][venue_id] = {}
                    
                    # For each possible kickoff time (slot)
                    for slot in range(24):  # 24 possible hours
                        perceived_hour = perceived_kickoff_time(slot, venue_tz, camp_tz)
                        penalty = jet_lag_penalty(perceived_hour)
                        Phi[team_id][camp_id][venue_id][slot] = penalty
        
        self.params['Phi'] = Phi
        print(f"  Built jet-lag penalties for {len(Phi)} teams")
        return Phi
    
    
    def build_border_crossing_indicators(self):
        """
        Precompute border-crossing indicators.
        beta[camp_id][venue_id] = 1 if different countries, 0 otherwise
        """
        beta = {}
        
        for _, camp in self.data.base_camps.iterrows():
            camp_id = camp['base_camp_id']
            camp_country = camp['country']
            
            beta[camp_id] = {}
            
            for _, venue in self.data.venues.iterrows():
                venue_id = venue['venue_id']
                venue_country = venue['country']
                
                # 1 if different countries, 0 otherwise
                beta[camp_id][venue_id] = 1 if countries_differ(camp_country, venue_country) else 0
        
        self.params['beta'] = beta
        print(f"  Built border-crossing indicators")
        return beta
    
    def build_travel_contributions(self):
        """
        Precompute travel contribution (D parameter).
        D[team_id][camp_id][venue_id] = 2 * distance (for round trip)
        """
        D = {}
        
        for team_id in self.data.team_by_id.keys():
            D[team_id] = {}
            
            for camp_id in self.data.eligible_camps_by_team[team_id]:
                D[team_id][camp_id] = {}
                
                for _, venue in self.data.venues.iterrows():
                    venue_id = venue['venue_id']
                    
                    # 2 * distance for round trip
                    distance = self.params['dist'][camp_id][venue_id]
                    D[team_id][camp_id][venue_id] = 2 * distance
        
        self.params['D'] = D
        print(f"  Built travel contribution parameters")
        return D
    
    
    def build_visa_penalties(self):
        """
        Precompute visa penalties for base camps.
        P[team_id][camp_id] = penalty value
        """
        P = {}
        
        # Get US camps
        us_camps = set(self.data.get_camps_in_country("USA"))
        
        for team_id in self.data.team_by_id.keys():
            P[team_id] = {}
            
            for camp_id in self.data.eligible_camps_by_team[team_id]:
                penalty = 0
                
                # Hard exclusion: full ban teams cannot use US camps
                if team_id in FULL_BAN_TEAMS and camp_id in us_camps:
                    penalty = float('inf')  # Effectively forbidden
                
                # Soft penalty: visa-bond teams get penalty for US camps
                elif team_id in VISA_BOND_TEAMS and camp_id in us_camps:
                    penalty = VISA_BOND_PENALTY
                
                P[team_id][camp_id] = penalty
        
        self.params['P'] = P
        print(f"  Built visa penalties for {len(P)} teams")
        return P
    
    def build_big_m_values(self):
        """
        Calculate Big-M values for the model.
        Used for conditional constraints and linearizations.
        """
        # Calculate M for camp exclusivity constraints
        # M should be large enough to deactivate optimality constraints
        M_exclusivity = 0
        
        for team_id in self.data.team_by_id.keys():
            # Calculate total travel cost for each camp
            camp_costs = {}
            for camp_id in self.data.eligible_camps_by_team[team_id]:
                cost = 0
                for _, venue in self.data.venues.iterrows():
                    venue_id = venue['venue_id']
                    cost += self.params['D'][team_id][camp_id][venue_id]
                
                # Add visa penalty if applicable
                if team_id in VISA_BOND_TEAMS:
                    cost += VISA_BOND_PENALTY
                
                camp_costs[camp_id] = cost
            
            # Find max cost difference for this team
            if camp_costs:
                max_cost = max(camp_costs.values())
                min_cost = min(camp_costs.values())
                M_exclusivity = max(M_exclusivity, max_cost - min_cost)
        
        # Ensure M is at least reasonably large
        M_exclusivity = max(M_exclusivity, 100000)
        
        self.params['M_exclusivity'] = M_exclusivity
        print(f"  Big-M for camp exclusivity: {M_exclusivity:.0f}")
        
        return M_exclusivity
    
    def summary(self):
        """Print summary of parameters."""
        print("\n" + "="*60)
        print("PARAMETER SUMMARY")
        print("="*60)
        if 'dist' in self.params:
            num_camps = len(self.params['dist'])
            num_venues = len(list(self.params['dist'].values())[0]) if num_camps > 0 else 0
            print(f"Distance matrix: {num_camps} camps × {num_venues} venues")
        
        if 'Phi' in self.params:
            print(f"Jet-lag penalties: {len(self.params['Phi'])} teams")
        
        if 'P' in self.params:
            print(f"Visa penalties: {len(self.params['P'])} teams")
        
        if 'M_exclusivity' in self.params:
            print(f"Big-M (exclusivity): {self.params['M_exclusivity']:.0f}")
        
        print("="*60 + "\n")


def build_parameters(data_loader):
    """Convenience function to build all parameters."""
    builder = ParameterBuilder(data_loader)
    params = builder.build_all()
    builder.summary()
    return params
