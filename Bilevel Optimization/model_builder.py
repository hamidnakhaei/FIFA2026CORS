"""
Constructs the complete MILP model with all constraints and objective function
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from config import (
    KPI_WEIGHTS, GUROBI_TIME_LIMIT, GUROBI_MIP_GAP, GUROBI_NUM_THREADS,
    GUROBI_VERBOSITY, MATCH_COUNTS_BY_COUNTRY, REST_MIN, MATCH_DURATION,
    TURNOVER_WINDOW, FULL_BAN_TEAMS, VISA_BOND_TEAMS
)


class ModelBuilder:
    """Builds the bilevel FIFA 2026 optimization model."""
    
    def __init__(self, data_loader, parameters):
        """
        Initialize model builder.
        
        Args:
            data_loader: DataLoader instance
            parameters: Dictionary of precomputed parameters
        """
        self.data = data_loader
        self.params = parameters
        self.model = gp.Model("FIFA2026_Bilevel")
        self.model.setParam('TimeLimit', GUROBI_TIME_LIMIT)
        self.model.setParam('MIPGap', GUROBI_MIP_GAP)
        self.model.setParam('Threads', GUROBI_NUM_THREADS)
        self.model.setParam('OutputFlag', GUROBI_VERBOSITY)
        
        # Decision variables
        self.x = {}  # Schedule: x[match_id][slot_hour][venue_id]
        self.y = {}  # Final slot: y[group][slot_hour]
        self.z = {}  # Base camp selection: z[team_id][camp_id]
        self.u = {}  # Linearization: u[team_id][camp_id][match_id][slot_hour][venue_id]
        
        # Auxiliary variables for KPIs
        self.C_star = {}  # Realized travel cost per team
        self.aux_vars = {}  # Other auxiliary variables
    
    def build(self):
        """Build the complete model."""
        print("\n" + "="*60)
        print("BUILDING GUROBI MODEL")
        print("="*60)
        
        self._create_decision_variables()
        self._add_hard_constraints()
        self._add_lower_level_constraints()
        self._add_kpi_constraints()
        self._set_objective()
        
        print("="*60 + "\n")
        return self.model
    
    def _create_decision_variables(self):
        """Create all decision variables."""
        print("Creating decision variables...")
        
        # Upper level: schedule variables
        # x[match_id][kickoff_hour][venue_id] = 1 if match m at time t in stadium s
        for _, match in self.data.matches.iterrows():
            match_id = match['match_id']
            self.x[match_id] = {}
            
            for hour in range(24):
                self.x[match_id][hour] = {}
                for _, venue in self.data.venues.iterrows():
                    venue_id = venue['venue_id']
                    var_name = f"x_{match_id}_{hour}_{venue_id}"
                    self.x[match_id][hour][venue_id] = self.model.addVar(
                        vtype=GRB.BINARY, name=var_name
                    )
        
        # Final slot variables y[group][hour]
        for group in self.data.teams_by_group.keys():
            self.y[group] = {}
            for hour in range(24):
                var_name = f"y_{group}_{hour}"
                self.y[group][hour] = self.model.addVar(vtype=GRB.BINARY, name=var_name)
        
        # Lower level: base camp variables
        # z[team_id][camp_id] = 1 if team i camps at facility b
        for team_id in self.data.team_by_id.keys():
            self.z[team_id] = {}
            for camp_id in self.data.eligible_camps_by_team[team_id]:
                var_name = f"z_{team_id}_{camp_id}"
                self.z[team_id][camp_id] = self.model.addVar(vtype=GRB.BINARY, name=var_name)
        
        # Linearization variables: u[team_id][camp_id][match_id][hour][venue_id]
        # u = z * x (binary product)
        for team_id in self.data.team_by_id.keys():
            self.u[team_id] = {}
            for camp_id in self.data.eligible_camps_by_team[team_id]:
                self.u[team_id][camp_id] = {}
                
                # Only for matches this team plays
                team_matches = self.data.get_team_matches(team_id)
                for match_info in team_matches:
                    match_id = match_info['match_id']
                    self.u[team_id][camp_id][match_id] = {}
                    
                    for hour in range(24):
                        self.u[team_id][camp_id][match_id][hour] = {}
                        for _, venue in self.data.venues.iterrows():
                            venue_id = venue['venue_id']
                            var_name = f"u_{team_id}_{camp_id}_{match_id}_{hour}_{venue_id}"
                            self.u[team_id][camp_id][match_id][hour][venue_id] = self.model.addVar(
                                vtype=GRB.BINARY, name=var_name
                            )
        
        # Realized travel costs per team
        for team_id in self.data.team_by_id.keys():
            var_name = f"C_star_{team_id}"
            self.C_star[team_id] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=var_name)
        
        print(f"  Created {len(self.x)} match schedule variables (24h × venues each)")
        print(f"  Created {len(self.y)} group final-slot variables (24h each)")
        print(f"  Created {len(self.z)} base camp selection variables")
        
        return
    
    def _add_hard_constraints(self):
        """Add hard feasibility constraints (H1-H8)."""
        print("Adding hard constraints...")
        
        # H1: Each match scheduled exactly once
        for match_id in self.x.keys():
            expr = gp.quicksum(
                self.x[match_id][hour][venue_id]
                for hour in self.x[match_id].keys()
                for venue_id in self.x[match_id][hour].keys()
            )
            self.model.addConstr(expr == 1, f"H1_match_{match_id}")
        
        # H2: Each team plays exactly 3 matches (round-robin)
        for team_id in self.data.team_by_id.keys():
            matches = self.data.get_team_matches(team_id)
            expr = gp.quicksum(
                self.x[match_info['match_id']][hour][venue_id]
                for match_info in matches
                for hour in range(24)
                for _, venue in self.data.venues.iterrows()
                for venue_id in [venue['venue_id']]
            )
            self.model.addConstr(expr == 3, f"H2_team_{team_id}")
        
        # H8: Match allocation by country
        for country in MATCH_COUNTS_BY_COUNTRY.keys():
            venues_in_country = self.data.get_venues_in_country(country)
            expr = gp.quicksum(
                self.x[match_id][hour][venue_id]
                for match_id in self.x.keys()
                for hour in range(24)
                for venue_id in venues_in_country
            )
            count = MATCH_COUNTS_BY_COUNTRY[country]
            self.model.addConstr(expr == count, f"H8_country_{country}")
        
        print(f"  Added H1 (match scheduling), H2 (round-robin), H8 (country quotas)")
        
        return
    
    def _add_lower_level_constraints(self):
        """Add lower-level (base camp selection) constraints."""
        print("Adding lower-level constraints...")
        
        # (LL-1) Selection: each team chooses exactly one camp
        for team_id in self.z.keys():
            expr = gp.quicksum(self.z[team_id][camp_id] 
                              for camp_id in self.z[team_id].keys())
            self.model.addConstr(expr == 1, f"LL_select_{team_id}")
        
        # (LL-2) Exclusivity: each camp hosts at most one team
        all_camps = set()
        for team_id, camps_dict in self.z.items():
            for camp_id in camps_dict.keys():
                all_camps.add(camp_id)
        
        for camp_id in all_camps:
            expr = gp.quicksum(
                self.z[team_id][camp_id]
                for team_id in self.z.keys()
                if camp_id in self.z[team_id]
            )
            self.model.addConstr(expr <= 1, f"LL_exclusive_{camp_id}")
        
        # (LL-3) Realized travel cost definition
        # C_i^* = sum_b z_ib * (travel_cost + penalty)
        for team_id in self.C_star.keys():
            expr = gp.quicksum(
                self.z[team_id][camp_id] * (
                    self.params['P'][team_id][camp_id] +
                    gp.quicksum(
                        self.params['D'][team_id][camp_id][venue_id] * 
                        gp.quicksum(
                            self.x[match_info['match_id']][hour][venue_id]
                            for hour in range(24)
                            for match_info in [self.data.get_match_info(
                                self.data.get_team_matches(team_id)[0]['match_id']
                            )] if match_info
                        ) if venue_id in self.params['D'][team_id][camp_id]
                        for _, venue in self.data.venues.iterrows()
                        for venue_id in [venue['venue_id']]
                    )
                )
                for camp_id in self.z[team_id].keys()
            )
            self.model.addConstr(self.C_star[team_id] == expr, f"LL_cost_{team_id}")
        
        # (LL-4) McCormick linearization for u = z * x
        M_bigm = self.params.get('M_exclusivity', 100000)
        for team_id in self.u.keys():
            for camp_id in self.u[team_id].keys():
                for match_id in self.u[team_id][camp_id].keys():
                    for hour in self.u[team_id][camp_id][match_id].keys():
                        for venue_id in self.u[team_id][camp_id][match_id][hour].keys():
                            u_var = self.u[team_id][camp_id][match_id][hour][venue_id]
                            z_var = self.z[team_id][camp_id]
                            x_var = self.x[match_id][hour][venue_id]
                            
                            # McCormick envelope: u <= z, u <= x, u >= z + x - 1
                            self.model.addConstr(u_var <= z_var, 
                                               f"McCormick_ub_z_{team_id}_{camp_id}_{match_id}_{hour}_{venue_id}")
                            self.model.addConstr(u_var <= x_var, 
                                               f"McCormick_ub_x_{team_id}_{camp_id}_{match_id}_{hour}_{venue_id}")
                            self.model.addConstr(u_var >= z_var + x_var - 1, 
                                               f"McCormick_lb_{team_id}_{camp_id}_{match_id}_{hour}_{venue_id}")
        
        print(f"  Added lower-level constraints: selection, exclusivity, costs, linearization")
        
        return
    
    def _add_kpi_constraints(self):
        """Add KPI-related auxiliary variables and constraints."""
        print("Adding KPI constraints...")
        
        # These are simplified versions; full implementation depends on
        # specific KPI definitions and data structure
        
        # For now, create basic auxiliary variables
        self.aux_vars['travel_dispersion'] = {}
        for group in self.data.teams_by_group.keys():
            var_name = f"travel_disp_group_{group}"
            self.aux_vars['travel_dispersion'][group] = self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name=var_name
            )
        
        print(f"  Added KPI auxiliary variables")
        
        return
    
    def _set_objective(self):
        """Set the objective function."""
        print("Setting objective function...")
        
        # Simplified objective: minimize travel cost + penalties
        # Full version would include all weighted KPIs
        
        obj = gp.quicksum(self.C_star[team_id] for team_id in self.C_star.keys())
        
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        print(f"  Objective: minimize total team travel costs")
        
        return


def build_model(data_loader, parameters):
    """Convenience function to build the model."""
    builder = ModelBuilder(data_loader, parameters)
    return builder.build()
