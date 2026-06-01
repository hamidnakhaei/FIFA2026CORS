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
        self.model.setParam('OutputFlag', 1 if GUROBI_VERBOSITY > 0 else 0)
        
        # Build slot list (all possible date-time combinations)
        self._build_slots()
        
        # Decision variables
        self.x = {}  # Schedule: x[match_id][slot_idx][venue_id]
        self.y = {}  # Final slot: y[group][slot_idx]
        self.a = {}  # Venue assignment: a[match_id][venue_id] = sum_t x[match_id][slot_idx][venue_id]
        self.z = {}  # Base camp selection: z[team_id][camp_id]
        
        # Auxiliary variables for KPIs
        self.C_star = {}  # Realized travel cost per team
        self.JL = {}     # Jet-lag cost per team
        self.BC = {}     # Border-crossing count per team
        self.aux_vars = {}  # Other auxiliary variables
    
    def _build_slots(self):
        """
        Build list of all possible slots (date, hour combinations).
        
        Slots are enumerated from actual match dates and kickoff hours used in matches.
        """
        import pandas as pd
        
        # Get unique dates from matches
        unique_dates = sorted(self.data.matches['date'].unique())
        
        # Extract unique hours from kickoff times
        # kickoff_local is in "HH:MM" format
        self.data.matches['hour'] = pd.to_datetime(
            self.data.matches['kickoff_local'], format='%H:%M'
        ).dt.hour
        unique_hours = sorted(self.data.matches['hour'].unique())
        
        # Create slots: each slot is a (date, hour) pair
        # Hours: only those used in actual match times
        self.slots = []
        self.slot_map = {}  # Map slot_idx to (date, hour)
        
        for date in unique_dates:
            for hour in unique_hours:
                slot_idx = len(self.slots)
                self.slots.append((date, hour))
                self.slot_map[slot_idx] = (date, hour)
        
        print(f"  Built {len(self.slots)} slots (date-hour combinations)")
        print(f"    Kickoff hours used: {sorted(unique_hours)}")
    
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
        # x[match_id][slot_idx][venue_id] = 1 if match m at slot (date,hour) in stadium s
        for _, match in self.data.matches.iterrows():
            print('1', match['match_id'])
            match_id = match['match_id']
            self.x[match_id] = {}
            
            for slot_idx in range(len(self.slots)):
                self.x[match_id][slot_idx] = {}
                for _, venue in self.data.venues.iterrows():
                    venue_id = venue['venue_id']
                    date, hour = self.slot_map[slot_idx]
                    var_name = f"x_{match_id}_{date}_{hour}_{venue_id}"
                    self.x[match_id][slot_idx][venue_id] = self.model.addVar(
                        vtype=GRB.BINARY, name=var_name
                    )
        
        # Venue-assignment variables (collapsed slot dimension)
        # a[match_id][venue_id] = 1 if match m is played at stadium s (regardless of slot)
        # a_ms = sum_t x_mts (derived variable, not stored but used in constraints)
        for _, match in self.data.matches.iterrows():
            match_id = match['match_id']
            self.a[match_id] = {}
            for _, venue in self.data.venues.iterrows():
                venue_id = venue['venue_id']
                var_name = f"a_{match_id}_{venue_id}"
                self.a[match_id][venue_id] = self.model.addVar(vtype=GRB.BINARY, name=var_name)
        
        # Final slot variables y[group][slot_idx]
        for group in self.data.teams_by_group.keys():
            print('2', group)
            self.y[group] = {}
            for slot_idx in range(len(self.slots)):
                date, hour = self.slot_map[slot_idx]
                var_name = f"y_{group}_{date}_{hour}"
                self.y[group][slot_idx] = self.model.addVar(vtype=GRB.BINARY, name=var_name)
        
        # Lower level: base camp variables
        # z[team_id][camp_id] = 1 if team i camps at facility b
        for team_id in self.data.team_by_id.keys():
            print('3', team_id)
            self.z[team_id] = {}
            for camp_id in self.data.eligible_camps_by_team[team_id]:
                var_name = f"z_{team_id}_{camp_id}"
                self.z[team_id][camp_id] = self.model.addVar(vtype=GRB.BINARY, name=var_name)
        
        # Realized travel costs per team (KPI 1.1, 1.2)
        for team_id in self.data.team_by_id.keys():
            print('5', team_id)
            var_name = f"C_star_{team_id}"
            self.C_star[team_id] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=var_name)
        
        # Jet-lag costs per team (KPI 1.3)
        for team_id in self.data.team_by_id.keys():
            var_name = f"JL_{team_id}"
            self.JL[team_id] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=var_name)
        
        # Border-crossing counts per team (KPI 1.4)
        for team_id in self.data.team_by_id.keys():
            var_name = f"BC_{team_id}"
            self.BC[team_id] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=3, name=var_name)
        
        
        num_x_vars = len(self.x) * len(self.slots) * len(self.data.venues)
        num_a_vars = len(self.a) * len(self.data.venues)
        print(f"  Created {num_x_vars} schedule variables ({len(self.slots)} slots × {len(self.data.venues)} venues)")
        print(f"  Created {num_a_vars} venue-assignment variables")
        print(f"  Created {len(self.y) * len(self.slots)} group final-slot variables")
        print(f"  Created {len(self.z)} base camp selection variables")
        
        return
    
    def _add_hard_constraints(self):
        """Add hard feasibility constraints (H1-H8)."""
        
        # Venue assignment linking: a_ms = sum_t x_mts
        for match_id in self.x.keys():
            for _, venue in self.data.venues.iterrows():
                venue_id = venue['venue_id']
                expr = gp.quicksum(
                    self.x[match_id][slot_idx][venue_id]
                    for slot_idx in range(len(self.slots))
                )
                self.model.addConstr(self.a[match_id][venue_id] == expr, 
                                    f"venue_assign_{match_id}_{venue_id}")
        
        # H1: Each match scheduled exactly once
        for match_id in self.x.keys():
            expr = gp.quicksum(
                self.x[match_id][slot_idx][venue_id]
                for slot_idx in self.x[match_id].keys()
                for venue_id in self.x[match_id][slot_idx].keys()
            )
            self.model.addConstr(expr == 1, f"H1_match_{match_id}")
        
        # H2: Each team plays exactly 3 matches (round-robin)
        for team_id in self.data.team_by_id.keys():
            matches = self.data.get_team_matches(team_id)
            expr = gp.quicksum(
                self.x[match_info['match_id']][slot_idx][venue_id]
                for match_info in matches
                for slot_idx in range(len(self.slots))
                for _, venue in self.data.venues.iterrows()
                for venue_id in [venue['venue_id']]
            )
            self.model.addConstr(expr == 3, f"H2_team_{team_id}")
        
        # H8: Match allocation by country
        for country in MATCH_COUNTS_BY_COUNTRY.keys():
            venues_in_country = self.data.get_venues_in_country(country)
            expr = gp.quicksum(
                self.x[match_id][slot_idx][venue_id]
                for match_id in self.x.keys()
                for slot_idx in range(len(self.slots))
                for venue_id in venues_in_country
            )
            count = MATCH_COUNTS_BY_COUNTRY[country]
            self.model.addConstr(expr == count, f"H8_country_{country}")
        
        print(f"  Added venue assignment, H1 (match scheduling), H2 (round-robin), H8 (country quotas)")
        
        return
    
    def _add_lower_level_constraints(self):
        """
        Add lower-level (base camp selection) constraints with exact linearization.
        
        Key features:
        - No McCormick u variables (removed completely)
        - Uses venue-assignment variable a to collapse slot dimension
        - Big-M indicator constraints to pin C_i^* without products
        - Conditional optimality cuts that respect camp exclusivity
        """
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
        
        # Precompute camp costs
        # C_ib = sum_{m in M_i} sum_{s in S} D_ibs * a_ms + P_ib
        # where D_ibs = 2 * dist(b, s) and P_ib is the visa penalty
        camp_costs = {}  # camp_costs[team_id][camp_id] = dict of (match_id, venue_id) -> cost coeff
        
        for team_id in self.data.team_by_id.keys():
            camp_costs[team_id] = {}
            for camp_id in self.data.eligible_camps_by_team[team_id]:
                camp_costs[team_id][camp_id] = {}
                
                # Precompute linear expression: sum_m sum_s D_ibs * a_ms
                team_matches = self.data.get_team_matches(team_id)
                
                for match_info in team_matches:
                    match_id = match_info['match_id']
                    camp_costs[team_id][camp_id][match_id] = {}
                    
                    for _, venue in self.data.venues.iterrows():
                        venue_id = venue['venue_id']
                        # D_ibs = 2 * distance (round trip)
                        travel_coeff = self.params['D'][team_id][camp_id][venue_id]
                        camp_costs[team_id][camp_id][match_id][venue_id] = travel_coeff
        
        # (LL-3) Realised travel cost: big-M indicator constraints
        # C_i^* >= sum_m sum_s D_ibs * a_ms + P_ib - M * (1 - z_ib)
        # When z_ib = 1: C_i^* >= C_ib (lower bound)
        # When z_ib = 0: C_i^* >= C_ib - M (vacuous, M is large)
        
        M_realised = self.params.get('M_exclusivity', 100000)
        
        for team_id in self.z.keys():
            for camp_id in self.z[team_id].keys():
                # Build the cost expression for this (team, camp) pair
                cost_expr = gp.quicksum(
                    camp_costs[team_id][camp_id][match_id][venue_id] * self.a[match_id][venue_id]
                    for match_id in camp_costs[team_id][camp_id].keys()
                    for venue_id in camp_costs[team_id][camp_id][match_id].keys()
                )
                
                # Add visa penalty (constant)
                P_ib = self.params['P'][team_id][camp_id]
                cost_expr += P_ib
                
                # Big-M indicator: C_i^* >= cost_expr - M * (1 - z_ib)
                self.model.addConstr(
                    self.C_star[team_id] >= cost_expr - M_realised * (1 - self.z[team_id][camp_id]),
                    f"LL_cstar_ind_{team_id}_{camp_id}"
                )
        
        # (LL-4) Conditional optimality cut (no-justified-envy)
        # C_i^* <= C_ib + M * sum_{j != i : b in B_j} z_jb
        # When camp b is unclaimed by others: C_i^* <= C_ib (forced to be optimal)
        # When camp b is taken by j: C_i^* <= C_ib + M (deactivated)
        
        M_optimality = self.params.get('M_exclusivity', 100000)
        
        for team_id in self.z.keys():
            for camp_id in self.z[team_id].keys():
                # Build the cost expression for this camp
                cost_expr = gp.quicksum(
                    camp_costs[team_id][camp_id][match_id][venue_id] * self.a[match_id][venue_id]
                    for match_id in camp_costs[team_id][camp_id].keys()
                    for venue_id in camp_costs[team_id][camp_id][match_id].keys()
                )
                
                # Add visa penalty
                P_ib = self.params['P'][team_id][camp_id]
                cost_expr += P_ib
                
                # Build the "camp claimed by others" sum
                other_claims = gp.quicksum(
                    self.z[j][camp_id]
                    for j in self.z.keys()
                    if j != team_id and camp_id in self.z[j]
                )
                
                # Conditional optimality: C_i^* <= C_ib + M * (sum of other claims)
                self.model.addConstr(
                    self.C_star[team_id] <= cost_expr + M_optimality * other_claims,
                    f"LL_condopt_{team_id}_{camp_id}"
                )
        
        # (LL-5) Base-camp-dependent KPI indicators (no products)
        # Similar big-M structure for JL, BC, ALT
        
        M_kpi = self.params.get('M_exclusivity', 100000)
        
        # KPI 1.3: Jet-lag cost
        for team_id in self.z.keys():
            for camp_id in self.z[team_id].keys():
                # JL_i >= sum_m sum_t sum_s Phi_ibts * x_mts - M * (1 - z_ib)
                jl_expr = gp.LinExpr()
                team_matches = self.data.get_team_matches(team_id)
                
                for match_info in team_matches:
                    match_id = match_info['match_id']
                    for slot_idx in range(len(self.slots)):
                        date, hour = self.slot_map[slot_idx]
                        for _, venue in self.data.venues.iterrows():
                            venue_id = venue['venue_id']
                            # Get jet-lag penalty for this (team, camp, venue, hour)
                            phi_penalty = self.params['Phi'][team_id][camp_id][venue_id].get(hour, 0)
                            jl_expr.addTerms(phi_penalty, self.x[match_id][slot_idx][venue_id])
                
                self.model.addConstr(
                    self.JL[team_id] >= jl_expr - M_kpi * (1 - self.z[team_id][camp_id]),
                    f"LL_jl_ind_{team_id}_{camp_id}"
                )
        
        # KPI 1.4: Border-crossing count
        for team_id in self.z.keys():
            for camp_id in self.z[team_id].keys():
                # BC_i >= sum_m sum_s beta_ibs * a_ms - M * (1 - z_ib)
                bc_expr = gp.quicksum(
                    self.params['beta'][camp_id][venue_id] * self.a[match_id][venue_id]
                    for match_id in (match_info['match_id'] 
                                    for match_info in self.data.get_team_matches(team_id))
                    for _, venue in self.data.venues.iterrows()
                    for venue_id in [venue['venue_id']]
                )
                
                self.model.addConstr(
                    self.BC[team_id] >= bc_expr - M_kpi * (1 - self.z[team_id][camp_id]),
                    f"LL_bc_ind_{team_id}_{camp_id}"
                )
        
        # KPI 2.4: Altitude disruption
        for team_id in self.z.keys():
            for camp_id in self.z[team_id].keys():
                # ALT_i >= sum_m sum_s A_ibs * a_ms - M * (1 - z_ib)
                # where A_ibs is precomputed altitude penalty
                alt_expr = gp.quicksum(
                    self.params['A'].get(camp_id, {}).get(venue_id, 0) * self.a[match_id][venue_id]
                    for match_id in (match_info['match_id'] 
                                    for match_info in self.data.get_team_matches(team_id))
                    for _, venue in self.data.venues.iterrows()
                    for venue_id in [venue['venue_id']]
                )
                
                self.model.addConstr(
                    self.ALT[team_id] >= alt_expr - M_kpi * (1 - self.z[team_id][camp_id]),
                    f"LL_alt_ind_{team_id}_{camp_id}"
                )
        
        print(f"  Added selection, exclusivity, conditional optimality cuts, and KPI indicators")
        
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
    """Convenience function to build the model.
    
    Returns:
        Tuple of (gurobi_model, model_builder) for solution extraction
    """
    builder = ModelBuilder(data_loader, parameters)
    model = builder.build()
    return model, builder
