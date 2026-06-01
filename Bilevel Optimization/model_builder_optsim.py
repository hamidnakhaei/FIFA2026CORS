"""
Constructs the MILP model for FIFA 2026 Optimization+Simulation framework.

This model includes only upper-level hard constraints and base-camp-free KPIs.
Base-camp-dependent KPIs are handled by the simulation phase in the solver.

Key differences from bilevel model:
- No lower-level (z) variables for base camp selection
- No C_star, JL, BC, ALT variables for base-camp-dependent KPIs
- Venue-assignment variable 'a' is still created for use by simulator
- Objective includes only base-camp-free KPIs + surrogate KPI predictions
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from config import (
    KPI_WEIGHTS, GUROBI_TIME_LIMIT, GUROBI_MIP_GAP, GUROBI_NUM_THREADS,
    GUROBI_VERBOSITY, MATCH_COUNTS_BY_COUNTRY, REST_MIN, MATCH_DURATION,
    TURNOVER_WINDOW, HOST_NATIONS
)


class ModelBuilder:
    """Builds the FIFA 2026 Opt+Sim optimization model (upper level only)."""
    
    def __init__(self, data_loader, parameters):
        """
        Initialize model builder.
        
        Args:
            data_loader: DataLoader instance
            parameters: ParameterBuilder instance with precomputed parameters
        """
        self.data = data_loader
        self.params = parameters
        self.model = gp.Model("FIFA2026_OptSim")
        self.model.setParam('TimeLimit', GUROBI_TIME_LIMIT)
        self.model.setParam('MIPGap', GUROBI_MIP_GAP)
        self.model.setParam('Threads', GUROBI_NUM_THREADS)
        self.model.setParam('OutputFlag', 1 if GUROBI_VERBOSITY > 0 else 0)
        
        # Build slot list
        self._build_slots()
        
        # Decision variables (upper level only)
        self.x = {}  # Schedule: x[match_id][slot_idx][venue_id]
        self.y = {}  # Final slot: y[group][slot_idx]
        self.a = {}  # Venue assignment: a[match_id][venue_id]
        
        # Auxiliary variables for base-camp-free KPIs
        self.aux_vars = {}
    
    def _build_slots(self):
        """Build list of all possible slots (date-hour combinations)."""
        import pandas as pd
        
        unique_dates = sorted(self.data.matches['date'].unique())
        self.data.matches['hour'] = pd.to_datetime(
            self.data.matches['kickoff_local'], format='%H:%M'
        ).dt.hour
        unique_hours = sorted(self.data.matches['hour'].unique())
        
        self.slots = []
        self.slot_map = {}
        self.slot_tau = {}  # slot_idx -> chronological time in hours (tau_t)

        # Reference epoch = earliest date, so tau is a non-negative hour count
        date_origin = pd.Timestamp(min(unique_dates)).normalize()

        for date in unique_dates:
            day_offset = (pd.Timestamp(date).normalize() - date_origin).days * 24
            for hour in unique_hours:
                slot_idx = len(self.slots)
                self.slots.append((date, hour))
                self.slot_map[slot_idx] = (date, hour)
                self.slot_tau[slot_idx] = day_offset + float(hour)

        print(f"  Built {len(self.slots)} slots (date-hour combinations)")
        print(f"    Kickoff hours used: {sorted(unique_hours)}")
    
    def build(self):
        """Build the complete model (upper level only)."""
        print("\n" + "="*60)
        print("BUILDING GUROBI MODEL (OptSim - Upper Level Only)")
        print("="*60)
        
        self._create_decision_variables()
        self._add_hard_constraints()
        self._add_kpi_constraints()
        self._set_initial_objective()
        
        print("="*60 + "\n")
        return self.model
    
    def _create_decision_variables(self):
        """Create all decision variables."""
        print("Creating decision variables...")
        
        # Upper level: schedule variables
        for _, match in self.data.matches.iterrows():
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
        
        # Venue-assignment variables (needed for simulation)
        for _, match in self.data.matches.iterrows():
            match_id = match['match_id']
            self.a[match_id] = {}
            for _, venue in self.data.venues.iterrows():
                venue_id = venue['venue_id']
                var_name = f"a_{match_id}_{venue_id}"
                self.a[match_id][venue_id] = self.model.addVar(vtype=GRB.BINARY, name=var_name)
        
        # Final slot variables
        for group in self.data.teams_by_group.keys():
            self.y[group] = {}
            for slot_idx in range(len(self.slots)):
                date, hour = self.slot_map[slot_idx]
                var_name = f"y_{group}_{date}_{hour}"
                self.y[group][slot_idx] = self.model.addVar(vtype=GRB.BINARY, name=var_name)
        
        num_x_vars = len(self.x) * len(self.slots) * len(self.data.venues)
        num_a_vars = len(self.a) * len(self.data.venues)
        print(f"  Created {num_x_vars} schedule variables ({len(self.slots)} slots × {len(self.data.venues)} venues)")
        print(f"  Created {num_a_vars} venue-assignment variables (for simulation)")
        print(f"  Created {len(self.y) * len(self.slots)} group final-slot variables")
        
        return
    
    def _team_country(self, team_id):
        """
        Resolve a team's country for host-nation restriction (H6).

        Prefers an explicit 'country' column on the teams table; falls back to
        treating the team_id as a country code (host IDs are 'USA'/'MEX'/'CAN').
        """
        info = self.data.get_team_info(team_id)
        if info is not None and 'country' in info.index:
            val = info['country']
            if isinstance(val, str) and val:
                return val.upper()
        return str(team_id).upper()

    def _add_hard_constraints(self):
        """Add hard feasibility constraints (H1-H8)."""
        print("Adding hard constraints...")
        
        # Pre-compute venue list for efficiency
        venue_ids = [v['venue_id'] for _, v in self.data.venues.iterrows()]
        num_slots = len(self.slots)
        
        # Venue assignment linking: a_ms = sum_t x_mts
        print("  Adding venue assignment constraints...")
        constraint_count = 0
        for match_id in self.x.keys():
            for venue_id in venue_ids:
                expr = gp.quicksum(
                    self.x[match_id][slot_idx][venue_id]
                    for slot_idx in range(num_slots)
                )
                self.model.addConstr(self.a[match_id][venue_id] == expr, 
                                    f"venue_assign_{match_id}_{venue_id}")
                constraint_count += 1
        print(f"    Added {constraint_count} venue assignment constraints")
        
        # H1: Each match scheduled exactly once
        print("  Adding H1 constraints (each match exactly once)...")
        for match_id in self.x.keys():
            expr = gp.quicksum(
                self.x[match_id][slot_idx][venue_id]
                for slot_idx in range(num_slots)
                for venue_id in venue_ids
            )
            self.model.addConstr(expr == 1, f"H1_match_{match_id}")
        print(f"    Added {len(self.x)} H1 constraints")
        
        # H2: Each team plays exactly 3 matches (round-robin)
        print("  Adding H2 constraints (each team exactly 3 matches)...")
        for team_id in self.data.team_by_id.keys():
            matches = self.data.get_team_matches(team_id)
            match_ids = [m['match_id'] for m in matches]
            expr = gp.quicksum(
                self.x[match_id][slot_idx][venue_id]
                for match_id in match_ids
                for slot_idx in range(num_slots)
                for venue_id in venue_ids
            )
            self.model.addConstr(expr == 3, f"H2_team_{team_id}")
        print(f"    Added {len(self.data.team_by_id)} H2 constraints")
        
        # H3: One match per team per round
        print("  Adding H3 constraints (one match per team per round)...")
        rounds = sorted(self.data.matches['round'].unique())
        # Map each round to the set of slot indices belonging to it.
        # Rounds are scheduled in chronological blocks; we partition slots by
        # the rounds actually present on each match date.
        round_h3_count = 0
        for team_id in self.data.team_by_id.keys():
            matches = self.data.get_team_matches(team_id)
            for r in rounds:
                match_ids_r = [m['match_id'] for m in matches if m['round'] == r]
                if not match_ids_r:
                    continue
                expr = gp.quicksum(
                    self.x[match_id][slot_idx][venue_id]
                    for match_id in match_ids_r
                    for slot_idx in range(num_slots)
                    for venue_id in venue_ids
                )
                self.model.addConstr(expr == 1, f"H3_team_{team_id}_round_{r}")
                round_h3_count += 1
        print(f"    Added {round_h3_count} H3 constraints")

        # H4: Stadium turnover -- at most one match per stadium in any rolling
        # window of length TURNOVER_WINDOW (= REST_MIN + match duration).
        print("  Adding H4 constraints (stadium turnover window)...")
        window = TURNOVER_WINDOW
        h4_count = 0
        for venue_id in venue_ids:
            for t in range(num_slots):
                tau_t = self.slot_tau[t]
                window_slots = [
                    tp for tp in range(num_slots)
                    if tau_t <= self.slot_tau[tp] < tau_t + window
                ]
                if len(window_slots) <= 1:
                    continue
                expr = gp.quicksum(
                    self.x[match_id][tp][venue_id]
                    for match_id in self.x.keys()
                    for tp in window_slots
                )
                self.model.addConstr(expr <= 1, f"H4_venue_{venue_id}_slot_{t}")
                h4_count += 1
        print(f"    Added {h4_count} H4 constraints")

        # H5: Minimum team rest -- a team cannot play two matches whose kickoffs
        # are closer than REST_MIN + match duration (kickoff-to-kickoff).
        print("  Adding H5 constraints (minimum team rest)...")
        rest_window = REST_MIN + MATCH_DURATION
        h5_count = 0
        for team_id in self.data.team_by_id.keys():
            matches = self.data.get_team_matches(team_id)
            match_ids = [m['match_id'] for m in matches]
            for a_i in range(len(match_ids)):
                for b_i in range(a_i + 1, len(match_ids)):
                    m1, m2 = match_ids[a_i], match_ids[b_i]
                    for t in range(num_slots):
                        for tp in range(num_slots):
                            if abs(self.slot_tau[t] - self.slot_tau[tp]) < rest_window:
                                expr = (
                                    gp.quicksum(self.x[m1][t][v] for v in venue_ids)
                                    + gp.quicksum(self.x[m2][tp][v] for v in venue_ids)
                                )
                                self.model.addConstr(
                                    expr <= 1,
                                    f"H5_{team_id}_{m1}_{m2}_{t}_{tp}"
                                )
                                h5_count += 1
        print(f"    Added {h5_count} H5 constraints")

        # H6: Host-nation venue restriction -- host-nation teams play only in
        # stadiums located in their own country.
        print("  Adding H6 constraints (host-nation venue restriction)...")
        host_set = set(HOST_NATIONS)
        h6_count = 0
        for team_id in self.data.team_by_id.keys():
            team_country = self._team_country(team_id)
            if team_country not in host_set:
                continue
            allowed_venues = set(self.data.get_venues_in_country(team_country))
            disallowed = [v for v in venue_ids if v not in allowed_venues]
            if not disallowed:
                continue
            matches = self.data.get_team_matches(team_id)
            for m in matches:
                match_id = m['match_id']
                expr = gp.quicksum(
                    self.x[match_id][slot_idx][venue_id]
                    for slot_idx in range(num_slots)
                    for venue_id in disallowed
                )
                self.model.addConstr(expr == 0, f"H6_{team_id}_{match_id}")
                h6_count += 1
        print(f"    Added {h6_count} H6 constraints")

        # H7: Simultaneous final matches -- the two final-round matches of each
        # group kick off in the same slot.
        print("  Adding H7 constraints (simultaneous group finals)...")
        final_round = rounds[-1] if rounds else None
        h7_count = 0
        for group, slot_vars in self.y.items():
            # Final-round matches of this group
            group_matches = self.data.matches[self.data.matches['group'] == group]
            final_match_ids = group_matches[
                group_matches['round'] == final_round
            ]['match_id'].tolist()
            if not final_match_ids:
                continue

            # Exactly one final slot per group
            self.model.addConstr(
                gp.quicksum(slot_vars[s] for s in range(num_slots)) == 1,
                f"H7_oneslot_{group}"
            )
            h7_count += 1

            for s in range(num_slots):
                # Both final matches happen at or after the final slot only in it
                in_slot = gp.quicksum(
                    self.x[m][s][v] for m in final_match_ids for v in venue_ids
                )
                self.model.addConstr(
                    in_slot >= 2 * slot_vars[s],
                    f"H7_inslot_{group}_{s}"
                )
                # No final match strictly after the designated final slot
                later = gp.quicksum(
                    self.x[m][sp][v]
                    for m in final_match_ids
                    for sp in range(num_slots)
                    if self.slot_tau[sp] > self.slot_tau[s]
                    for v in venue_ids
                )
                self.model.addConstr(
                    later <= 2 * (1 - slot_vars[s]),
                    f"H7_nolater_{group}_{s}"
                )
                h7_count += 2
        print(f"    Added {h7_count} H7 constraints")

        # H8: Match allocation by country
        print("  Adding H8 constraints (country quotas)...")
        for country in MATCH_COUNTS_BY_COUNTRY.keys():
            venues_in_country = self.data.get_venues_in_country(country)
            expr = gp.quicksum(
                self.x[match_id][slot_idx][venue_id]
                for match_id in self.x.keys()
                for slot_idx in range(num_slots)
                for venue_id in venues_in_country
            )
            count = MATCH_COUNTS_BY_COUNTRY[country]
            self.model.addConstr(expr == count, f"H8_country_{country}")
        print(f"    Added {len(MATCH_COUNTS_BY_COUNTRY)} H8 constraints")
        
        print(f"  [OK] Hard constraints complete")
    
    def _add_kpi_constraints(self):
        """
        Add auxiliary variables and constraints for base-camp-free KPIs.
        
        In Opt+Sim, base-camp-dependent KPIs (1.2, 1.3, 1.4, 2.4) are computed
        only in the simulation phase.
        """
        print("Adding base-camp-free KPI constraints...")
        
        # Placeholder: KPI auxiliary variables will be added as needed
        # Examples:
        # - Heat load tracking
        # - Venue load balance
        # - Prime-time alignment
        
        # For now, minimal setup - details added during iteration
        self.aux_vars['kpi_terms'] = {}
        
        print(f"  Added base-camp-free KPI variables")
        
        return
    
    def _set_initial_objective(self):
        """
        Set the initial objective function (will be updated in solver loop).
        
        Objective = base-camp-free KPI penalties + surrogate base-camp KPI predictions
        """
        print("Setting initial objective function...")
        
        # Start with zero objective - will be updated with surrogate in solver loop
        obj = gp.LinExpr(0)
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        print(f"  Objective: minimize base-camp-free + surrogate KPIs (updated each iteration)")
        
        return
    
    def update_objective_with_surrogate(self, surrogate_penalty):
        """
        Update model objective with current surrogate KPI penalty.
        
        Called each iteration by the solver with updated surrogate weights.
        
        Args:
            surrogate_penalty: Linear expression representing surrogate KPI penalties
        """
        self.model.setObjective(surrogate_penalty, GRB.MINIMIZE)
    
    def get_schedule_solution(self):
        """Extract schedule solution from model."""
        schedule = {}
        for match_id in self.x.keys():
            for slot_idx in self.x[match_id].keys():
                for venue_id in self.x[match_id][slot_idx].keys():
                    if self.x[match_id][slot_idx][venue_id].X > 0.5:
                        schedule[match_id] = {
                            'slot_idx': slot_idx,
                            'venue_id': venue_id,
                            'slot': self.slot_map[slot_idx]
                        }
        return schedule
    
    def get_venue_assignment(self):
        """Extract venue assignment from model."""
        assignment = {}
        for match_id in self.a.keys():
            for venue_id in self.a[match_id].keys():
                if self.a[match_id][venue_id].X > 0.5:
                    if match_id not in assignment:
                        assignment[match_id] = []
                    assignment[match_id].append(venue_id)
        return assignment


def build_model(data_loader, parameters):
    """
    Convenience function to build the model.
    
    Returns:
        Tuple of (gurobi_model, model_builder) for solution extraction
    """
    builder = ModelBuilder(data_loader, parameters)
    model = builder.build()
    return model, builder