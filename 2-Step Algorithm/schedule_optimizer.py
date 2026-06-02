"""
Schedule optimization solver (Step A) for FIFA 2026 Group-Stage.
Solves the MILP to assign matches to slots and stadiums, minimizing weighted KPIs.
"""
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict
from pyomo.environ import (
    ConcreteModel,
    Set,
    Param,
    Var,
    Objective,
    Constraint,
    minimize,
    Binary,
    SolverFactory,
    value,
    NonNegativeReals,
    NonNegativeIntegers,
)


class ScheduleOptimizer:
    """Solve Step A: optimize match schedule while holding base camps fixed."""

    def __init__(self, data_loader, kpi_calculator, base_camp_assignment: Dict):
        self.loader = data_loader
        self.kpi_calc = kpi_calculator
        self.base_camp_assignment = base_camp_assignment

        self.matches = data_loader.get_matches()
        self.venues = data_loader.get_venues()
        self.teams = data_loader.get_teams()
        self.base_camps = data_loader.get_base_camps()

        # Get sets and indices
        (
            self.M,
            self.T,
            self.S,
            self.I,
            self.G,
            self.M_i,
            self.M_g,
            self.S_c,
        ) = data_loader.get_sets_and_indices()

        self.params = data_loader.get_parameters()
        self.model = None
        
        # Extract unique (date, time) slots from matches
        self.date_time_slots = self.T
        self.slot_index_to_datetime = {i: slot for i, slot in enumerate(self.date_time_slots)}
        self.datetime_to_slot_index = {slot: i for i, slot in enumerate(self.date_time_slots)}
        
        # Compute KPI normalization factors
        self.kpi_normalization_factors = self._compute_kpi_normalization_factors()

    def _compute_kpi_normalization_factors(self) -> Dict:
        """
        Compute reference/baseline values for each KPI to normalize them.
        KPI_normalized = KPI / reference_value makes all KPIs dimensionless and comparable.
        
        Returns dict mapping kpi_name -> reference_value
        """
        factors = {}
        
        # KPI 1.2: Travel dispersion (km)
        # Worst case: all teams travel max distance for all matches
        max_dist = self.params.get("dist", {})
        factors["kpi_1_2"] = max(max_dist.values()) * len(self.M) * 2.0  # Upper bound estimate
        
        # KPI 1.3: Circadian shift cost (hours penalty)
        # Worst case: all teams at wrong timezone, all matches in subjective night
        max_penalty_per_match = 8.0  # Max circadian penalty per match
        factors["kpi_1_3"] = max_penalty_per_match * len(self.M) * 2.0  # 2 teams per match
        
        # KPI 1.4: Geographic dispersion (count)
        # Range: 0 to 3 clusters per team × 32 teams
        factors["kpi_1_4"] = 3.0 * len(self.I)
        
        # KPI 1.6: Rest asymmetry (hours)
        # Worst case: max rest difference across all matches
        max_rest_hours = 7 * 24  # Max rest in group stage
        factors["kpi_1_6"] = max_rest_hours * len(self.M)
        
        # KPI 1.7: Visa restrictions (count)
        # Worst case: all affected teams play at US stadiums
        us_stadiums = self.venues[self.venues["country"] == "USA"]["venue_id"].tolist()
        ban_teams = set(self.params.get("us_visa_ban_teams", []))
        factors["kpi_1_7"] = len(ban_teams) * len(us_stadiums)  # Maximum affected matches
        
        # KPI 2.2: Heat load (WBGT hours per team)
        # Excess above 28°C, worst case ~10°C excess × 3 matches × 32 teams
        factors["kpi_2_2"] = 10.0 * 3 * len(self.I)
        
        # KPI 3.3: First-mover balance (standard deviation)
        # Range: 0 to ~2 early slots per team
        factors["kpi_3_3"] = 2.0 * len(self.I)
        
        # KPI 4.1: Venue-load balance (mean absolute deviation)
        # Worst case: unbalanced distribution of 72 matches across 12 stadiums
        avg_matches = len(self.M) / len(self.S) if len(self.S) > 0 else 1
        max_deviation = len(self.M) - avg_matches  # Max possible deviation
        factors["kpi_4_1"] = max_deviation * len(self.S) / 2  # Typical MAD estimate
        
        # KPI 4.2: Same-city overlap (count)
        # Worst case: multiple overlaps across all cities and slots
        factors["kpi_4_2"] = len(self.M) * len(self.T) / 10  # Rough estimate
        
        # KPI 5.2: Marquee-match overlap penalty (count × popularity)
        # Worst case: all high-profile matches in same slot
        factors["kpi_5_2"] = len(self.M)  # Number of high-profile match pairs
        
        # KPI 5.3: Host-city economic equity (mean absolute deviation of commercial value)
        # Range: depends on match values, normalize by sum of all match values
        total_match_value = sum(self.params.get("match_value", {}).values())
        factors["kpi_5_3"] = total_match_value / 2 if total_match_value > 0 else 1.0
        
        # Apply minimum threshold to avoid division by zero
        for kpi in factors:
            factors[kpi] = max(factors[kpi], 0.1)
        
        return factors
    
    def _compute_kpi_coefficients(self) -> Dict:
        """
        Precompute KPI contributions for each (match, time_slot, stadium) combination.
        Returns dict mapping (m, t, s) -> weighted_kpi_cost
        t is an index into date_time_slots.
        
        Each coefficient represents the marginal cost of assigning x[m,t,s]=1.
        """
        kpi_costs = {}
        weights = self.params.get("weights", {})
        
        for m in self.M:
            match_row = self.matches[self.matches["match_id"] == m]
            match = match_row.iloc[0]
            
            team_a_id = match["team_a_id"]
            team_b_id = match["team_b_id"]
            match_group = match["group"]
            
            for t_idx, (date, time_str) in enumerate(self.date_time_slots):
                for s in self.S:
                    cost = 0.0
                    
                    # KPI 1.2: Intra-group travel dispersion
                    # TD_i = 2 * Σ dist(base_camp_i, stadium)
                    bc_a = self.base_camp_assignment[team_a_id]
                    dist_a = self.params.get("dist", {}).get((bc_a, s), 0.0)
                    cost += 2.0 * dist_a * weights.get("kpi_1_2", 0.0)
                    
                    bc_b = self.base_camp_assignment[team_b_id]
                    dist_b = self.params.get("dist", {}).get((bc_b, s), 0.0)
                    cost += 2.0 * dist_b * weights.get("kpi_1_2", 0.0)
                    
                    # KPI 1.3: Circadian shift cost
                    # φ(τ_hat) = perceived night penalty for each team
                    try:
                        kickoff_hour = float(time_str.split(":")[0])
                    except:
                        kickoff_hour = 12.0
                    
                    for team_id in [team_a_id, team_b_id]:
                        bc_id = self.base_camp_assignment[team_id]
                        camp_tz = self.params.get("tzone_basecamp", {}).get(bc_id, 0)
                        stadium_tz = self.params.get("tzone_stadium", {}).get(s, 0)
                        tz_offset = stadium_tz - camp_tz
                        
                        # Perceived kickoff time (mod 24)
                        tau_hat = (kickoff_hour - tz_offset) % 24
                        
                        # Circadian penalty function φ(τ_hat)
                        tau_lo = 23.0  # Start of subjective night
                        tau_hi = 7.0   # End of subjective night
                        max_penalty = 8.0
                        
                        if tau_hat >= tau_lo or tau_hat <= tau_hi:
                            if tau_hat >= tau_lo:
                                penalty = min(tau_hat - tau_lo, 24 - (tau_hat - tau_lo))
                            else:
                                penalty = min(tau_hat + 24 - tau_lo, 24 - (tau_hat + 24 - tau_lo))
                            penalty = min(penalty, max_penalty)
                        else:
                            penalty = 0.0
                        
                        cost += penalty * weights["kpi_1_3"]
                    
                    # KPI 1.4: Match-venue geographic dispersion
                    # Count unique clusters/countries visited by each team
                    for team_id in [team_a_id, team_b_id]:
                        bc_id = self.base_camp_assignment[team_id]
                        bc_country = self.base_camps[self.base_camps["base_camp_id"] == bc_id]["country"].values
                        stadium_country = self.venues[self.venues["venue_id"] == s]["country"].values
                        
                        if len(bc_country) > 0 and len(stadium_country) > 0:
                            # Cost increases if crossing country border
                            if bc_country[0] != stadium_country[0]:
                                cost += weights["kpi_1_4"]
                    
                    # KPI 1.7: Entry and visa restriction exposure
                    # Penalty if match at US stadium and team has visa issues
                    us_stadiums = self.venues[self.venues["country"] == "USA"]["venue_id"].tolist()
                    if s in us_stadiums:
                        ban_teams = set(self.params.get("us_visa_ban_teams", []))
                        bond_teams = set(self.params.get("us_visa_bond_teams", []))
                        
                        for team_id in [team_a_id, team_b_id]:
                            if team_id in ban_teams:
                                cost += 1.0 * weights["kpi_1_7"]
                            elif team_id in bond_teams:
                                cost += 0.5 * weights["kpi_1_7"]
                    
                    # KPI 2.2: Per-team heat load
                    # Excess WBGT: h_mt = max(0, WBGT - 28)
                    venue_weather = self.params.get("weather", {})
                    if isinstance(venue_weather, dict):
                        temp_c = venue_weather.get(s, {}).get("temperature_c", 20.0)
                    else:
                        temp_c = 20.0
                    
                    wbgt_estimated = 0.5 * temp_c + 14.0
                    excess_wbgt = max(0.0, wbgt_estimated - 28.0)
                    cost += excess_wbgt * 2.0 * weights["kpi_2_2"]  # Factor of 2 for two teams

                    # KPI 1.6: Rest asymmetry between opponents
                    # This requires knowing relative match timing, use heuristic coefficient
                    
                    # KPI 3.3: Round-order balance (first-mover)
                    # Heuristic: penalize afternoon/evening slots for balance

                    
                    # KPI 4.1: Venue-load balance
                    # Penalize underutilized stadiums (encourage spread)

                    
                    # KPI 4.2: Same-city overlap
                    # Penalize multiple matches in same city on same date (address operationally)
                    
                    # KPI 5.2: Marquee-match slot quality
                    # Bonus for high-profile matches in primetime slots
                    
                    # KPI 5.3: Host-city economic equity
                    # Distribute commercial value across venues

                

        
        return kpi_costs

    def build_model(self) -> ConcreteModel:
        """Build the MILP model for schedule optimization."""
        model = ConcreteModel()

        # Sets
        model.M = Set(initialize=list(self.M))  # Matches
        model.T = Set(initialize=list(self.T))  # Time slots (date/time combinations)
        model.T_indx = Set(initialize=list(range(len(self.T))))  # Time slots (date/time combinations)
        model.S = Set(initialize=list(self.S))  # Stadiums
        model.I = Set(initialize=list(self.I))  # Teams
        model.G = Set(initialize=list(self.G))  # Groups

        # Decision Variables
        model.x = Var(model.M, model.T, model.S, within=Binary)  # Assignment vars
        model.y = Var(model.G, model.T, within=Binary)  # Final slot indicator

        # Auxiliary variables for KPIs (used in constraints)
        model.delta_m = Var(model.M, within=NonNegativeReals)  # KPI 1.6: Rest asymmetry per match
        model.e_i = Var(model.I, within=NonNegativeReals)  # KPI 3.3: First-mover balance per team
        model.d_s = Var(model.S, within=NonNegativeReals)  # KPI 4.1: Venue load deviation
        model.o_ct = Var(model.G, model.T, within=NonNegativeReals)  # KPI 4.2: Same-city overlap
        model.o_P_mm_prime = Var(model.M, model.M, model.T, within=NonNegativeReals)  # KPI 5.2: Marquee match overlap
        model.p_s = Var(model.S, within=NonNegativeReals)  # KPI 5.3: Host-city economic equity deviation
        model.vc_s = Var(model.S, within=NonNegativeReals)  # KPI 5.3: Venue commercial value
        model.venue_count = Var(model.S, within=NonNegativeIntegers)  # Number of matches per venue
        model.r_im = Var(model.I, model.M, within=NonNegativeReals)  # Rest days for team i before match m

        # Parameters
        model.N_c = Param(
            ["USA", "MEX", "CAN"], initialize=self.params["N_c"]
        )  # Matches per country requirement (indexed)

        # Precompute KPI coefficients for each (match, time_slot, stadium)
        kpi_coefficients = self._compute_kpi_coefficients()
        model.kpi_cost = Param(
            model.M,
            model.T,
            model.S,
            initialize={(m, t, s): kpi_coefficients.get((m, t, s), 0.0)
                        for m in self.M for t in self.T for s in self.S},
        )

        # Objective: Minimize full weighted KPI (all 13 KPIs)
        # Direct KPIs (1.2, 1.3, 1.4, 1.7, 2.2) via precomputed coefficients
        # + Auxiliary variable KPIs (1.6, 3.3, 4.1, 4.2, 5.2, 5.3)
        def objective_rule(model):
            weights = self.params.get("weights", {})
            norm_factors = self.kpi_normalization_factors
            
            # Direct coefficient-based KPIs (already include weights)
            # Normalize by the sum of their normalization factors weighted by their weights
            direct_cost = sum(
                model.kpi_cost[m, t, s] * model.x[m, t, s]
                for m in model.M
                for t in model.T
                for s in model.S
            )
            # Normalize direct cost by typical scale
            direct_cost_norm = (
                weights.get("kpi_1_2", 0.0) * norm_factors.get("kpi_1_2", 1.0) +
                weights.get("kpi_1_3", 0.0) * norm_factors.get("kpi_1_3", 1.0) +
                weights.get("kpi_1_4", 0.0) * norm_factors.get("kpi_1_4", 1.0) +
                weights.get("kpi_1_7", 0.0) * norm_factors.get("kpi_1_7", 1.0) +
                weights.get("kpi_2_2", 0.0) * norm_factors.get("kpi_2_2", 1.0)
            )
            direct_cost_normalized = (
                direct_cost / direct_cost_norm if direct_cost_norm > 0 else direct_cost
            )
            
            # Auxiliary KPIs: normalize by their reference values
            # KPI 1.6: Rest asymmetry (sum of rest differences)
            kpi_1_6 = sum(model.delta_m[m] for m in model.M)
            kpi_1_6_normalized = kpi_1_6 / norm_factors.get("kpi_1_6", 1.0)
            
            # KPI 3.3: First-mover balance (sum of deviations)
            kpi_3_3 = sum(model.e_i[i] for i in model.I)
            kpi_3_3_normalized = kpi_3_3 / norm_factors.get("kpi_3_3", 1.0)
            
            # KPI 4.1: Venue-load balance (mean absolute deviation of match counts)
            kpi_4_1 = sum(model.d_s[s] for s in model.S)
            kpi_4_1_normalized = kpi_4_1 / norm_factors.get("kpi_4_1", 1.0)
            
            # KPI 4.2: Same-city overlap (count of overlaps)
            kpi_4_2 = sum(model.o_ct[g, t] for g in model.G for t in model.T)
            kpi_4_2_normalized = kpi_4_2 / norm_factors.get("kpi_4_2", 1.0)
            
            # KPI 5.2: Marquee-match overlap penalty
            kpi_5_2 = sum(model.o_P_mm_prime[m1, m2, t] for m1 in model.M for m2 in model.M for t in model.T)
            kpi_5_2_normalized = kpi_5_2 / norm_factors.get("kpi_5_2", 1.0)
            
            # KPI 5.3: Host-city economic equity
            kpi_5_3 = sum(model.p_s[s] for s in model.S)
            kpi_5_3_normalized = kpi_5_3 / norm_factors.get("kpi_5_3", 1.0)


            # Combine: now all KPIs are normalized to ~[0, scale] range
            return (direct_cost_normalized +
                    weights.get("kpi_1_6", 0.0) * kpi_1_6_normalized +
                    weights.get("kpi_3_3", 0.0) * kpi_3_3_normalized +
                    weights.get("kpi_4_1", 0.0) * kpi_4_1_normalized +
                    weights.get("kpi_4_2", 0.0) * kpi_4_2_normalized +
                    weights.get("kpi_5_2", 0.0) * kpi_5_2_normalized +
                    weights.get("kpi_5_3", 0.0) * kpi_5_3_normalized)

        model.obj = Objective(rule=objective_rule, sense=minimize)

        # Constraints for Auxiliary KPI Variables

        # KPI 1.6: Rest Asymmetry Between Opponents
        # delta_m >= |r_im[team_a, m] - r_im[team_b, m]|
        # Constraints: delta_m >= r_im[a] - r_im[b] and delta_m >= r_im[b] - r_im[a]
        def rest_asymmetry_constraint_1(model, m):
            match_row = self.matches[self.matches["match_id"] == m]
            if len(match_row) == 0:
                return Constraint.Skip
            match = match_row.iloc[0]
            team_a = match["team_a_id"]
            team_b = match["team_b_id"]
            return model.delta_m[m] >= model.r_im[team_a, m] - model.r_im[team_b, m]

        def rest_asymmetry_constraint_2(model, m):
            match_row = self.matches[self.matches["match_id"] == m]
            if len(match_row) == 0:
                return Constraint.Skip
            match = match_row.iloc[0]
            team_a = match["team_a_id"]
            team_b = match["team_b_id"]
            return model.delta_m[m] >= model.r_im[team_b, m] - model.r_im[team_a, m]

        model.h_kpi_1_6_a = Constraint(model.M, rule=rest_asymmetry_constraint_1)
        model.h_kpi_1_6_b = Constraint(model.M, rule=rest_asymmetry_constraint_2)

        # Rest computation: r_im[i,m] = hours of rest before match m for team i
        # For each team and pair of their matches (m_prev, m), if m_prev happens before m:
        # r_im[i,m] >= time_between(t_prev, t) - match_duration - Big_M*(2 - x[m_prev,t_prev,s_prev] - x[m,t,s])
        match_duration = self.params["match_duration"]
        big_m = 7 * 24  # Max 7 days between group stage matches
        
        rest_constraint_idx = 0
        rest_constraints = {}
        
        for i in self.I:
            team_matches = list(self.M_i.get(i, []))
            # For each pair of matches this team plays
            for m_prev in team_matches:
                for m in team_matches:
                    if m_prev == m:
                        continue
                    
                    # For each pair of time slots
                    for t_prev in model.T:
                        for s_prev in model.S:
                            for t in model.T:
                                for s in model.S:
                                    # Compute time difference
                                    try:
                                        date_prev, time_prev = self.slot_index_to_datetime[t_prev]
                                        date_curr, time_curr = self.slot_index_to_datetime[t]
                                        
                                        dt_prev = datetime.strptime(f"{date_prev} {time_prev}", "%Y-%m-%d %H:%M")
                                        dt_curr = datetime.strptime(f"{date_curr} {time_curr}", "%Y-%m-%d %H:%M")
                                        time_diff_hours = (dt_curr - dt_prev).total_seconds() / 3600.0
                                        rest_hours = time_diff_hours - match_duration
                                        
                                        # Only for positive rest (m is after m_prev)
                                        if rest_hours > 0:
                                            rest_constraint_idx += 1
                                            constraint_key = (i, m, rest_constraint_idx)
                                            rest_constraints[constraint_key] = (
                                                model.r_im[i, m] + 
                                                big_m * (2 - model.x[m_prev, t_prev, s_prev] - model.x[m, t, s])
                                                >= rest_hours
                                            )
                                    except:
                                        continue
        
        # Add all rest constraints to model
        for idx, (key, constraint_expr) in enumerate(rest_constraints.items()):
            setattr(model, f"h_kpi_rest_{idx}", Constraint(expr=constraint_expr))

        # ---------------------------------------------------------------------------------------

        # KPI 3.3: First-Mover Balance
        # e_i >= a_i - 1 and e_i >= 1 - a_i (where a_i is count of early kickoffs)
        # Simplified: penalize deviation from equal distribution across teams
        # def first_mover_constraint_1(model, i):
        #     team_matches = list(self.M_i.get(i, []))
        #     # Count early matches for this team (kickoff before 15:00)
        #     early_count = sum(
        #         model.x[m, t, s] 
        #         for m in team_matches 
        #         for t in model.T 
        #         for s in model.S
        #         if t < len(self.date_time_slots) and float(self.date_time_slots[t][1].split(":")[0]) < 15.0
        #     )
        #     return model.e_i[i] >= early_count - 1

        # def first_mover_constraint_2(model, i):
        #     team_matches = list(self.M_i.get(i, []))
        #     if len(team_matches) == 0:
        #         return Constraint.Skip
        #     early_count = sum(
        #         model.x[m, t, s]
        #         for m in team_matches
        #         for t in model.T
        #         for s in model.S
        #         if t < len(self.date_time_slots) and float(self.date_time_slots[t][1].split(":")[0]) < 15.0
        #     )
        #     return model.e_i[i] >= 1 - early_count

        # model.h_kpi_3_3_a = Constraint(model.I, rule=first_mover_constraint_1)
        # model.h_kpi_3_3_b = Constraint(model.I, rule=first_mover_constraint_2)

        # ---------------------------------------------------------------------------------------

        # KPI 4.1: Venue-Load Balance
        # venue_count[s] = sum of matches at stadium s
        # d_s >= venue_count[s] - mean_count and d_s >= mean_count - venue_count[s]
        def venue_count_constraint(model, s):
            return model.venue_count[s] == sum(model.x[m, t, s] for m in model.M for t in model.T)

        mean_venue_count = len(self.M) / len(self.S)
        
        def venue_load_deviation_1(model, s):
            return model.d_s[s] >= model.venue_count[s] - mean_venue_count

        def venue_load_deviation_2(model, s):
            return model.d_s[s] >= mean_venue_count - model.venue_count[s]

        model.h_kpi_4_1_count = Constraint(model.S, rule=venue_count_constraint)
        model.h_kpi_4_1_dev_1 = Constraint(model.S, rule=venue_load_deviation_1)
        model.h_kpi_4_1_dev_2 = Constraint(model.S, rule=venue_load_deviation_2)

        # ---------------------------------------------------------------------------------------

        # KPI 4.2: Same-City Overlap
        # o_ct[c, t] >= count of matches in city c at slot t minus 1 (or 0)
        # Precompute city mappings
        city_stadiums = {}  # city -> list of stadiums in that city
        for _, venue in self.venues.iterrows():
            city = venue.get("city", "Unknown")
            stadium_id = venue["venue_id"]
            if city not in city_stadiums:
                city_stadiums[city] = []
            city_stadiums[city].append(stadium_id)
        
        def same_city_overlap_constraint(model, g, t):
            # For each city and time slot, count simultaneous matches
            overlap_penalty = 0
            for city, stadiums in city_stadiums.items():
                city_match_count = sum(
                    model.x[m, t, s]
                    for m in model.M
                    for s in stadiums
                )
            
            return model.o_ct[g, t] >= city_match_count - 1  # Penalty if more than 1 match in same city at same time

        model.h_kpi_4_2 = Constraint(model.G, model.T, rule=same_city_overlap_constraint)

        # ---------------------------------------------------------------------------------------

        # KPI 5.2: Marquee-Match Slot Quality with Overlap Penalty
        # o_P_mm_prime[m, m', t] >= x[m,t,s1] + x[m',t,s2] - 1 (overlap if both in same slot)
        def marquee_overlap_constraint(model, m1, m2, t):
            if m1 >= m2:
                return Constraint.Skip
            # Penalty for two high-profile matches sharing a slot
            mu_m1 = self.params.get("match_value", {}).get(m1, 0.5)
            mu_m2 = self.params.get("match_value", {}).get(m2, 0.5)
            # Only penalize if both are at top 20
            threshold = np.quantile(list(self.params["match_value"].values()), 0.8)
            if mu_m1 >= threshold and mu_m2 >= threshold:
                return model.o_P_mm_prime[m1, m2, t] >= (
                    sum(model.x[m1, t, s] for s in model.S) +
                    sum(model.x[m2, t, s] for s in model.S) - 1
                )
            return Constraint.Skip

        model.h_kpi_5_2 = Constraint(model.M, model.M, model.T, rule=marquee_overlap_constraint)

        # ---------------------------------------------------------------------------------------

        # KPI 5.3: Host-City Economic Equity
        # Compute venue commercial value: vc_s = Σ_{m:v(m)=s} μ_m * q_t
        # where q_t = 1.0 if primetime (19-23), 0 otherwise
        # Then p_s >= |vc_s - mean_vc|
        
        # Precompute match values and primetime indicators
        match_values = {}  # match_id -> value
        primetime_indicator = {}  # (m, t) -> 1 if primetime, 0 otherwise
        
        for m in self.M:
            match_values[m] = self.params.get("match_value", {}).get(m, 0.5)
        
        for t_idx, (date, time_str) in enumerate(self.date_time_slots):
            try:
                kickoff_hour = float(time_str.split(":")[0])
                is_primetime = 1.0 if 19.0 <= kickoff_hour <= 23.0 else 0.0
            except:
                is_primetime = 0.0
            
            for m in self.M:
                primetime_indicator[(m, t_idx)] = is_primetime
        
        # Auxiliary variables for venue commercial value
        model.vc_s = Var(model.S, within=NonNegativeReals)  # Venue commercial value
        
        # Constraint: vc_s[s] = Σ_{m,t} x[m,t,s] * μ_m * q_t (match value * primetime bonus)
        def venue_commercial_constraint(model, s):
            return model.vc_s[s] == sum(
                model.x[m, t, s] * match_values[m] * primetime_indicator[(m, t)] * self.params["popularity"][m]
                for m in model.M
                for t in model.T
            )
        
        model.h_kpi_5_3_vc = Constraint(model.S, rule=venue_commercial_constraint)
        
        # Mean commercial value across venues
        mean_venue_commercial = (
            sum(match_values[m] * self.params["popularity"][m] for m in self.M) / len(self.S)
            if len(self.S) > 0 else 1.0
        )
        
        # Constraint: p_s >= |vc_s[s] - mean_vc| (mean absolute deviation)
        def economic_equity_constraint_1(model, s):
            return model.p_s[s] >= model.vc_s[s] - mean_venue_commercial

        def economic_equity_constraint_2(model, s):
            return model.p_s[s] >= mean_venue_commercial - model.vc_s[s]

        model.h_kpi_5_3_dev_1 = Constraint(model.S, rule=economic_equity_constraint_1)
        model.h_kpi_5_3_dev_2 = Constraint(model.S, rule=economic_equity_constraint_2)

        # =====================================================================================

        # Constraints

        # H1: Each match scheduled exactly once
        def h1_rule(model, m):
            return sum(model.x[m, t, s] for t in model.T for s in model.S) == 1

        model.h1 = Constraint(model.M, rule=h1_rule, doc="H1: Each match once")

        # H2: Round-robin (each team plays 3 matches)
        def h2_rule(model, i):
            team_matches = list(self.M_i.get(i, []))
            return (
                sum(
                    model.x[m, t, s]
                    for m in team_matches
                    for t in model.T
                    for s in model.S
                )
                == 3
            )

        model.h2 = Constraint(model.I, rule=h2_rule, doc="H2: Round-robin")

        # H4: stadium turnover
        def h4_rule(model, s, t):
            time_window = self.params["delta_min"] + self.params["match_duration"] 
            date_str, time_str = self.slot_index_to_datetime[t]

            dt = datetime.strptime(
                f"{date_str} {time_str}",
                "%Y-%m-%d %H:%M"
            )
            dt_later = dt + timedelta(hours=time_window)

            new_date = dt_later.strftime("%Y-%m-%d")
            new_time = dt_later.strftime("%H:%M")

            # Find all time slots that fall within the window
            relevant_slots = [
                t_prime for t_prime, dt_prime in self.slot_index_to_datetime.items()
                if dt_prime[0] <= new_date and dt_prime[1] <= new_time and dt_prime[0] > date_str and dt_prime[1] > time_str
            ]
            
            return sum(model.x[m, t_prime, s] for m in model.M for t_prime in relevant_slots) <= 1
        
        model.h4 = Constraint(model.S, model.T, rule=h4_rule, doc="H4: Stadium turnover")


        # H5: minimum team rest
        def h5_rule(model, i, m1, m2, t1, t2):
            if m1 >= m2:
                return Constraint.Skip  # Avoid double counting pairs

            team_matches = list(self.M_i.get(i, []))
            if m1 not in team_matches or m2 not in team_matches:
                return Constraint.Skip  # Only consider matches involving team i

            time_window = self.params["R_min"] + self.params["match_duration"]
            date_str_1, time_str_1 = self.slot_index_to_datetime[t1]
            date_str_2, time_str_2 = self.slot_index_to_datetime[t2]

            dt1 = datetime.strptime(f"{date_str_1} {time_str_1}", "%Y-%m-%d %H:%M")
            dt2 = datetime.strptime(f"{date_str_2} {time_str_2}", "%Y-%m-%d %H:%M")

            time_diff_hours = abs((dt2 - dt1).total_seconds()) / 3600.0

            if time_diff_hours < time_window:
                return (
                    sum(model.x[m1, t1, s] for s in model.S) +
                    sum(model.x[m2, t2, s] for s in model.S)
                    <= 1
                )
            else:
                return Constraint.Skip  # No constraint needed if time difference is sufficient
            
        model.h5 = Constraint(model.I, model.M, model.M, model.T, model.T, rule=h5_rule, doc="H5: Minimum team rest")

        # H6: host-nation matches in their country
        def h6_rule(model, m, t, s):
            match_row = self.matches[self.matches["match_id"] == m]
            match = match_row.iloc[0]

            team_a_id = match["team_a_id"]
            team_b_id = match["team_b_id"]

            team_a_country = self.teams[self.teams["team_id"] == team_a_id]["country"].values[0]
            team_b_country = self.teams[self.teams["team_id"] == team_b_id]["country"].values[0]

            stadium_country = self.venues[self.venues["venue_id"] == s]["country"].values[0]

            if (team_a_country in ["USA", "MEX", "CAN"] and team_a_country != stadium_country) or (team_b_country in ["USA", "MEX", "CAN"] and team_b_country != stadium_country):
                return model.x[m, t, s] == 0    
            
            return Constraint.Skip  # No constraint needed if no host nation involved
    
        model.h6 = Constraint(model.M, model.T, model.S, rule=h6_rule, doc="H6: Host nation matches")
       
        # H7: Simultaneous final matches
        def h7a_rule(model, g):
            return sum(model.y[g, t] for t in model.T) == 1

        model.h7a = Constraint(model.G, rule=h7a_rule, doc="H7a: Final slot chosen")

        def h7b_rule(model, g, t):
            group_matches = list(self.M_g.get(g, []))
            return (
                sum(model.x[m, t, s] for m in group_matches for s in model.S)
                >= 2 * model.y[g, t]
            )

        model.h7b = Constraint(
            model.G, model.T, rule=h7b_rule, doc="H7b: Final matches in slot"
        )

        def h7c_rule(model, g, t):
            group_matches = list(self.M_g.get(g, []))
            date_str, time_str = self.slot_index_to_datetime[t]

            later_slots = [
                t_prime for t_prime, dt_prime in self.slot_index_to_datetime.items()
                if dt_prime[0] > date_str or (dt_prime[0] == date_str and dt_prime[1] > time_str)
            ]

            return (
                sum(model.x[m, t_prime, s] for m in group_matches for s in model.S for t_prime in later_slots)
                <= 2 * (1 - model.y[g, t])
            )
        model.h7c = Constraint(
            model.G, model.T, rule=h7c_rule, doc="H7c: Final matches not after final slot"
        )

        # H8: Match allocation by country
        def h8_rule(model, c):
            if c == "USA":
                country_stadiums = list(self.S_c.get("USA", []))
            elif c == "MEX":
                country_stadiums = list(self.S_c.get("MEX", []))
            else:  # CAN
                country_stadiums = list(self.S_c.get("CAN", []))

            return (
                sum(
                    model.x[m, t, s]
                    for m in model.M
                    for t in model.T
                    for s in country_stadiums
                )
                == model.N_c[c]
            )

        model.h8 = Constraint(["USA", "MEX", "CAN"], rule=h8_rule, doc="H8: Country allocation")

        self.model = model
        return model

    def solve(self, time_limit: int = 300, solver_name: str = "glpk") -> Dict:
        """
        Solve the schedule optimization problem.
        Args:
            time_limit: Maximum solver time in seconds
            solver_name: Solver to use (glpk, cbc, gurobi, ipopt, etc.)
        Returns:
            Dictionary with solution details
        """
        if self.model is None:
            self.build_model()

        # Try to use the requested solver, fall back to available solvers
        solver_candidates = [solver_name, "cbc", "glpk", "ipopt", "gurobi"]
        solver = None
        used_solver = None
        
        for candidate in solver_candidates:
            try:
                solver = SolverFactory(candidate)
                # Check if solver is available
                if solver.available():
                    used_solver = candidate
                    print(f"  Using solver: {candidate}")
                    break
            except:
                continue

        # Configure solver options
        if used_solver == "cbc":
            solver.options["timeLimit"] = time_limit
        elif used_solver == "glpk":
            solver.options["tmlim"] = time_limit
        elif used_solver == "gurobi":
            solver.options["TimeLimit"] = time_limit
        
        result = solver.solve(self.model, tee=True)

        # Extract solution
        schedule = {}
        for m in self.model.M:
            for t in self.model.T:
                for s in self.model.S:
                    if value(self.model.x[m, t, s]) > 0.5:
                        schedule[int(m)] = (int(t), s)

        return {
            "status": str(result.solver.status),
            "objective": value(self.model.obj),
            "schedule": schedule,
            "model": self.model,
            "solver": used_solver,
        }
