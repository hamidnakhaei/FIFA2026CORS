"""
Schedule optimization solver (Step A) for FIFA 2026 Group-Stage.
Solves the MILP to assign matches to slots and stadiums, minimizing weighted KPIs.
"""

from datetime import datetime
from typing import Dict
from pyomo.environ import (
    ConcreteModel,
    Set,
    ConstraintList,
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
    """Solve Step A: optimize match schedule (teams have no fixed base camps)."""

    def __init__(self, data_loader, kpi_calculator):
        self.loader = data_loader
        self.kpi_calc = kpi_calculator

        self.matches = data_loader.get_matches()
        self.venues = data_loader.get_venues()
        self.teams = data_loader.get_teams()

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
        ban_teams = set(self.params["us_visa_ban_teams"])
        bond_teams = set(self.params["us_visa_bond_teams"])
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
        total_match_value = sum(self.params["match_value"].values())
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
        
        This method only computes KPI 1.7 and 2.2.
        """
        kpi_costs = {}
        weights = self.params["weights"]
        
        # for m in self.M:
        #     match_row = self.matches[self.matches["match_id"] == m]
        #     match = match_row.iloc[0]
            
        #     # team_a_id = match["team_a_id"]
        #     # team_b_id = match["team_b_id"]
            
        for (date, time_str) in self.T:
            for s in self.S:
                cost = 0.0
                    
                    # KPI 1.7: Entry and visa restriction exposure
                    # Penalty if match at US stadium and team has visa issues
                    # us_stadiums = self.venues[self.venues["country"] == "USA"]["venue_id"].tolist()
                    # if s in us_stadiums:
                    #     ban_teams = set(self.params.get("us_visa_ban_teams", []))
                    #     bond_teams = set(self.params.get("us_visa_bond_teams", []))
                        
                    #     for team_id in [team_a_id, team_b_id]:
                    #         if team_id in ban_teams:
                    #             cost += 1.0 * weights.get("kpi_1_7", 0.0)
                    #         elif team_id in bond_teams:
                    #             cost += 0.5 * weights.get("kpi_1_7", 0.0)
                    
                    # KPI 2.2: Per-team heat load
                    # Excess WBGT: h_mt = max(0, WBGT - 28)
                venue_weather = self.params["weather"]
                if isinstance(venue_weather, dict):
                    datetime_str = f"{date} {time_str}"
                    temp_c = venue_weather[s].get(datetime_str, 20.0)
                else:
                    temp_c = 20.0
                
                wbgt_estimated = temp_c
                excess_wbgt = max(0.0, wbgt_estimated - 28.0)
                cost += excess_wbgt * 2.0 
                
                kpi_costs[(date, time_str, s)] = cost
                
        return kpi_costs

    def build_model(self) -> ConcreteModel:
        """Build the MILP model for schedule optimization."""
        model = ConcreteModel()

        # Sets
        model.M = Set(initialize=list(self.M))  # Matches
        model.T = Set(initialize=list(self.T))  # Time slots (date/time combinations)
        model.S = Set(initialize=list(self.S))  # Stadiums
        model.I = Set(initialize=list(self.I))  # Teams
        model.G = Set(initialize=list(self.G))  # Groups
        model.M_i = Set(model.I, ordered=False, initialize=lambda m, i: self.M_i[i])  # Matches per team
        model.IKS = Set(
                    dimen=3,
                    initialize=lambda m: (
                        (i, k, s)
                        for i in m.I
                        for k in model.M_i[i]
                        for s in m.S
                    )
                )
        model.IKKSS = Set(
                    dimen=5,
                    initialize=lambda m: (
                        (i, k1, k2, s1, s2)
                        for i in m.I
                        for k1 in model.M_i[i]
                        for k2 in model.M_i[i]
                        for s1 in m.S
                        for s2 in m.S
                    )
                )
                    
        # Parameters
        model.N_c = Param(
            ["USA", "MEX", "CAN"], initialize=self.params["N_c"]
        )  # Matches per country requirement (indexed)

        # Precompute KPI coefficients for each (match, time_slot, stadium) for KPI 1.7 and 2.2
        kpi_coefficients = self._compute_kpi_coefficients()
        model.kpi_cost = Param(
            model.T,
            model.S,
            initialize={(t, s): kpi_coefficients[(t[0], t[1], s)]
                        for t in model.T for s in model.S},
        )

        city_stadiums = {}  # city -> list of stadiums in that city
        for _, venue in self.venues.iterrows():
            city = venue["city"]
            stadium_id = venue["venue_id"]
            if city not in city_stadiums:
                city_stadiums[city] = []
            city_stadiums[city].append(stadium_id)

        model.C = Set(initialize=list(city_stadiums.keys()))  # Cities as a set

        ################################################################################## Vars

        # Decision Variables
        model.x = Var(model.M, model.T, model.S, within=Binary)  # Assignment vars
        model.y = Var(model.G, model.T, within=Binary)  # Final slot indicator

        # ===== AUXILIARY VARIABLES FOR INTER-STADIUM METRICS =====
        
        # Stadium assignment for each team's match position
        # For team i's kth match (k=0,1,2), which stadium is it at?
        # stadium_i_k ∈ S (select exactly one stadium)
        model.stadium_i_k = Var(model.IKS, within=Binary)
        # Binary transition indicators for inter-stadium costs
        # For team i between match positions k and k', is it playing at (s1, s2)?
        model.transition_i_k1_k2_s1_s2 = Var(model.IKKSS, within=Binary)
        # Continuous travel distance for each team (KPI 1.2)
        model.travel_distance_i = Var(model.I, within=NonNegativeReals)
        # Cumulative timezone offset for each match position (KPI 1.3)
        # tz_offset_i_k = cumulative tz from team i's first stadium to kth stadium
        # model.tz_offset_i = Var(model.I, within=NonNegativeReals, bounds=(0, 24))
        # Country transition count for each team (KPI 1.4)
        # model.country_transitions_i = Var(model.I, within=NonNegativeIntegers, bounds=(0, 2))

        # ===== AUXILIARY VARIABLES FOR OTHER KPIS =====
        # model.delta_m = Var(model.M, within=NonNegativeReals)  # KPI 1.6: Rest asymmetry per match
        # model.r_im = Var(model.I, model.M, within=NonNegativeReals)  # KPI 1.6: Rest days for team i before match m
        # model.e_i = Var(model.I, within=NonNegativeReals)  # KPI 3.3: First-mover balance per team
        model.d_s = Var(model.S, within=NonNegativeReals)  # KPI 4.1: Venue load deviation
        model.venue_count = Var(model.S, within=NonNegativeIntegers)  # KPI 4.1: Number of matches per venue
        # model.o_ct = Var(model.C, model.T, within=NonNegativeReals)  # KPI 4.2: Same-city overlap
        # model.o_P_mm_prime = Var(model.M, model.M, model.T, within=NonNegativeReals)  # KPI 5.2: Marquee match overlap
        # model.p_s = Var(model.S, within=NonNegativeReals)  # KPI 5.3: Host-city economic equity deviation
        # model.vc_s = Var(model.S, within=NonNegativeReals)  # KPI 5.3: Venue commercial value

        ################################################################################## Cons

        # ===== CONSTRAINTS FOR INTER-STADIUM METRICS (KPI 1.2, 1.3, 1.4) =====
        
        # C1: Link stadium assignment to schedule
        # For team i's kth match: exactly one stadium, and it must match the schedule assignment
        model.stadium_assign = ConstraintList()
        for i in model.I:
            for k in model.M_i[i]:
                model.stadium_assign.add(sum(model.stadium_i_k[i, k, s] for s in model.S) == 1)
        
        # C2: Stadium assignment must match schedule
        # If x[m_i_k, t, s] = 1, then stadium_i_k[i, k, s] = 1
        model.stadium_schedule_link = ConstraintList()
        for i in model.I:
            for k in model.M_i[i]:
                for s in model.S:
                    model.stadium_schedule_link.add(model.stadium_i_k[i, k, s] == sum(model.x[k, t, s] for t in model.T))
        
        # C3: Inter-stadium transitions (for computing travel, timezone, country costs)
        # For team i between matches k and k+1, exactly one (s1, s2) pair is active
        model.transition_assign = ConstraintList()
        for i in model.I:
            for k1 in model.M_i[i]:
                for k2 in model.M_i[i]:
                    if k1 >= k2:
                        continue
                    model.transition_assign.add(sum(model.transition_i_k1_k2_s1_s2[i, k1, k2, s1, s2] for s1 in model.S for s2 in model.S) == 1)

       # C4: Transition must match stadium assignments
        # transition_i_k1_k2_s1_s2 = 1 only if stadium_i_k[i, k1, s1] = 1 AND stadium_i_k[i, k2, s2] = 1
        model.transition_link = ConstraintList()
        for i in model.I:
            for k1 in model.M_i[i]:
                for k2 in model.M_i[i]:
                    if k1 >= k2:
                        continue
                    for s1 in model.S:
                        for s2 in model.S:
                            model.transition_link.add(model.transition_i_k1_k2_s1_s2[i, k1, k2, s1, s2] <= model.stadium_i_k[i, k1, s1])
                            model.transition_link.add(model.transition_i_k1_k2_s1_s2[i, k1, k2, s1, s2] <= model.stadium_i_k[i, k2, s2])

        # C5: KPI 1.2 - Travel distance (sum of inter-stadium distances)
        # travel_i = Σ_{s1,s2} transition_i_0_s1_s2 * dist(s1, s2) + transition_i_1_s1_s2 * dist(s1, s2)
        def travel_distance_calc(model, i):
            """Compute total travel distance for team i (inter-stadium, no round-trip factor)."""
            dist_dict = self.params["dist_v_v"]  # Precomputed stadium-to-stadium distances
            travel = 0.0
            for k1 in model.M_i[i]:  
                for k2 in model.M_i[i]:
                    if k1 >= k2:
                        continue
                    for s1 in model.S:
                        for s2 in model.S:
                            d = dist_dict[s1, s2]
                            travel += model.transition_i_k1_k2_s1_s2[i, k1, k2, s1, s2] * d
            return model.travel_distance_i[i] == travel
        
        model.c_travel_distance = Constraint(model.I, rule=travel_distance_calc)
        
        # C6: KPI 1.3 - Cumulative jet-lag (timezone offset)
        # def tz_offset_calc(model, i):
        #     """Cumulative timezone offset for team i's kth match."""
        #     tz_stadium = self.params.get("tzone_stadium", {})
            
        #     # Later matches: cumulative offset from first stadium
        #     # tz_offset_i_k = tz(stadium_i_k) - tz(stadium_i_0)
        #     # Determine s_k and s_0 from stadium assignments and compute offset
        #     expr = 0
        #     for s_0 in model.S:
        #         for s_k in model.S:
        #             tz_0 = tz_stadium.get(s_0, 0)
        #             tz_k = tz_stadium.get(s_k, 0)
        #             tz_diff = tz_k - tz_0
        #             # Activate this term if both s_0 and s_k are assigned
        #             expr += tz_diff * sum(model.transition_i_k1_k2_s1_s2[i, k1, k2, s_0, s_k] for k1 in model.M_i[i] for k2 in model.M_i[i] if k1 < k2)
        #     return model.tz_offset_i[i] == expr
        
        # model.c_tz_offset = Constraint(model.I, rule=tz_offset_calc)
        
        # C7: KPI 1.4 - Country transitions
        # Count how many times team i crosses a country border between consecutive stadiums
        # def country_transitions_calc(model, i):
        #     """Count inter-stadium country transitions for team i."""
        #     stadium_country = {}
        #     for _, venue in self.venues.iterrows():
        #         stadium_country[venue["venue_id"]] = venue.get("country", "Unknown")
            
        #     transitions = 0.0
        #     for k1 in model.M_i[i]:  
        #         for k2 in model.M_i[i]:
        #             if k1 >= k2:
        #                 continue
        #             for s1 in model.S:
        #                 for s2 in model.S:
        #                     c1 = stadium_country.get(s1, "Unknown")
        #                     c2 = stadium_country.get(s2, "Unknown")
        #                     # Add 1 to transitions count if countries differ and this transition is active
        #                     if c1 != c2:
        #                         transitions += model.transition_i_k1_k2_s1_s2[i, k1, k2, s1, s2]
            
        #     return model.country_transitions_i[i] == transitions
        
        # model.c_country_transitions = Constraint(model.I, rule=country_transitions_calc)

        # KPI 1.6: Rest Asymmetry Between Opponents
        # delta_m >= |r_im[team_a, m] - r_im[team_b, m]|
        # Constraints: delta_m >= r_im[a] - r_im[b] and delta_m >= r_im[b] - r_im[a]
        # def rest_asymmetry_constraint_1(model, m):
        #     match_row = self.matches[self.matches["match_id"] == m]
        #     if len(match_row) == 0:
        #         return Constraint.Skip
        #     match = match_row.iloc[0]
        #     team_a = match["team_a_id"]
        #     team_b = match["team_b_id"]
        #     return model.delta_m[m] >= model.r_im[team_a, m] - model.r_im[team_b, m]

        # def rest_asymmetry_constraint_2(model, m):
        #     match_row = self.matches[self.matches["match_id"] == m]
        #     if len(match_row) == 0:
        #         return Constraint.Skip
        #     match = match_row.iloc[0]
        #     team_a = match["team_a_id"]
        #     team_b = match["team_b_id"]
        #     return model.delta_m[m] >= model.r_im[team_b, m] - model.r_im[team_a, m]

        # model.h_kpi_1_6_a = Constraint(model.M, rule=rest_asymmetry_constraint_1)
        # model.h_kpi_1_6_b = Constraint(model.M, rule=rest_asymmetry_constraint_2)

        # Rest computation: r_im[i,m] = hours of rest before match m for team i
        # For each team and pair of their matches (m_prev, m), if m_prev happens before m:
        # r_im[i,m] >= time_between(t_prev, t) - match_duration - Big_M*(2 - x[m_prev,t_prev,s_prev] - x[m,t,s])
        # match_duration = self.params["match_duration"]
        # big_m = 7 * 24  # Max 7 days between group stage matches
        
        # rest_constraint_idx = 0
        # rest_constraints = {}
        
        # for i in model.I:
        #     team_matches = list(model.M_i[i])
        #     # For each pair of matches this team plays
        #     for m_prev in team_matches:
        #         for m in team_matches:
        #             if m_prev == m:
        #                 continue
                    
        #             # For each pair of time slots
        #             for t_prev in model.T:
        #                 for s_prev in model.S:
        #                     for t in model.T:
        #                         for s in model.S:
        #                             # Compute time difference
        #                             try:
        #                                 date_prev, time_prev = self.slot_index_to_datetime[t_prev]
        #                                 date_curr, time_curr = self.slot_index_to_datetime[t]
                                        
        #                                 dt_prev = datetime.strptime(f"{date_prev} {time_prev}", "%Y-%m-%d %H:%M")
        #                                 dt_curr = datetime.strptime(f"{date_curr} {time_curr}", "%Y-%m-%d %H:%M")
        #                                 time_diff_hours = (dt_curr - dt_prev).total_seconds() / 3600.0
        #                                 rest_hours = time_diff_hours - match_duration
                                        
        #                                 # Only for positive rest (m is after m_prev)
        #                                 if rest_hours > 0:
        #                                     rest_constraint_idx += 1
        #                                     constraint_key = (i, m, rest_constraint_idx)
        #                                     rest_constraints[constraint_key] = (
        #                                         model.r_im[i, m] + 
        #                                         big_m * (2 - model.x[m_prev, t_prev, s_prev] - model.x[m, t, s])
        #                                         >= rest_hours
        #                                     )
        #                             except:
        #                                 continue
        
        # # Add all rest constraints to model
        # for idx, (key, constraint_expr) in enumerate(rest_constraints.items()):
        #     setattr(model, f"h_kpi_rest_{idx}", Constraint(expr=constraint_expr))

        # ---------------------------------------------------------------------------------------

        # KPI 4.1: Venue-Load Balance
        # venue_count[s] = sum of matches at stadium s
        # d_s >= venue_count[s] - mean_count and d_s >= mean_count - venue_count[s]
        def venue_count_constraint(model, s):
            return model.venue_count[s] == sum(model.x[m, t, s] for m in model.M for t in model.T)

        mean_venue_count = len(model.M) / len(model.S)
        
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
        
        # def same_city_overlap_constraint(model, c, date, time):
        #     t = (date, time)
        #     # For each city and time slot, count simultaneous matches
        #     city_match_count = sum(
        #             model.x[m, t, s]
        #             for m in model.M
        #             for s in city_stadiums[c]
        #         )
            
        #     return model.o_ct[c, t] >= city_match_count - 1  # Penalty if more than 1 match in same city at same time

        # model.h_kpi_4_2 = Constraint(model.C, model.T, rule=same_city_overlap_constraint)

        # ---------------------------------------------------------------------------------------

        # KPI 5.2: Marquee-Match Slot Quality with Overlap Penalty
        # o_P_mm_prime[m, m', t] >= x[m,t,s1] + x[m',t,s2] - 1 (overlap if both in same slot)
        # def marquee_overlap_constraint(model, m1, m2, date, time):
        #     t = (date, time)
        #     if m1 >= m2:
        #         return Constraint.Skip
        #     # Penalty for two high-profile matches sharing a slot
        #     mu_m1 = self.params.get("match_value", {}).get(m1, 0.5)
        #     mu_m2 = self.params.get("match_value", {}).get(m2, 0.5)
        #     # Only penalize if both are at top 20
        #     threshold = np.quantile(list(self.params["match_value"].values()), 0.8)
        #     if mu_m1 >= threshold and mu_m2 >= threshold:
        #         return model.o_P_mm_prime[m1, m2, t] >= (
        #             sum(model.x[m1, t, s] for s in model.S) +
        #             sum(model.x[m2, t, s] for s in model.S) - 1
        #         )
        #     return Constraint.Skip

        # model.h_kpi_5_2 = Constraint(model.M, model.M, model.T, rule=marquee_overlap_constraint)

        # ---------------------------------------------------------------------------------------

        # KPI 5.3: Host-City Economic Equity
        # Compute venue commercial value: vc_s = Σ_{m:v(m)=s} μ_m * q_t
        # where q_t = 1.0 if primetime (19-23), 0 otherwise
        # Then p_s >= |vc_s - mean_vc|
        
        # Precompute match values and primetime indicators
        # match_values = {}  # match_id -> value
        # primetime_indicator = {}  # (m, t) -> 1 if primetime, 0 otherwise
        
        # for m in model.M:
        #     match_values[m] = self.params.get("match_value", {}).get(m, 0.5)
        
        # for (date, time_str) in model.T:
        #     kickoff_hour = float(time_str.split(":")[0])
        #     is_primetime = 1.0 if 19.0 <= kickoff_hour <= 23.0 else 0.0
            
        #     for m in model.M:
        #         primetime_indicator[(m, (date, time_str))] = is_primetime
        
        # # Constraint: vc_s[s] = Σ_{m,t} x[m,t,s] * μ_m * q_t (match value * primetime bonus)
        # def venue_commercial_constraint(model, s):
        #     return model.vc_s[s] == sum(
        #         model.x[m, t, s] * match_values[m] * primetime_indicator[(m, t)] * self.params["popularity"][m]
        #         for m in model.M
        #         for t in model.T
        #     )
        
        # model.h_kpi_5_3_vc = Constraint(model.S, rule=venue_commercial_constraint)
        
        # # Mean commercial value across venues
        # mean_venue_commercial = (
        #     sum(match_values[m] * self.params["popularity"][m] for m in model.M) / len(model.S)
        #     if len(model.S) > 0 else 1.0
        # )
        
        # # Constraint: p_s >= |vc_s[s] - mean_vc| (mean absolute deviation)
        # def economic_equity_constraint_1(model, s):
        #     return model.p_s[s] >= model.vc_s[s] - mean_venue_commercial

        # def economic_equity_constraint_2(model, s):
        #     return model.p_s[s] >= mean_venue_commercial - model.vc_s[s]

        # model.h_kpi_5_3_dev_1 = Constraint(model.S, rule=economic_equity_constraint_1)
        # model.h_kpi_5_3_dev_2 = Constraint(model.S, rule=economic_equity_constraint_2)

        # =====================================================================================

        # Hard Constraints

        # H1-1: Each match scheduled exactly once
        def h1_rule(model, m):
            return sum(model.x[m, t, s] for t in model.T for s in model.S) == 1

        model.h1 = Constraint(model.M, rule=h1_rule, doc="H1: Each match once")

        # H2: Round-robin (each team plays 3 matches)
        def h2_rule(model, i):
            team_matches = list(model.M_i[i])
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
        def h4_rule(model, s, date):
            # time_window = self.params["delta_min"] + self.params["match_duration"] 
            # date_str, time_str = date, time

            # dt = datetime.strptime(
            #     f"{date_str} {time_str}",
            #     "%Y-%m-%d %H:%M"
            # )
            # dt_later = dt + timedelta(hours=time_window)

            # # Find all time slots that fall within the window [dt, dt_later]
            # # Convert each slot to datetime and check if it's in the interval
            # relevant_slots = [
            #     t_prime for t_prime in model.T
            #     if dt <= datetime.strptime(f"{t_prime[0]} {t_prime[1]}", "%Y-%m-%d %H:%M") <= dt_later
            # ]

            # if not relevant_slots:
            #     return Constraint.Skip  # No slots in window, so no turnover constraint needed
            
            return (sum(model.x[m, t_prime, s] for m in model.M for t_prime in model.T if t_prime[0] == date) <= 1)
        
        model.h4 = Constraint(model.S, [t[0] for t in model.T], rule=h4_rule, doc="H4: Stadium turnover")
        
        # H5: minimum team rest
        model.H5 = ConstraintList()
        time_window = self.params["R_min"] + self.params["match_duration"]
        for i in model.I:
            for date1, time1 in model.T:
                for date2, time2 in model.T:
                    if (date1, time1) != (date2, time2):
                        if date1 > date2:
                            dt1 = datetime.strptime(f"{date1} {time1}", "%Y-%m-%d %H:%M")
                            dt2 = datetime.strptime(f"{date2} {time2}", "%Y-%m-%d %H:%M")
                            time_diff_hours = abs((dt2 - dt1).total_seconds()) / 3600.0

                            if time_diff_hours < time_window:
                                t1 = (date1, time1)
                                t2 = (date2, time2)
                                model.H5.add(sum(model.x[m1, t1, s] for m1 in model.M_i[i] for s in model.S) + sum(model.x[m2, t2, s] for m2 in model.M_i[i] for s in model.S)<= 1)
        
        # H6: host-nation matches in their country
        def h6_rule(model, team):

            return sum(model.x[m, t, s] for m in model.M for t in model.T for s in model.S if m in model.M_i[team] and s not in self.S_c[team]) == 0  # Ensure variables are created
            
        model.h6 = Constraint(["USA", "MEX", "CAN"], rule=h6_rule, doc="H6: Host nation matches")
        
        # H7: Simultaneous final matches
        def h7a_rule(model, g):
            return sum(model.y[g, t] for t in model.T) == 1

        model.h7a = Constraint(model.G, rule=h7a_rule, doc="H7a: Final slot chosen")
        
        def h7b_rule(model, g, date, time):
            t = (date, time)
            group_matches = list(self.M_g[g])
            return (
                sum(model.x[m, t, s] for m in group_matches for s in model.S)
                >= 2 * model.y[g, t]
            )

        model.h7b = Constraint(
            model.G, model.T, rule=h7b_rule, doc="H7b: Final matches in slot"
        )

        def h7c_rule(model, g, date, time):
            t = (date, time)
            group_matches = list(self.M_g[g])
            date_str, time_str = date, time

            later_slots = [
                t_prime for t_prime, dt_prime in model.T
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
                country_stadiums = list(self.S_c["USA"]) 
            elif c == "MEX":
                country_stadiums = list(self.S_c["MEX"])
            else:  # CAN
                country_stadiums = list(self.S_c["CAN"])

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
        
        ############################################################### Obj

        # Objective: Minimize full weighted KPI (all 13 KPIs)
        # Direct KPIs (1.7, 2.2) via precomputed coefficients
        # + Inter-stadium KPIs (1.2, 1.3, 1.4) via auxiliary constraints
        # + Auxiliary variable KPIs (1.6, 3.3, 4.1, 4.2, 5.2, 5.3)
        def objective_rule(model):
            weights = self.params["weights"]
            norm_factors = self.kpi_normalization_factors
            
            # KPI 1.7 + 2.2: Direct coefficient-based KPIs (from precomputed coefficients)
            direct_cost = sum(
                model.kpi_cost[t, s] * model.x[m, t, s]
                for m in model.M
                for t in model.T
                for s in model.S
            )
            direct_cost_norm = (
                weights["kpi_1_7"] * norm_factors["kpi_1_7"] +
                weights["kpi_2_2"] * norm_factors["kpi_2_2"]
            )
            direct_cost_normalized = (
                direct_cost / direct_cost_norm if direct_cost_norm > 0 else direct_cost
            )
            
            # # KPI 1.2: Travel distance (sum of inter-stadium distances)
            kpi_1_2 = sum(model.travel_distance_i[i] for i in model.I)
            kpi_1_2_normalized = kpi_1_2 / norm_factors["kpi_1_2"]
            
            # kpi_1_3 = sum(model.tz_offset_i[i] for i in model.I)
            # kpi_1_3_normalized = kpi_1_3 / norm_factors.get("kpi_1_3", 1.0)
            
            # # KPI 1.4: Country transitions (inter-stadium border crossings)
            # kpi_1_4 = sum(model.country_transitions_i[i] for i in model.I)
            # kpi_1_4_normalized = kpi_1_4 / norm_factors.get("kpi_1_4", 1.0)
            
            # # KPI 1.6: Rest asymmetry (sum of rest differences)
            # kpi_1_6 = sum(model.delta_m[m] for m in model.M)
            # kpi_1_6_normalized = kpi_1_6 / norm_factors.get("kpi_1_6", 1.0)
            
            # # KPI 4.1: Venue-load balance (mean absolute deviation of match counts)
            kpi_4_1 = sum(model.d_s[s] for s in model.S)
            kpi_4_1_normalized = kpi_4_1 / norm_factors.get("kpi_4_1", 1.0)
            
            # # KPI 4.2: Same-city overlap (count of overlaps)
            # kpi_4_2 = sum(model.o_ct[c, t] for c in model.C for t in model.T)
            # kpi_4_2_normalized = kpi_4_2 / norm_factors.get("kpi_4_2", 1.0)
            
            # # KPI 5.2: Marquee-match overlap penalty
            # kpi_5_2 = sum(model.o_P_mm_prime[m1, m2, t] for m1 in model.M for m2 in model.M for t in model.T)
            # kpi_5_2_normalized = kpi_5_2 / norm_factors.get("kpi_5_2", 1.0)
            
            # # KPI 5.3: Host-city economic equity
            # kpi_5_3 = sum(model.p_s[s] for s in model.S)
            # kpi_5_3_normalized = kpi_5_3 / norm_factors.get("kpi_5_3", 1.0)

            # Combine all normalized KPIs
            return (direct_cost_normalized +
                    weights["kpi_1_2"] * kpi_1_2_normalized+
                    weights["kpi_4_1"] * kpi_4_1_normalized)
                    # weights.get("kpi_1_3", 0.0) * kpi_1_3_normalized +
                    # weights.get("kpi_1_4", 0.0) * kpi_1_4_normalized +
                    # weights.get("kpi_1_6", 0.0) * kpi_1_6_normalized +
                    # weights.get("kpi_4_2", 0.0) * kpi_4_2_normalized +
                    # weights.get("kpi_5_2", 0.0) * kpi_5_2_normalized +
                    # weights.get("kpi_5_3", 0.0) * kpi_5_3_normalized)

            # return direct_cost_normalized  # For initial testing, focus on direct coefficient-based KPIs

        model.obj = Objective(rule=objective_rule, sense=minimize)

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
                        schedule[m] = (t, s)

        schedule_DF = []
        for m in self.model.M:
            for t in self.model.T:
                for s in self.model.S:
                    if value(self.model.x[m, t, s]) > 0.5:
                        schedule_DF.append((m, t, s))

        return {
            "status": str(result.solver.status),
            "objective": value(self.model.obj),
            "schedule": schedule,
            "schedule_df": schedule_DF,
            "model": self.model,
            "solver": used_solver,
        }
