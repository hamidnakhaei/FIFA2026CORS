"""
Schedule optimization solver (Step A) for FIFA 2026 Group-Stage.
Solves the MILP to assign matches to slots and stadiums, minimizing weighted KPIs.
"""

from collections import Counter
from datetime import date, timedelta, datetime
import pandas as pd
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

    def __init__(self, data_loader):
        self.loader = data_loader

        self.matches = data_loader.get_matches()
        self.venues = data_loader.get_venues()
        self.teams = data_loader.get_teams()

        # Get sets and indices
        (
            self.M,
            self.T,
            self.T_s,
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

    def evaluate_schedule(self, match_to_venue: dict) -> dict:
        """KPI components (model's definition) for any {match_id: venue_id} map."""
        dvv = self.params["dist_v_v"]
        team_travel = {}
        for i in self.I:
            vs = [match_to_venue[m] for m in self.M_i[i] if m in match_to_venue]
            team_travel[i] = 2.0 * sum(dvv[a, b]
                                    for j, a in enumerate(vs) for b in vs[j+1:])
        counts = Counter(match_to_venue.values())
        mean_c = len(self.M) / len(self.S)
        return {
            "sum_travel": sum(team_travel.values()),
            "max_travel": max(team_travel.values()),
            "avg_travel": sum(team_travel.values()) / len(team_travel),
            "venue_dev": sum(abs(counts.get(s, 0) - mean_c) for s in self.S),
        }

    def _baseline_rest_asymmetry(self) -> float:
        """Sum over matches of |rest_before(team_a) - rest_before(team_b)| in hours,
        on the original released schedule. Rest before a team's first match is 0."""
        matches = self.matches          # has the 'datetime' column from get_matches()
        md = self.params["match_duration"]

        rest_before = {}                # (team_id, match_id) -> hours of rest before m
        for i in self.I:
            tm = matches[(matches["team_a_id"] == i) |
                        (matches["team_b_id"] == i)].sort_values("datetime")
            prev_dt = None
            for _, row in tm.iterrows():
                if prev_dt is None:
                    rest_before[(i, row["match_id"])] = 0.0
                else:
                    hrs = (row["datetime"] - prev_dt).total_seconds() / 3600.0
                    rest_before[(i, row["match_id"])] = max(0.0, hrs - md)
                prev_dt = row["datetime"]

        total = 0.0
        for _, row in matches.iterrows():
            a = rest_before.get((row["team_a_id"], row["match_id"]), 0.0)
            b = rest_before.get((row["team_b_id"], row["match_id"]), 0.0)
            total += abs(a - b)
        return total

    def _compute_kpi_normalization_factors(self) -> Dict:
        """
        Compute reference/baseline values for each KPI to normalize them.
        KPI_normalized = KPI / reference_value makes all KPIs dimensionless and comparable.
        
        Returns dict mapping kpi_name -> reference_value
        """
        factors = {}
        
        # KPI 2.2: Heat load (WBGT hours per team)
        # Excess above 28°C, worst case ~10°C excess × 3 matches × 32 teams
        factors["kpi_2_2"] = max(10.0 * 3 * len(self.I), 0.1)
        
        # KPI 1.2: Travel dispersion (km)
        # Worst case: all teams travel max distance for all matches
        base = self.evaluate_schedule(
            {r["match_id"]: r["venue_id"] for _, r in self.matches.iterrows()}
        )
        factors["kpi_1_2"] = {"sum": max(base["sum_travel"], 0.1),
                            "max": max(base["max_travel"], 0.1)}
        
        # KPI 4.1: Venue-load balance (mean absolute deviation)
        # Worst case: unbalanced distribution of 72 matches across 12 stadiums
        factors["kpi_4_1"] = max(base["venue_dev"], 0.1)
        
        # KPI 1.6: Rest asymmetry (sum of rest differences)
        factors["kpi_1_6"] = max(self._baseline_rest_asymmetry(), 0.1)
               
        return factors
    
    def _compute_kpi_coefficients(self) -> Dict:
        """
        Precompute KPI contributions for each (match, time_slot, stadium) combination.
        Returns dict mapping (m, t, s) -> weighted_kpi_cost
        t is an index into date_time_slots.
        
        This method only computes KPI 1.7 and 2.2.
        """
        kpi_costs = {}
        
        for s in self.S:
            for (date_str, time_str) in self.T_s[s]:        
                # KPI 2.2: Per-team heat load
                # Excess WBGT: h_mt = max(0, WBGT - 28)
                venue_weather = self.params["weather"]
                if isinstance(venue_weather, dict):
                    datetime_str = f"{date_str} {time_str}"
                    temp_c = venue_weather[s].get(datetime_str, 20.0)
                else:
                    temp_c = 20.0
                
                wbgt_estimated = temp_c
                excess_wbgt = max(0.0, wbgt_estimated - 28.0)
                kpi_costs[(date_str, time_str, s)] = excess_wbgt * 2.0 
                
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
        model.T_s = Set(model.S, ordered=False, initialize=lambda m, s: self.T_s[s])  # Valid time slots per stadium
        
        # top_k = int(len(self.S) * (len(self.S) - 1) / 2 * 0.5)  # Top 50% of stadium pairs
        # self.top_20_dist_pairs = [
        #                     k for k, v in sorted(
        #                         self.params["dist_v_v"].items(),
        #                         key=lambda x: x[1],
        #                         reverse=True
        #                     )[:top_k]
        #                 ]
        model.ISS = Set(
                    dimen=3,
                    initialize=lambda m: (
                        (i, s1, s2)
                        for i in m.I
                        for s1 in m.S
                        for s2 in m.S
                        if s1 < s2  # Avoid duplicates and self-pairs)
                ))
        model.IS = Set(
                    dimen=2,
                    initialize=lambda m: (
                        (i, s1)
                        for i in m.I
                        for s1 in m.S
                    )
                )
        model.TS = Set(
                    dimen=3,
                    initialize=lambda m: (
                        (t, s)
                        for s in m.S
                        for t in m.T_s[s]
                    )
                )
                    
        # Parameters
        model.N_c = Param(
            ["USA", "MEX", "CAN"], initialize=self.params["N_c"]
        )  # Matches per country requirement (indexed)

        # Precompute KPI coefficients for each (match, time_slot, stadium) for KPI 1.7 and 2.2
        kpi_coefficients = self._compute_kpi_coefficients()
        model.kpi_cost = Param(
            model.TS,
            initialize={(t, s): kpi_coefficients[(t[0], t[1], s)]
                        for s in model.S for t in model.T_s[s]},
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
        model.x = Var(model.M, model.TS, within=Binary)  # Assignment vars
        model.y = Var(model.G, model.T, within=Binary)  # Final slot indicator

        # ===== AUXILIARY VARIABLES FOR INTER-STADIUM METRICS =====
        
        # Stadium assignment for each team's match position
        # For team i's kth match (k=0,1,2), which stadium is it at?
        # stadium_i_k ∈ S (select exactly one stadium)
        model.stadium_i_s = Var(model.IS, within=Binary)
        # Binary transition indicators for inter-stadium costs
        # For team i between match positions k and k', is it playing at (s1, s2)?
        model.transition_i_s1_s2 = Var(model.ISS, within=NonNegativeReals)  # Relaxed to continuous for better LP relaxation
        # Continuous travel distance for each team (KPI 1.2)
        # model.travel_distance_i = Var(model.I, within=NonNegativeReals)
        model.max_d = Var(within=NonNegativeReals)  # Max travel distance across teams (for KPI 1.2)
        
        # ===== AUXILIARY VARIABLES FOR OTHER KPIS =====
        model.d_s = Var(model.S, within=NonNegativeReals)  # KPI 4.1: Venue load deviation
        model.venue_count = Var(model.S, within=NonNegativeIntegers)  # KPI 4.1: Number of matches per venue
        model.delta_m = Var(model.M, within=NonNegativeReals)  # KPI 1.6: Rest asymmetry per match
        model.r_im = Var(model.I, model.M, within=NonNegativeReals)  # KPI 1.6: Rest days for team i before match m
        
        
        ################################################################################## Cons

        # ===== CONSTRAINTS FOR INTER-STADIUM METRICS (KPI 1.2) =====
        
        # C2: Stadium assignment must match schedule
        # If x[m_i_k, t, s] = 1, then stadium_i_k[i, s] = 1
        model.stadium_schedule_link = ConstraintList()
        for i in model.I:
            for s in model.S:
                model.stadium_schedule_link.add(3*model.stadium_i_s[i, s] >= sum(model.x[m, t, s] for m in model.M_i[i] for t in model.T_s[s]))
    
        # C4: Transition must match stadium assignments
        # transition_i_s1_s2 = 1 only if BOTH stadium_i_s[i, s1] = 1 AND stadium_i_s[i, s2] = 1
        model.transition_link = ConstraintList()
        for i in model.I:
            for s1 in model.S:
                for s2 in model.S:
                    if s1 < s2:  # Only define for s1 < s2 to avoid duplicates
                        model.transition_link.add(model.transition_i_s1_s2[i, s1, s2] >= model.stadium_i_s[i, s1] + model.stadium_i_s[i, s2] - 1)
        
        model.max_travel_distance = ConstraintList()
        dist_dict = self.params["dist_v_v"]
        for i in model.I:
            model.max_travel_distance.add(2 * sum(model.transition_i_s1_s2[i, s1, s2]*dist_dict[s1, s2] for s1 in model.S for s2 in model.S if s1 < s2)
             <= model.max_d)

        # ---------------------------------------------------------------------------------------

        # KPI 4.1: Venue-Load Balance
        # venue_count[s] = sum of matches at stadium s
        # d_s >= venue_count[s] - mean_count and d_s >= mean_count - venue_count[s]
        def venue_count_constraint(model, s):
            return model.venue_count[s] == sum(model.x[m, t, s] for m in model.M for t in model.T_s[s])

        mean_venue_count = len(model.M) / len(model.S)
        
        def venue_load_deviation_1(model, s):
            return model.d_s[s] >= model.venue_count[s] - mean_venue_count

        def venue_load_deviation_2(model, s):
            return model.d_s[s] >= mean_venue_count - model.venue_count[s]

        model.h_kpi_4_1_count = Constraint(model.S, rule=venue_count_constraint)
        model.h_kpi_4_1_dev_1 = Constraint(model.S, rule=venue_load_deviation_1)
        model.h_kpi_4_1_dev_2 = Constraint(model.S, rule=venue_load_deviation_2)


        # KPI 1.6: Rest Asymmetry Between Opponents
        # delta_m >= |r_im[team_a, m] - r_im[team_b, m]|
        # Constraints: delta_m >= r_im[a] - r_im[b] and delta_m >= r_im[b] - r_im[a]
        def rest_asymmetry_constraint_1(model, m):
            match_row = self.matches[self.matches["match_id"] == m]
            match = match_row.iloc[0]
            team_a = match["team_a_id"]
            team_b = match["team_b_id"]
            return model.delta_m[m] >= model.r_im[team_a, m] - model.r_im[team_b, m]

        def rest_asymmetry_constraint_2(model, m):
            match_row = self.matches[self.matches["match_id"] == m]
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
        
        model.rest_constraints = ConstraintList()  # To store rest constraints before adding to model
        
        for i in model.I:
            team_matches_pairs = [(m_prev, m) for m_prev in model.M_i[i] for m in model.M_i[i] if m_prev != m]
            # For each pair of matches this team plays
            for m_prev, m in team_matches_pairs:    
                for s in model.S:
                    for s_prev in model.S:            
                        # For each pair of time slots
                        for t_prev in model.T_s[s_prev]:
                            for t in model.T_s[s]:
                                # Compute time difference
                                date_prev, time_prev = self.slot_index_to_datetime[t_prev]
                                date_curr, time_curr = self.slot_index_to_datetime[t]
                                
                                dt_prev = datetime.strptime(f"{date_prev} {time_prev}", "%Y-%m-%d %H:%M")
                                dt_curr = datetime.strptime(f"{date_curr} {time_curr}", "%Y-%m-%d %H:%M")
                                time_diff_hours = (dt_curr - dt_prev).total_seconds() / 3600.0
                                rest_hours = time_diff_hours - match_duration
                                
                                # Only for positive rest (m is after m_prev)
                                if rest_hours > 72:  # Less than 3 days rest is not relevant for asymmetry
                                    model.rest_constraints.add(
                                        model.r_im[i, m] + 
                                        big_m * (2 - model.x[m_prev, t_prev, s_prev] - model.x[m, t, s])
                                        >= rest_hours)

        # =====================================================================================

        # Hard Constraints

        # H1-1: Each match scheduled exactly once
        def h1_rule(model, m):
            return sum(model.x[m, t, s] for s in model.S for t in model.T_s[s]) == 1

        model.h1 = Constraint(model.M, rule=h1_rule, doc="H1: Each match once")

        # H2: Round-robin (each team plays 3 matches)
        def h2_rule(model, i):
            team_matches = list(model.M_i[i])
            return (
                sum(
                    model.x[m, t, s]
                    for m in team_matches
                    for s in model.S
                    for t in model.T_s[s]
                )
                == 3
            )

        model.h2 = Constraint(model.I, rule=h2_rule, doc="H2: Round-robin")

        # H4: stadium turnover
        model.h4 = ConstraintList()
        for s in model.S:
            for date1 in set([t[0] for t in model.T_s[s]]):
                model.h4.add(sum(model.x[m, t_prime, s] for m in model.M for t_prime in model.T_s[s] if t_prime[0] == date1) <= 1)

        
        # H5: minimum team rest
        model.H5 = ConstraintList()
        for i in model.I:
            for date1 in set([t[0] for t in model.T]):
                d1 = date.fromisoformat(date1)
                all_3_days = [
                    (d1 + timedelta(days=di)).isoformat()
                    for di in range(4)
                ]
                model.H5.add(sum(model.x[m1, t, s] for s in model.S for m1 in model.M_i[i] for t in model.T_s[s] if t[0] in all_3_days) <= 1)
        
        # H6: host-nation matches in their country
        def h6_rule(model, team):
            return sum(model.x[m, t, s] for m in model.M for s in model.S for t in model.T_s[s] if m in model.M_i[team] and s not in self.S_c[team]) == 0  # Ensure variables are created
            
        model.h6 = Constraint(["USA", "MEX", "CAN"], rule=h6_rule, doc="H6: Host nation matches")
        
        # H7: Simultaneous final matches
        def h7a_rule(model, g):
            return sum(model.y[g, t] for t in model.T) == 1

        model.h7a = Constraint(model.G, rule=h7a_rule, doc="H7a: Final slot chosen")
        
        def h7b_rule(model, g, date1, time1):
            t = (date1, time1)
            group_matches = list(self.M_g[g])
            return (
                sum(model.x[m, t, s] for m in group_matches for s in model.S if t in model.T_s[s])
                >= 2 * model.y[g, t]
            )

        model.h7b = Constraint(
            model.G, model.T, rule=h7b_rule, doc="H7b: Final matches in slot"
        )

        def h7c_rule(model, g, date1, time1):
            group_matches = list(self.M_g[g])

            return (
                sum(model.x[m, t_prime, s] for m in group_matches for s in model.S for t_prime in model.T_s[s] if t_prime[0] > date1 or (t_prime[0] == date1 and t_prime[1] > time1))
                <= 6 * (1 - model.y[g, (date1, time1)])
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
                    for s in country_stadiums
                    for t in model.T_s[s]
                )
                == model.N_c[c]
            )

        model.h8 = Constraint(["USA", "MEX", "CAN"], rule=h8_rule, doc="H8: Country allocation")
        
        
        ############################################################### Obj

        # Objective: Minimize full weighted KPI (all 13 KPIs)
        # Direct KPIs (1.7, 2.2) via precomputed coefficients
        # + Inter-stadium KPIs (1.2) via auxiliary constraints
        # + Auxiliary variable KPIs (4.1)
        # + Geographic clustering penalty (NEW)
        def objective_rule(model):
            weights = self.params["weights"]
            norm_factors = self.kpi_normalization_factors
            
            # KPI 1.7 + 2.2: Direct coefficient-based KPIs (from precomputed coefficients)
            cost2_2 = sum(
                model.kpi_cost[t, s] * model.x[m, t, s]
                for m in model.M
                for s in model.S
                for t in model.T_s[s]
            )
            kpi_2_2_normalized = cost2_2 / norm_factors["kpi_2_2"]
            
            # # KPI 1.2: Travel distance (sum of inter-stadium distances)
            dist_dict = self.params["dist_v_v"]
            sum_travel = sum(model.transition_i_s1_s2[i, s1, s2]*dist_dict[s1, s2] for i in model.I for s1 in model.S for s2 in model.S if s1 < s2)
            kpi_1_2_normalized = 0.85 * (sum_travel / norm_factors["kpi_1_2"]["sum"]) + 0.15 * (model.max_d / norm_factors["kpi_1_2"]["max"])
            
            # # KPI 4.1: Venue-load balance (mean absolute deviation of match counts)
            kpi_4_1 = sum(model.d_s[s] for s in model.S)
            kpi_4_1_normalized = kpi_4_1 / norm_factors["kpi_4_1"]

            # # KPI 1.6: Rest asymmetry (sum of rest differences)
            kpi_1_6 = sum(model.delta_m[m] for m in model.M)
            kpi_1_6_normalized = kpi_1_6 / norm_factors["kpi_1_6"]
            
           # Combine all normalized KPIs
            return (weights["kpi_2_2"] * kpi_2_2_normalized+
                    weights["kpi_1_2"] * kpi_1_2_normalized+
                    weights["kpi_1_6"] * kpi_1_6_normalized+
                    weights["kpi_4_1"] * kpi_4_1_normalized)

        model.obj = Objective(rule=objective_rule, sense=minimize)

        self.model = model
        return model
    
    def solve(self, time_limit: int = 300, solver_name: str = "gurobi", warmstart: bool = True) -> Dict:
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
            solver.options["TimeLimit"]    = time_limit
            solver.options["MIPFocus"]     = 1      # prioritize finding/improving incumbents
            solver.options["Heuristics"]   = 0.5
            solver.options["ImproveStartTime"] = 60 # after 60s, stop caring about the bound entirely
            solver.options["Threads"]      = 8

        supports_ws = warmstart and used_solver in ("gurobi", "cbc")
        result = solver.solve(self.model, tee=True, warmstart=supports_ws)

        # Extract solution
        schedule = {}
        for m in self.model.M:
            for s in self.model.S:
                for t in self.model.T_s[s]:
                    if value(self.model.x[m, t, s]) > 0.5:
                        # get local time
                        v_offset = int(self.venues[self.venues['venue_id'] == s]['utc_offset_june'].iloc[0])
                        t_prime = pd.to_datetime(f"{t[0]} {t[1]}") + pd.Timedelta(hours=v_offset)
                        t_str = (t_prime.strftime("%Y-%m-%d"), t_prime.strftime("%H:%M"))
                        schedule[m] = (t_str, s)

        return {
            "status": str(result.solver.status),
            "objective": value(self.model.obj),
            "schedule": schedule,
            "model": self.model,
            "solver": used_solver,
        }
