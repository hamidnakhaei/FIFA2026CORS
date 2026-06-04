"""
Schedule optimization solver (Step A) for FIFA 2026 Group-Stage.
Solves the MILP to assign matches to slots and stadiums, minimizing weighted KPIs.
"""

from datetime import date, datetime, timedelta
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
              
        # KPI 2.2: Heat load (WBGT hours per team)
        # Excess above 28°C, worst case ~10°C excess × 3 matches × 32 teams
        factors["kpi_2_2"] = 10.0 * 3 * len(self.I)
        
        # KPI 4.1: Venue-load balance (mean absolute deviation)
        # Worst case: unbalanced distribution of 72 matches across 12 stadiums
        avg_matches = len(self.M) / len(self.S) if len(self.S) > 0 else 1
        max_deviation = len(self.M) - avg_matches  # Max possible deviation
        factors["kpi_4_1"] = max_deviation * len(self.S) / 2  # Typical MAD estimate
        
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
        
        for (date, time_str) in self.T:
            for s in self.S:
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
                kpi_costs[(date, time_str, s)] = excess_wbgt * 2.0 
                
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
        model.ISS = Set(
                    dimen=3,
                    initialize=lambda m: (
                        (i, s1, s2)
                        for i in m.I
                        for s1 in m.S
                        for s2 in m.S
                    )
                )
        model.IS = Set(
                    dimen=2,
                    initialize=lambda m: (
                        (i, s1)
                        for i in m.I
                        for s1 in m.S
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
        model.stadium_i_s = Var(model.IS, within=Binary)
        # Binary transition indicators for inter-stadium costs
        # For team i between match positions k and k', is it playing at (s1, s2)?
        model.transition_i_s1_s2 = Var(model.ISS, within=Binary)
        # Continuous travel distance for each team (KPI 1.2)
        # model.travel_distance_i = Var(model.I, within=NonNegativeReals)
        
        # ===== AUXILIARY VARIABLES FOR OTHER KPIS =====
        model.d_s = Var(model.S, within=NonNegativeReals)  # KPI 4.1: Venue load deviation
        model.venue_count = Var(model.S, within=NonNegativeIntegers)  # KPI 4.1: Number of matches per venue
        
        ################################################################################## Cons

        # ===== CONSTRAINTS FOR INTER-STADIUM METRICS (KPI 1.2) =====
        
        # C2: Stadium assignment must match schedule
        # If x[m_i_k, t, s] = 1, then stadium_i_k[i, s] = 1
        model.stadium_schedule_link = ConstraintList()
        for i in model.I:
            for s in model.S:
                model.stadium_schedule_link.add(3*model.stadium_i_s[i, s] >= sum(model.x[k, t, s] for k in model.M_i[i] for t in model.T))
    
        # C4: Transition must match stadium assignments
        # transition_i_s1_s2 = 1 only if stadium_i_k[i, k1, s1] = 1 AND stadium_i_k[i, k2, s2] = 1
        model.transition_link = ConstraintList()
        for i in model.I:
            for s1 in model.S:
                for s2 in model.S:
                    model.transition_link.add(model.stadium_i_s[i, s1] + model.stadium_i_s[i, s2] <= 1+model.transition_i_s1_s2[i, s1, s2])
        
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
            return (sum(model.x[m, t_prime, s] for m in model.M for t_prime in model.T if t_prime[0] == date) <= 1)
        
        model.h4 = Constraint(model.S, [t[0] for t in model.T], rule=h4_rule, doc="H4: Stadium turnover")
        
        # H5: minimum team rest
        model.H5 = ConstraintList()
        time_window = self.params["R_min"] + self.params["match_duration"]
        for i in model.I:
            for date1 in set([t[0] for t in model.T]):
                d1 = date.fromisoformat(date1)
                all_3_days = [
                    (d1 + timedelta(days=i)).isoformat()
                    for i in range(4)
                ]
                model.H5.add(sum(model.x[m1, t, s] for m1 in model.M_i[i] for s in model.S for t in model.T if t[0] in all_3_days) <= 1)
        
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
            group_matches = list(self.M_g[g])

            return (
                sum(model.x[m, t_prime, s] for m in group_matches for s in model.S for t_prime in model.T if t_prime[0] > date or (t_prime[0] == date and t_prime[1] > time))
                <= 6 * (1 - model.y[g, (date, time)])
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
        # + Inter-stadium KPIs (1.2) via auxiliary constraints
        # + Auxiliary variable KPIs (4.1)
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
                weights["kpi_2_2"] * norm_factors["kpi_2_2"]
            )
            direct_cost_normalized = (
                direct_cost / direct_cost_norm if direct_cost_norm > 0 else direct_cost
            )
            
            # # KPI 1.2: Travel distance (sum of inter-stadium distances)
            dist_dict = self.params["dist_v_v"]
            kpi_1_2 = sum(model.transition_i_s1_s2[i, s1, s2]*dist_dict[s1, s2] for i in model.I for s1 in model.S for s2 in model.S)
            kpi_1_2_normalized = kpi_1_2 / norm_factors["kpi_1_2"]
            
            # # KPI 4.1: Venue-load balance (mean absolute deviation of match counts)
            kpi_4_1 = sum(model.d_s[s] for s in model.S)
            kpi_4_1_normalized = kpi_4_1 / norm_factors.get("kpi_4_1", 1.0)
            
           # Combine all normalized KPIs
            return (direct_cost_normalized +
                    # weights["kpi_1_2"] * kpi_1_2_normalized+
                    weights["kpi_4_1"] * kpi_4_1_normalized)

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
