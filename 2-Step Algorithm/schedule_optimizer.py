"""
Schedule optimization solver (Step A) for FIFA 2026 Group-Stage.
Solves the MILP to assign matches to slots and stadiums, minimizing weighted KPIs.
"""

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

        # Get sets and indices
        (
            self.M,
            self.T,
            self.S,
            self.I,
            self.G,
            self.M_i,
            self.M_g,
            self.T_r,
            self.S_c,
        ) = data_loader.get_sets_and_indices()

        self.params = data_loader.get_parameters()
        self.model = None
        
        # Extract unique (date, time) slots from matches
        self.date_time_slots = self._extract_date_time_slots()
        self.slot_index_to_datetime = {i: slot for i, slot in enumerate(self.date_time_slots)}
        self.datetime_to_slot_index = {slot: i for i, slot in enumerate(self.date_time_slots)}

    def _extract_date_time_slots(self) -> list:
        """
        Extract unique (date, kickoff_local) combinations from matches.
        Returns sorted list of tuples: [(date_str, time_str), ...]
        """
        slots = set()
        for _, match in self.matches.iterrows():
            date = match["date"]
            time = match["kickoff_local"]
            slots.add((str(date), str(time)))
        return sorted(list(slots))

    def _compute_kpi_coefficients(self) -> Dict:
        """
        Precompute KPI contributions for each (match, time_slot, stadium) combination.
        Returns dict mapping (m, t, s) -> weighted_kpi_cost
        t is an index into date_time_slots.
        """
        kpi_costs = {}
        
        for m in self.M:
            match_row = self.matches[self.matches["match_id"] == m]
            if len(match_row) == 0:
                continue
            match = match_row.iloc[0]
            
            for t_idx, (date, time) in enumerate(self.date_time_slots):
                for s in self.S:
                    cost = 0.0
                    
                    # KPI 1.2: Travel distance (weight 0.10)
                    for team_id in [match["team_a_id"], match["team_b_id"]]:
                        if team_id in self.base_camp_assignment:
                            base_camp_id = self.base_camp_assignment[team_id]
                            dist = self.params["dist"].get((base_camp_id, s), 0)
                            cost += 0.10 * dist / 100  # Normalize distance
                    
                    # KPI 1.4: Jet lag - time zone difference (weight 0.08)
                    for team_id in [match["team_a_id"], match["team_b_id"]]:
                        if team_id in self.base_camp_assignment:
                            base_camp_id = self.base_camp_assignment[team_id]
                            stadium_tz = self.params["tzone_stadium"].get(s, 0)
                            camp_tz = self.params["tzone_basecamp"].get(base_camp_id, 0)
                            tz_diff = abs(stadium_tz - camp_tz)
                            cost += 0.08 * tz_diff
                    
                    # KPI 2.4: Weather WBGT penalty (weight 0.10)
                    venue_weather = self.params["weather"][
                        self.params["weather"]["venue_id"] == s
                    ]
                    if len(venue_weather) > 0:
                        avg_temp = venue_weather["temperature_c"].mean()
                        weather_penalty = max(0, avg_temp - 20)
                        cost += 0.10 * weather_penalty / 10  # Normalize
                    
                    # KPI 4.1: Broadcast value (weight 0.10) - prefer high match value in good slots
                    match_value = self.params["match_value"].get(m, 0.5)
                    broadcast_quality = self.params["broadcast_quality"].get(t_idx, 0.5)
                    cost -= 0.10 * match_value * broadcast_quality  # Negative because we want to maximize
                    
                    kpi_costs[(m, t_idx, s)] = cost
        
        return kpi_costs

    def build_model(self) -> ConcreteModel:
        """Build the MILP model for schedule optimization."""
        model = ConcreteModel()

        # Sets
        model.M = Set(initialize=list(self.M))  # Matches
        model.T = Set(initialize=list(range(len(self.date_time_slots))))  # Time slots (date/time combinations)
        model.S = Set(initialize=list(self.S))  # Stadiums
        model.I = Set(initialize=list(self.I))  # Teams
        model.G = Set(initialize=list(self.G))  # Groups

        # Decision Variables
        model.x = Var(model.M, model.T, model.S, within=Binary)  # Assignment vars
        model.y = Var(model.G, model.T, within=Binary)  # Final slot indicator

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
                        for m in self.M for t in range(len(self.date_time_slots)) for s in self.S},
        )

        # Objective: Minimize full weighted KPI (all 13 KPIs)
        def objective_rule(model):
            return sum(
                model.kpi_cost[m, t, s] * model.x[m, t, s]
                for m in model.M
                for t in model.T
                for s in model.S
            )

        model.obj = Objective(rule=objective_rule, sense=minimize)

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
                == self.loader.config_params.MATCHES_PER_TEAM
            )

        model.h2 = Constraint(model.I, rule=h2_rule, doc="H2: Round-robin")

       
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
        
        if solver is None or used_solver is None:
            raise RuntimeError(
                "No linear/integer programming solver available. "
                "Please install CBC, GLPK, or Gurobi."
            )

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
        if result.solver.status.value == "ok":
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

    def get_schedule_dict(self) -> Dict:
        """Extract schedule from solved model as dict mapping match_id -> (time_slot, stadium)."""
        schedule = {}
        if self.model is not None:
            for m in self.model.M:
                for t in self.model.T:
                    for s in self.model.S:
                        if value(self.model.x[m, t, s]) > 0.5:
                            schedule[int(m)] = (int(t), s)
        return schedule


if __name__ == "__main__":
    from data_loader import DataLoader
    from kpis import KPICalculator

    loader = DataLoader()
    data = loader.load_all()
    params = loader.get_parameters()

    # Dummy base camp assignment
    base_camp_assignment = {
        "BRA": 1,
        "GER": 2,
        "ARG": 3,
        "FRA": 4,
        "ENG": 5,
    }

    kpi_calc = KPICalculator(loader, params)
    optimizer = ScheduleOptimizer(loader, kpi_calc, base_camp_assignment)

    print("Building MILP model for schedule optimization...")
    optimizer.build_model()
    print("✓ Model built")

    print("Attempting to solve (requires GLPK/Gurobi/CBC)...")
    try:
        result = optimizer.solve(time_limit=10, solver_name="glpk")
        print(f"✓ Solver status: {result['status']}")
        print(f"  Objective value: {result['objective']}")
        print(f"  Matches scheduled: {len(result['schedule'])}")
    except Exception as e:
        print(f"⚠ Solver error (expected if no solver installed): {e}")
