"""
Base camp optimization (Step B) for FIFA 2026 Group-Stage.

Re-optimizes team base camps while holding the schedule fixed, by solving the
minimax-travel assignment problem (eq. 29-33 of the model):

        min_{u, D}  D
        s.t.  sum_b u[i,b] = 1                     for all teams i   (one camp)
              D >= sum_b ctilde[i,b] * u[i,b]      for all teams i   (minimax)
              u[i,b] = 0  for i banned, b in US                      (US ban)
              u[i,b] in {0,1}

where ctilde[i,b] is the round-trip travel of team i based at facility b
(summed over its three matches, factor 2 for out-and-back), plus a US visa-bond
penalty for bond teams choosing a US camp. The objective minimizes the maximum
per-team travel distance across all 48 teams (KPI 1.2 equity / minimax).

The primary solver is Pyomo + a MILP backend (gurobi/cbc/glpk), matching
schedule_optimizer.py. If no solver is available, a greedy fallback is used:
since the only coupling between teams is the shared bound D, each team
independently picking its minimum feasible (penalized) travel is optimal for the
minimax objective.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from copy import deepcopy


class BaseCampOptimizer:
    """Solve Step B: minimize the maximum per-team travel distance,
    holding the schedule fixed."""

    def __init__(self, data_loader, kpi_calculator, schedule: Dict,
                 bond_penalty_km: float = 1500.0):
        self.loader = data_loader
        self.kpi_calc = kpi_calculator
        self.schedule = schedule

        self.teams = data_loader.get_teams()
        self.base_camps = data_loader.get_base_camps()
        self.params = data_loader.get_parameters()

        # US entry / visa sets (KPI 1.7)
        self.ban_teams = set(self.params.get("us_visa_ban_teams", []))
        self.bond_teams = set(self.params.get("us_visa_bond_teams", []))
        self.bond_penalty_km = bond_penalty_km

        # Pre-compute the stadium each match is played in (schedule is fixed)
        #   schedule: match_id -> (slot, stadium_id)
        self.match_stadium = {
            int(m): stadium_id for m, (slot, stadium_id) in self.schedule.items()
        }

        # team_id -> list of its match_ids (each team plays 3)
        self.team_matches = self._build_team_matches()

        # Candidate base-camp facilities and which ones are in the US
        self.facilities = list(self.base_camps["base_camp_id"].unique())
        self.us_facilities = self._build_us_facilities()

    # ------------------------------------------------------------------ #
    #  Setup helpers
    # ------------------------------------------------------------------ #
    def _build_team_matches(self) -> Dict:
        """Map each team to the match_ids it plays (from the KPI matches table)."""
        matches = self.kpi_calc.matches
        team_matches = {}
        for team_id in self.teams["team_id"].unique():
            mids = matches[
                (matches["team_a_id"] == team_id)
                | (matches["team_b_id"] == team_id)
            ]["match_id"].tolist()
            team_matches[team_id] = [int(m) for m in mids]
        return team_matches

    def _build_us_facilities(self) -> set:
        """Set of base_camp_ids located in the USA (for the entry ban)."""
        bc = self.base_camps
        if "country" in bc.columns:
            return set(
                bc[bc["country"].astype(str).str.upper().isin(["USA", "US", "UNITED STATES"])][
                    "base_camp_id"
                ].tolist()
            )
        return set()

    def _eligible_facilities(self, team_id) -> List:
        """Facilities a team may use: all facilities, minus US ones if banned."""
        if team_id in self.ban_teams:
            return [b for b in self.facilities if b not in self.us_facilities]
        return list(self.facilities)

    def _penalized_travel(self, team_id, base_camp_id) -> float:
        """ctilde[i,b]: round-trip travel (sum over the team's 3 matches, x2)
        plus the US visa-bond penalty if a bond team picks a US camp."""
        total = 0.0
        for m in self.team_matches.get(team_id, []):
            stadium_id = self.match_stadium.get(m)
            if stadium_id is None:
                continue
            d = self.params["dist"].get((base_camp_id, stadium_id), 0.0)
            total += 2.0 * d  # out-and-back round trip
        if team_id in self.bond_teams and base_camp_id in self.us_facilities:
            total += self.bond_penalty_km
        return total

    # ------------------------------------------------------------------ #
    #  Objective evaluation
    # ------------------------------------------------------------------ #
    def compute_team_travel(self, base_camp_assignment: Dict) -> Dict:
        """Per-team penalized round-trip travel distance under an assignment."""
        return {
            team_id: self._penalized_travel(team_id, base_camp_id)
            for team_id, base_camp_id in base_camp_assignment.items()
        }

    def compute_max_travel(self, base_camp_assignment: Dict) -> float:
        """Minimax objective: the maximum per-team travel distance."""
        td = self.compute_team_travel(base_camp_assignment)
        if not td:
            return 0.0
        return float(max(td.values()))

    # Backward-compatible alias (older code referenced compute_org_objective).
    def compute_org_objective(self, base_camp_assignment: Dict) -> float:
        return self.compute_max_travel(base_camp_assignment)

    # ------------------------------------------------------------------ #
    #  Exact minimax MILP (Pyomo)
    # ------------------------------------------------------------------ #
    def solve_minimax(self, time_limit: int = 120, solver_name: str = "gurobi",
                      lexicographic: bool = True) -> Dict:
        """
        Solve the minimax-travel base-camp assignment as a MILP.

        Returns dict with best_assignment, best_cost (= max travel), status.
        Falls back to the greedy solver if Pyomo / a MILP solver is unavailable.
        """
        try:
            from pyomo.environ import (
                ConcreteModel, Set, Var, Objective, Constraint,
                minimize, Binary, NonNegativeReals, SolverFactory, value,
            )
        except Exception as e:
            print(f"  ⚠ Pyomo unavailable ({e}); using greedy minimax fallback.")
            return self._greedy_minimax()

        teams = list(self.team_matches.keys())

        # Penalized travel coefficients ctilde[i,b] over eligible facilities only.
        ctilde = {}
        eligible = {}
        for i in teams:
            elig = self._eligible_facilities(i)
            eligible[i] = elig
            for b in elig:
                ctilde[(i, b)] = self._penalized_travel(i, b)

        model = ConcreteModel()
        model.I = Set(initialize=teams)
        # Index set of valid (team, facility) pairs (respects the US ban).
        ib_pairs = [(i, b) for i in teams for b in eligible[i]]
        model.IB = Set(initialize=ib_pairs, dimen=2)

        model.u = Var(model.IB, within=Binary)
        model.D = Var(within=NonNegativeReals)

        # (30) exactly one camp per team
        def one_camp_rule(m, i):
            return sum(m.u[i, b] for b in eligible[i]) == 1
        model.one_camp = Constraint(model.I, rule=one_camp_rule)

        # (31) minimax linearization: D >= team i's assigned travel
        def minimax_rule(m, i):
            return m.D >= sum(ctilde[(i, b)] * m.u[i, b] for b in eligible[i])
        model.minimax = Constraint(model.I, rule=minimax_rule)

        # (29) minimize the worst-case travel
        model.obj = Objective(expr=model.D, sense=minimize)

        # Solve (try requested solver, then common fallbacks)
        solver, used = None, None
        for cand in [solver_name, "gurobi", "cbc", "glpk"]:
            try:
                s = SolverFactory(cand)
                if s is not None and s.available():
                    solver, used = s, cand
                    break
            except Exception:
                continue
        if solver is None:
            print("  ⚠ No MILP solver available; using greedy minimax fallback.")
            return self._greedy_minimax()

        print(f"  Using solver: {used}")
        if used == "gurobi":
            solver.options["TimeLimit"] = time_limit
        elif used == "cbc":
            solver.options["seconds"] = time_limit
        elif used == "glpk":
            solver.options["tmlim"] = time_limit

        result = solver.solve(model, tee=False)

        assignment = {}
        for (i, b) in ib_pairs:
            if value(model.u[i, b]) is not None and value(model.u[i, b]) > 0.5:
                assignment[i] = b
        best_cost = float(value(model.D))

        # Optional lexicographic tie-break: fix D = D*, minimize total travel,
        # so non-binding teams get the efficient (shortest-travel) camp.
        if lexicographic and assignment:
            try:
                model.Dfix = Constraint(expr=model.D <= best_cost + 1e-6)
                model.obj.deactivate()
                model.obj2 = Objective(
                    expr=sum(ctilde[(i, b)] * model.u[i, b] for (i, b) in ib_pairs),
                    sense=minimize,
                )
                solver.solve(model, tee=False)
                assignment = {
                    i: b for (i, b) in ib_pairs
                    if value(model.u[i, b]) is not None and value(model.u[i, b]) > 0.5
                }
                best_cost = self.compute_max_travel(assignment)
            except Exception as e:
                print(f"  ⚠ Lexicographic refinement skipped ({e}).")

        return {
            "best_assignment": assignment,
            "best_cost": best_cost,
            "status": str(result.solver.status),
            "solver": used,
        }

    # ------------------------------------------------------------------ #
    #  Greedy fallback (optimal for minimax when teams are uncoupled)
    # ------------------------------------------------------------------ #
    def _greedy_minimax(self) -> Dict:
        """Each team independently picks its minimum feasible penalized-travel
        camp. Because teams couple only through the shared bound D, the resulting
        max-travel equals the minimax optimum."""
        assignment = {}
        for i in self.team_matches.keys():
            elig = self._eligible_facilities(i)
            if not elig:
                continue
            best_b = min(elig, key=lambda b: self._penalized_travel(i, b))
            assignment[i] = best_b
        return {
            "best_assignment": assignment,
            "best_cost": self.compute_max_travel(assignment),
            "status": "greedy",
            "solver": "greedy",
        }

    # ------------------------------------------------------------------ #
    #  Public entry point (keeps main.py's call signature & return keys)
    # ------------------------------------------------------------------ #
    def optimize(
        self,
        initial_assignment: Dict,
        max_iterations: int = 10,                 # accepted for compatibility
        temperature_schedule: str = "exponential",  # (unused: Step B is now exact)
        num_changes_per_iteration: int = 5,       # (unused)
        time_limit: int = 120,
        solver_name: str = "gurobi",
    ) -> Dict:
        """
        Optimize base camps by minimizing the maximum per-team travel distance.

        The minimax problem is solved exactly (single MILP); the simulated-
        annealing arguments are accepted for backward compatibility but no longer
        drive the search. Returns the same keys main.py expects.
        """
        result = self.solve_minimax(time_limit=time_limit, solver_name=solver_name)

        best_assignment = result["best_assignment"]
        best_cost = result["best_cost"]

        # Report the improvement over the starting (e.g. current) assignment.
        init_cost = (
            self.compute_max_travel(initial_assignment)
            if initial_assignment else float("inf")
        )
        improved = best_cost <= init_cost

        team_travel = self.compute_team_travel(best_assignment)
        print(
            f"  ✓ Minimax travel solved ({result['status']}): "
            f"max={best_cost:.2f} km (was {init_cost:.2f} km), "
            f"mean={np.mean(list(team_travel.values())):.2f} km"
        )

        return {
            "best_assignment": best_assignment,
            "best_cost": best_cost,            # = minimax (max per-team travel)
            "final_assignment": best_assignment,
            "final_cost": best_cost,
            "team_travel": team_travel,
            "max_travel": best_cost,
            "improved_over_initial": improved,
            # SA-style fields kept so existing callers don't break:
            "costs_history": [init_cost, best_cost],
            "acceptance_rate": 1.0 if improved else 0.0,
            "iterations": 1,
            "status": result["status"],
            "solver": result.get("solver"),
        }


def generate_initial_assignment(teams_df: pd.DataFrame) -> Dict:
    """
    Generate an initial base camp assignment (used only as a starting point /
    baseline for the improvement report; the minimax solve does not depend on it).
    Returns dict mapping team_id -> base_camp_id.
    """
    assignment = {}
    for _, team in teams_df.iterrows():
        team_id = team["team_id"]
        base_camp_id = (hash(team_id) % 10) + 1
        assignment[team_id] = base_camp_id
    return assignment


def load_base_camp_assignment_from_data(base_camps_df: pd.DataFrame) -> Dict:
    """
    Load current base camp assignments from base_camps.csv.
    Returns dict mapping team_id -> base_camp_id for assigned teams.
    Only includes teams with confirmed assignments (team_id not null).
    """
    assignment = {}
    assigned = base_camps_df[base_camps_df["team_id"].notna()]
    for _, row in assigned.iterrows():
        team_id = row["team_id"]
        base_camp_id = row["base_camp_id"]
        assignment[team_id] = base_camp_id
    return assignment