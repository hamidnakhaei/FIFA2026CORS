"""
Base camp optimization (Step B) for FIFA 2026 Group-Stage.

Re-optimizes team base camps while holding the schedule fixed, by solving the
total-travel assignment problem:

        min_{u}  sum_i sum_b ctilde[i,b] * u[i,b]
        s.t.  sum_b u[i,b] = 1                     for all teams i   (one camp)
              sum_i u[i,b] <= 1                     for all camps b   (one team/camp)
              u[i,b] in {0,1}

where ctilde[i,b] is the round-trip travel of team i based at facility b
(summed over its three matches, factor 2 for out-and-back), plus a US visa-bond
penalty for bond teams choosing a US camp. The objective minimizes the TOTAL
(hence the average) per-team travel distance across all 48 teams
(KPI 1.1 efficiency objective).

CHANGES vs. the previous version:
  * Fix 2 (key-type mismatch): self.match_stadium is now keyed by int match_id
    so lookups in _penalized_travel succeed (previously every lookup missed,
    scoring all travel as 0 and producing an arbitrary assignment).
  * Fix 3 (objective): the model now minimizes the SUM of assigned travel
    (i.e. the average), instead of the minimax worst-case travel. The minimax
    bound D and its linearization constraint have been removed.

The primary solver is Pyomo + a MILP backend (gurobi/cbc/glpk), matching
schedule_optimizer.py. If no solver is available, a greedy fallback is used:
since the only coupling between teams is the shared "one team per camp"
constraint, a sum-minimizing assignment is a linear assignment problem.
"""

import pandas as pd
from typing import Dict, List


class BaseCampOptimizer:
    """Solve Step B: minimize the total (average) per-team travel distance,
    holding the schedule fixed."""

    def __init__(self, data_loader, schedule: Dict,
                 bond_penalty_km: float = 1500.0):
        self.loader = data_loader
        self.schedule = schedule

        self.teams = data_loader.get_teams()
        self.base_camps = data_loader.get_base_camps()
        self.params = data_loader.get_parameters()
        self.matches = data_loader.get_matches()

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

        # US entry / visa sets (KPI 1.7)
        self.ban_teams = set(self.params.get("us_visa_ban_teams", []))
        self.bond_teams = set(self.params.get("us_visa_bond_teams", []))
        self.bond_penalty_km = bond_penalty_km

        # Pre-compute the stadium each match is played in (schedule is fixed)
        #   schedule: match_id -> (slot, stadium_id)
        # FIX 2: key by int(match_id) so it matches the int match_ids stored in
        #        self.team_matches (previously keyed by str -> every lookup missed).
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
        matches = self.matches
        team_matches = {}
        for team_id in self.I:
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
                bc[bc["country"].astype(str).str.upper().isin(["USA"])][
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
        if (team_id in self.bond_teams or team_id in self.ban_teams) and base_camp_id in self.us_facilities:
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

    def compute_total_travel(self, base_camp_assignment: Dict) -> float:
        """Objective: the total per-team travel distance (sum over all teams)."""
        td = self.compute_team_travel(base_camp_assignment)
        return float(sum(td.values()))

    def compute_avg_travel(self, base_camp_assignment: Dict) -> float:
        """Average per-team travel distance."""
        td = self.compute_team_travel(base_camp_assignment)
        if not td:
            return 0.0
        return float(sum(td.values()) / len(td))

    def compute_max_travel(self, base_camp_assignment: Dict) -> float:
        """Worst-case (minimax) per-team travel distance, kept for reporting."""
        td = self.compute_team_travel(base_camp_assignment)
        if not td:
            return 0.0
        return float(max(td.values()))

    # ------------------------------------------------------------------ #
    #  Exact total-travel MILP (Pyomo)
    # ------------------------------------------------------------------ #
    def optimize(self, time_limit: int = 600, solver_name: str = "gurobi") -> Dict:
        """
        Solve the total-travel base-camp assignment as a MILP.

        Returns dict with assignment, cost (= total travel), avg_travel,
        max_travel, status. Falls back to the greedy solver if Pyomo / a MILP
        solver is unavailable.
        """
        try:
            from pyomo.environ import (
                ConcreteModel, Set, Var, Objective, Constraint,
                minimize, Binary, SolverFactory, value,
            )
        except Exception as e:
            print(f"  ⚠ Pyomo unavailable ({e}); using greedy fallback.")
            return self._greedy_min_sum()

        teams = list(self.I)

        # Penalized travel coefficients ctilde[i,b] over all facilities.
        ctilde = {}
        for i in teams:
            for b in self.facilities:
                ctilde[(i, b)] = self._penalized_travel(i, b)

        model = ConcreteModel()
        model.I = Set(initialize=teams)
        model.B = Set(initialize=self.facilities)
        ib_pairs = [(i, b) for i in teams for b in self.facilities]
        model.IB = Set(initialize=ib_pairs, dimen=2)

        model.u = Var(model.IB, within=Binary)

        # (30-1) exactly one camp per team
        def one_camp_rule(m, i):
            return sum(m.u[i, b] for b in self.facilities) == 1
        model.one_camp = Constraint(model.I, rule=one_camp_rule)

        # (30-2) at most one team per camp
        def one_team_rule(m, b):
            return sum(m.u[i, b] for i in teams) <= 1
        model.one_team = Constraint(model.B, rule=one_team_rule)

        # FIX 3: minimize TOTAL (hence average) assigned travel instead of the
        #        minimax worst case. No D variable / minimax constraint needed.
        model.obj = Objective(
            expr=sum(ctilde[(i, b)] * model.u[i, b] for (i, b) in ib_pairs),
            sense=minimize,
        )

        # Solve
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
            print("  ⚠ No MILP solver available; using greedy fallback.")
            return self._greedy_min_sum()

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

        total_cost = float(value(model.obj))
        n = len(assignment) if assignment else 1

        return {
            "assignment": assignment,
            "cost": total_cost,                       # total travel
            "avg_travel": total_cost / n,             # average per team
            "max_travel": self.compute_max_travel(assignment),
            "status": str(result.solver.status),
            "solver": used,
        }

    # ------------------------------------------------------------------ #
    #  Greedy fallback (no MILP solver): solve the min-sum assignment
    #  with a simple cheapest-feasible-camp heuristic respecting the
    #  one-team-per-camp constraint.
    # ------------------------------------------------------------------ #
    def _greedy_min_sum(self) -> Dict:
        """Greedy min-sum assignment: process teams and assign each its cheapest
        still-available eligible camp. Coupling is only the one-team-per-camp
        rule, so a greedy pass over (team, camp) costs is a reasonable
        heuristic when no MILP solver is present."""
        used_camps = set()
        assignment = {}

        # Build all feasible (cost, team, camp) triples, cheapest first.
        triples = []
        for i in self.I:
            for b in self._eligible_facilities(i):
                triples.append((self._penalized_travel(i, b), i, b))
        triples.sort(key=lambda x: x[0])

        for cost, i, b in triples:
            if i in assignment or b in used_camps:
                continue
            assignment[i] = b
            used_camps.add(b)

        total_cost = self.compute_total_travel(assignment)
        n = len(assignment) if assignment else 1
        return {
            "assignment": assignment,
            "cost": total_cost,
            "avg_travel": total_cost / n,
            "max_travel": self.compute_max_travel(assignment),
            "status": "greedy_fallback",
            "solver": "greedy",
        }

