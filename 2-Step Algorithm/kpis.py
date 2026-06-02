"""
KPI computation module for FIFA 2026 Group-Stage Optimization.
Implements all 13 selected KPIs with exact formulas from KPIs.tex.
"""

from typing import Dict
import numpy as np

class KPICalculator:
    """Calculate all 13 KPIs exactly as defined in KPIs.tex."""

    def __init__(self, data_loader, params: Dict):
        self.loader = data_loader
        self.params = params
        self.matches = data_loader.get_matches()
        self.venues = data_loader.get_venues()
        self.teams = data_loader.get_teams()
        self.base_camps = data_loader.get_base_camps()

    def compute_objective_weighted(self, schedule: Dict, base_camp_assignment: Dict) -> float:
        """Compute weighted sum of all 13 KPIs (single objective value)."""
        kpis = self.compute_all_kpis(schedule, base_camp_assignment)
        weights = self.params.get("weights", {})
        if not weights:
            # Fallback to config_params if not in params dict
            weights = self.loader.config_params.KPI_WEIGHTS
        return sum(kpis.get(k, 0) * weights.get(k, 0) for k in weights)

    def compute_all_kpis(
        self, schedule: Dict, base_camp_assignment: Dict
    ) -> Dict[str, float]:
        """
        Compute all 13 KPIs from KPIs.tex.

        Args:
            schedule: Dict mapping match_id to (slot, stadium)
            base_camp_assignment: Dict mapping team_id to base_camp_id

        Returns:
            Dictionary with KPI values (exact definitions from KPIs.tex)
        """
        kpis = {
            "kpi_1_2": self._kpi_1_2_intra_group_travel_dispersion(schedule, base_camp_assignment),
            "kpi_1_3": self._kpi_1_3_circadian_shift_cost(schedule, base_camp_assignment),
            "kpi_1_4": self._kpi_1_4_match_venue_geographic_dispersion(schedule, base_camp_assignment),
            "kpi_1_6": self._kpi_1_6_rest_asymmetry(schedule),
            "kpi_1_7": self._kpi_1_7_entry_visa_restriction(schedule, base_camp_assignment),
            "kpi_2_2": self._kpi_2_2_per_team_heat_load(schedule),
            "kpi_3_3": self._kpi_3_3_first_mover_balance(schedule),
            "kpi_4_1": self._kpi_4_1_venue_load_balance(schedule),
            "kpi_4_2": self._kpi_4_2_fan_accessibility(schedule),
            "kpi_5_2": self._kpi_5_2_marquee_match_quality(schedule),
            "kpi_5_3": self._kpi_5_3_host_city_economic_equity(schedule),
        }
        return kpis

    def _kpi_1_2_intra_group_travel_dispersion(
        self, schedule: Dict, base_camp_assignment: Dict
    ) -> float:
        """
        KPI 1.2: Intra-Group Travel Dispersion.
        Formula: Δ_g = max(TD_i) - min(TD_i) for each group.
        Minimize sum of ranges across all groups.
        """
        # First compute TD_i for each team (distance for their 3 matches)
        TD = {}
        for team_id in self.teams["team_id"].unique():
            total_distance = 0.0
            for match_id, (slot, stadium_id) in schedule.items():
                match = self.matches[self.matches["match_id"] == match_id].iloc[0]
                if team_id in [match["team_a_id"], match["team_b_id"]]:
                    if team_id in base_camp_assignment:
                        base_camp_id = base_camp_assignment[team_id]
                        distance = self.params["dist"].get((base_camp_id, stadium_id), 0)
                        total_distance += distance
            TD[team_id] = total_distance

        # Compute range for each group
        total_dispersion = 0.0
        for group in self.teams["group"].unique():
            group_teams = self.teams[self.teams["group"] == group]["team_id"].tolist()
            group_distances = [TD[id] for id in group_teams]
            delta_g = max(group_distances) - min(group_distances)
            total_dispersion += delta_g

        return total_dispersion

    def _kpi_1_3_circadian_shift_cost(
        self, schedule: Dict, base_camp_assignment: Dict
    ) -> float:
        """
        KPI 1.3: Circadian Shift Cost (Jet-Lag Burden).
        Uses perceived kickoff time and circadian penalty function φ(τ̂).
        Thresholds: τ_lo = 23, τ_hi = 7 (subjective night window).
        Formula: JL_i = Σ_k φ(τ̂_{i,k}), then sum across teams.
        """
        tau_lo = 23.0  # Start of subjective night (23:00)
        tau_hi = 7.0   # End of subjective night (7:00)
        max_penalty = 8.0  # Maximum circadian penalty in hours
        
        total_jetlag = 0.0

        for team_id in self.teams["team_id"].unique():
            team_jetlag = 0.0
            for match_id, (slot, stadium_id) in schedule.items():
                match = self.matches[self.matches["match_id"] == match_id].iloc[0]
                if team_id not in [match["team_a_id"], match["team_b_id"]]:
                    continue

                # Get kickoff time from slot
                if isinstance(slot, tuple):
                    _, kickoff_time = slot  # (date, time)
                    # Parse kickoff_time (e.g., "13:00")
                    try:
                        kickoff_hour = float(kickoff_time.split(":")[0])
                    except:
                        kickoff_hour = 12.0  # Default fallback
                else:
                    kickoff_hour = 12.0  # Fallback

                # Get timezone offset
                if team_id in base_camp_assignment:
                    base_camp_id = base_camp_assignment[team_id]
                    camp_tz = self.params["tzone_basecamp"].get(base_camp_id, 0)
                    stadium_tz = self.params["tzone_stadium"].get(stadium_id, 0)
                    tz_offset = stadium_tz - camp_tz
                else:
                    tz_offset = 0

                # Compute perceived kickoff time (mod 24)
                tau_hat = (kickoff_hour - tz_offset) % 24

                # Circadian penalty function φ(τ̂)
                if tau_hat >= tau_lo or tau_hat <= tau_hi:
                    # In or near subjective night window
                    if tau_hat >= tau_lo:
                        penalty = min(tau_hat - tau_lo, tau_hi + 24 - tau_lo)
                    else:
                        penalty = min(tau_hat + 24 - tau_lo, tau_hi - tau_hat)
                    penalty = min(penalty, max_penalty)
                else:
                    penalty = 0.0

                team_jetlag += penalty

            total_jetlag += team_jetlag

        return total_jetlag

    def _kpi_1_4_match_venue_geographic_dispersion(
        self, schedule: Dict, base_camp_assignment: Dict
    ) -> float:
        """
        KPI 1.4: Match-Venue Geographic Dispersion.
        Formula: CC_i = |{cluster(v_i,1), cluster(v_i,2), cluster(v_i,3)}|
        Minimize total cluster count across all teams.
        or 
        BC_i = Σ_k 𝟙[country(b_i) ≠ country(v_i,k)]
        Minimize total border crossing across all teams.
        """
        total_cc = 0.0
        total_bc = 0.0

        for team_id in self.teams["team_id"].unique():
            clusters = set()
            base_camp_id = base_camp_assignment[team_id]
            base_camp_country = self.base_camps[base_camp_id]["country"].values[0]
            for match_id, (slot, stadium_id) in schedule.items():
                match = self.matches[self.matches["match_id"] == match_id].iloc[0]
                if team_id in [match["team_a_id"], match["team_b_id"]]:
                    cluster = self.params["cluster"][stadium_id]
                    clusters.add(cluster)
                    if base_camp_country != self.venues[self.venues["venue_id"] == stadium_id]["country"].values[0]:
                        total_bc += 1  # Count base camp cluster mismatch

            # Penalty for concentration: fewer clusters = higher penalty
            cc_i = len(clusters)
            total_cc += cc_i  # Penalize high diversity 

        return total_bc

    def _kpi_1_6_rest_asymmetry(self, schedule: Dict) -> float:
        """
        KPI 1.6: Rest Asymmetry Between Opponents.
        Formula: RD_m = |r(m,i) - r(m,j)| for each match.
        Minimize sum of rest differences across all matches.
        """
        rest_penalty = 0.0

        for match_id, (slot, stadium_id) in schedule.items():
            match = self.matches[self.matches["match_id"] == match_id].iloc[0]
            team_a = match["team_a_id"]
            team_b = match["team_b_id"]

            # Compute rest days for each team before this match
            # Rest = days since last match
            rest_a = self._compute_rest_days(team_a, match_id, schedule)
            rest_b = self._compute_rest_days(team_b, match_id, schedule)

            asymmetry = abs(rest_a - rest_b)
            rest_penalty += int(asymmetry>0)

        return rest_penalty

    def _compute_rest_days(self, team_id: str, match_id: int, schedule: Dict) -> int:
        """Compute rest days for a team before a specific match."""
        team_matches = []
        for mid, (slot, stadium_id) in schedule.items():
            match = self.matches[self.matches["match_id"] == mid].iloc[0]
            if team_id in [match["team_a_id"], match["team_b_id"]]:
                team_matches.append((mid, match["date"]))

        team_matches.sort(key=lambda x: x[1])

        # Find index of current match
        for i, (mid, date) in enumerate(team_matches):
            if mid == match_id:
                if i == 0:
                    return 0  # First match, no rest
                else:
                    prev_date = team_matches[i - 1][1]
                    # Compute calendar days between
                    from datetime import datetime
                    try:
                        d1 = datetime.strptime(str(prev_date), "%Y-%m-%d")
                        d2 = datetime.strptime(str(date), "%Y-%m-%d")
                        return (d2 - d1).days - 1  # Rest days (excluding match days)
                    except:
                        return 0

        return 0

    def _kpi_1_7_entry_visa_restriction(
        self, schedule: Dict, base_camp_assignment: Dict
    ) -> float:
        """
        KPI 1.7: Entry and Visa Restriction Exposure (ERI).
        Formula: ERI = Σ_m 𝟙[v(m) ∈ V^US] · max(σ(i(m)), σ(j(m)))
        Where σ(t) = 1.0 if t ∈ F^ban, 0.5 if t ∈ F^bond, 0 otherwise.
        """
        us_stadiums = self.venues[self.venues["country"] == "USA"]["venue_id"].tolist()
        ban_teams = set(self.params.get("us_visa_ban_teams", self.loader.config_params.US_VISA_BAN_TEAMS))
        bond_teams = set(self.params.get("us_visa_bond_teams", self.loader.config_params.US_VISA_BOND_TEAMS))

        eri = 0.0
        for match_id, (slot, stadium_id) in schedule.items():
            if stadium_id in us_stadiums:
                match = self.matches[self.matches["match_id"] == match_id].iloc[0]
                team_a = match["team_a_id"]
                team_b = match["team_b_id"]

                # Compute severity weights
                sigma_a = 1.0 if team_a in ban_teams else (0.5 if team_a in bond_teams else 0.0)
                sigma_b = 1.0 if team_b in ban_teams else (0.5 if team_b in bond_teams else 0.0)

                # Add maximum severity to ERI
                eri += max(sigma_a, sigma_b)

        return eri

    def _kpi_2_2_per_team_heat_load(self, schedule: Dict) -> float:
        """
        KPI 2.2: Per-Team Heat Load.
        Formula: HL_i = Σ_k max(0, WBGT_m(i,k) - 28)
        WBGT calculation requires humidity/radiation data. Using temperature-based estimate:
        WBGT_outdoor ≈ 0.7*T_nwb + 0.2*T_g + 0.1*T_db
        For simplicity with available data: WBGT ≈ 0.5*T_db + 14
        """
        threshold = 28.0  # °C FIFPRO safety threshold
        total_heat_load = {}

        for team_id in self.teams["team_id"].unique():
            team_heat_load = 0.0
            for match_id, (slot, stadium_id) in schedule.items():
                match = self.matches[self.matches["match_id"] == match_id].iloc[0]
                if team_id not in [match["team_a_id"], match["team_b_id"]]:
                    continue

                # Get temperature for venue
                venue_weather = self.params["weather"][
                    self.params["weather"]["venue_id"] == stadium_id
                ]
###################################################################
                if len(venue_weather) > 0:
                    T_db = venue_weather["temperature_c"].mean()
                    # WBGT estimate (from KPIs.tex section on simplified formulation)
                    wbgt_estimated = 0.5 * T_db + 14.0
                    excess = max(0.0, wbgt_estimated - threshold)
                    team_heat_load += excess

            total_heat_load[team_id] = team_heat_load

        return max(total_heat_load.values()) 

    def _kpi_3_3_first_mover_balance(self, schedule: Dict) -> float:
        """
        KPI 3.3: First-Mover Balance Index (Round-Order Balance).
        Formula: a_i = Σ_r pos(i,r) where pos(i,r) = 1 if team i plays first in round r.
        Then FMB = σ({a_i : i ∈ T}) (standard deviation).
        Applies only to rounds 1-2; round 3 is simultaneous.
        """
        fmb = []
        # Map matches to rounds
        groups = {}
        for _, match in self.matches.iterrows():
            group = match["group"]
            if group not in groups:
                groups[group] = []
            groups[group].append(match["match_id"])

        # Compute first-mover advantage score per team
        first_mover_scores = {}
        for team_id in self.teams["team_id"].unique():
            first_mover_scores[team_id] = 0

        group_matches_sorted = {}
        # For each group, identify first/second match in rounds 1-2
        for group_id, group_matches in groups.items():
            teams = set()
            # sort based on 
            for mid in group_matches:
                slot, stadium_id = schedule[mid]
                kickoff_date, kickoff_time = slot  # (date, time)
                group_matches_sorted[mid] = (kickoff_date, kickoff_time)

            # sort based on date and then time
            group_matches_sorted = dict(sorted(group_matches_sorted.items(), key=lambda x: (x[1][0], x[1][1])))
            for round_idx, match_id in enumerate(group_matches[:4]):
                match = self.matches[self.matches["match_id"] == match_id].iloc[0]
                team_a = match["team_a_id"]
                team_b = match["team_b_id"]
                teams.add(team_a)
                teams.add(team_b)

                first_mover_scores[team_a] += round_idx 
                first_mover_scores[team_b] += round_idx 
            
            # Compute standard deviation (fairness metric)
            scores = list([first_mover_scores[team] for team in teams])
            if len(scores) > 1:
                fmb.append(np.std(scores))

        return max(fmb) if fmb else 0.0

    def _kpi_4_1_venue_load_balance(self, schedule: Dict) -> float:
        """
        KPI 4.1: Venue-Load Balance.
        Formula: VLB = σ({n_v}) / μ({n_v})
        Where n_v = number of group-stage matches at venue v.
        Minimize coefficient of variation.
        """
        venue_counts = {}
        for match_id, (slot, stadium_id) in schedule.items():
            venue_counts[stadium_id] = venue_counts.get(stadium_id, 0) + 1

        counts = list(venue_counts.values())
        if len(counts) > 1 and np.mean(counts) > 0:
            vlb = np.std(counts) / np.mean(counts)
        else:
            vlb = 0.0

        return vlb


#############################################################
    def _kpi_4_2_fan_accessibility(self, schedule: Dict) -> float:
        """
        KPI 4.2: Fan Accessibility and Same-City Overlap.
        Formula: SCO = Σ_{m,m'} 𝟙[city(m)=city(m') ∧ d(m)=d(m')]
        Count matches in same city on same date (operational issue).
        """
        same_city_overlap = 0

        # Get city mappings
        venue_city = {}
        for _, venue in self.venues.iterrows():
            venue_city[venue["venue_id"]] = venue.get("city", "Unknown")

        # Check for same-city, same-date matches
        match_list = list(schedule.items())
        for i, (m1_id, (slot1, v1)) in enumerate(match_list):
            m1 = self.matches[self.matches["match_id"] == m1_id].iloc[0]
            city1 = venue_city.get(v1, "")
            date1 = m1["date"]

            for j in range(i + 1, len(match_list)):
                m2_id, (slot2, v2) = match_list[j]
                m2 = self.matches[self.matches["match_id"] == m2_id].iloc[0]
                city2 = venue_city.get(v2, "")
                date2 = m2["date"]

                if city1 == city2 and date1 == date2:
                    same_city_overlap += 1

        return float(same_city_overlap)


    def _kpi_5_2_marquee_match_quality(self, schedule: Dict) -> float:
        """
        KPI 5.2: Marquee-Match Slot Quality.
        Formula: MSQ = Σ_m μ_ij · q(s(m))
        Where μ_ij = (E_i^-1 + E_j^-1) / 2 (stronger teams = lower Elo = higher score).
        Primetime bonus: q(s) = 1 if in primetime, 0.5 otherwise.
        Maximize MSQ.
        """
        primetime_window = (19.0, 23.0)
        msq = 0.0

        for match_id, (slot, stadium_id) in schedule.items():
            # Compute match strength score (inverse Elo: lower Elo = higher score)
            mu_ij = self.params["match_value"].get(match_id, 0.5)

            # Slot quality (primetime bonus)
            q_s = 0
            if isinstance(slot, tuple):
                _, kickoff_time = slot
                try:
                    kickoff_hour = float(kickoff_time.split(":")[0])
                    if primetime_window[0] <= kickoff_hour <= primetime_window[1]:
                        q_s = 1.0*self.params["popularity"].get(match_id, 1.0)  # Primetime bonus scaled by match popularity
                except:
                    pass

            msq += mu_ij * q_s

        # penalty for simultaneous popular matches (if two matches have high popularity in the same slot, we reduce the score)
        # get top 20 percent of matches by popularity
        popularity_threshold = np.percentile(list(self.params["popularity"].values()), 80)
        # for mathces with popularity above threshold, check if they are in the same slot and reduce score if so
        for match_id, (slot, stadium_id) in schedule.items():
            if self.params["popularity"].get(match_id, 0) >= popularity_threshold:
                for other_match_id, (other_slot, other_stadium_id) in schedule.items():
                    if match_id != other_match_id and self.params["popularity"].get(other_match_id, 0) >= popularity_threshold:
                        if slot == other_slot:  # Simultaneous matches
                            msq -= np.average([self.params["popularity"].get(match_id, 0), self.params["popularity"].get(other_match_id, 0)])
        

        return -msq  # Negative for minimization objective

    def _kpi_5_3_host_city_economic_equity(self, schedule: Dict) -> float:
        """
        KPI 5.3: Host-City Economic Equity.
        Formula: HCE = 1 - Gini({VC_v})
        Where VC_v = Σ_{m:v(m)=v} q(s(m)) · μ_ij(m).
        Gini coefficient ∈ [0,1]: higher HCE = more equal distribution.
        Maximize HCE (minimize Gini inequality).
        """
        venue_commercial = {}

        for match_id, (slot, stadium_id) in schedule.items():
            if stadium_id not in venue_commercial:
                venue_commercial[stadium_id] = 0.0

            mu_ij = self.params["match_value"].get(match_id, 0.5)

            # Slot quality (primetime bonus)
            q_s = 0
            if isinstance(slot, tuple):
                _, kickoff_time = slot
                try:
                    kickoff_hour = float(kickoff_time.split(":")[0])
                    if 19.0 <= kickoff_hour <= 23.0:
                        q_s = 1.0*self.params["popularity"].get(match_id, 1.0)  # Primetime bonus scaled by match popularity
                except:
                    pass

            venue_commercial[stadium_id] += q_s * mu_ij

        # Compute Gini coefficient
        values = list(venue_commercial.values())
        if len(values) > 1:
            gini = self._gini_coefficient(values)
        else:
            gini = 0.0

        return -gini  # Negative for minimization (opposite of maximize)

    @staticmethod
    def _gini_coefficient(values):
        """Compute Gini coefficient of a list of values."""
        vals = np.sort(np.asarray(values))
        n = len(vals)
        if n == 0 or vals.sum() == 0:
            return 0.0
        i = np.arange(1, n + 1)
        return (2 * np.sum(i * vals)) / (n * vals.sum()) - (n + 1) / n

