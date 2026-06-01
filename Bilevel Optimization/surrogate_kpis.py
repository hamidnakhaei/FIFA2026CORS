"""
Surrogate KPI model for Opt+Sim framework
Uses calibrated features to predict base-camp-dependent KPI values
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


class SurrogateKPIModel:
    """
    Calibrated linear surrogate for base-camp-dependent KPIs.
    
    Features are camp-free schedule characteristics:
    - Venue cluster diversity (CC_i): dispersion across FIFA clusters
    - Venue spread (SP_i): pairwise distances between team's venues
    - US venue load (USL_i): number of matches in USA (for visa-restricted teams)
    
    Weights are fitted by least-squares regression against simulated KPI values.
    """
    
    def __init__(self, data_loader, parameters):
        """
        Initialize the surrogate model.
        
        Args:
            data_loader: DataLoader instance
            parameters: ParameterBuilder with precomputed constants
        """
        self.data = data_loader
        self.params = parameters
        
        # Create index mappings from DataFrames
        match_ids = data_loader.matches['match_id'].tolist()
        venue_ids = data_loader.venues['venue_id'].tolist()
        self.match_id_to_idx = {mid: i for i, mid in enumerate(match_ids)}
        self.venue_id_to_idx = {vid: i for i, vid in enumerate(venue_ids)}
        self.venue_idx_to_id = {i: vid for i, vid in enumerate(venue_ids)}
        
        # Fitted weights (w1, w2, w3) for the surrogate
        self.weights = np.array([1.0, 1.0, 1.0])  # Initialize to equal weights
        
        # Regression models per KPI (simple linear: y = w·x)
        self.kpi_weights = {
            'travel': np.array([1.0, 1.0, 1.0]),
            'jet_lag': np.array([1.0, 1.0, 1.0]),
            'border_crossings': np.array([1.0, 1.0, 1.0]),
            'altitude': np.array([1.0, 1.0, 1.0])
        }
        
        # Whether model is fitted
        self.is_fitted = False
        
        # History of (features, kpi_values) pairs for fitting
        self.history = []
    
    def extract_features(self, venue_assignment: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract camp-free schedule features for all teams.
        
        Args:
            venue_assignment: Array a_ms indexed by [match_idx, venue_idx]
        
        Returns:
            Dict mapping team_id -> feature vector [CC_i, SP_i, USL_i]
        """
        features = {}
        
        for team_id in self.data.team_by_id.keys():
            cc_i = self._compute_cluster_diversity(team_id, venue_assignment)
            sp_i = self._compute_venue_spread(team_id, venue_assignment)
            usl_i = self._compute_us_load(team_id, venue_assignment)
            
            features[team_id] = np.array([cc_i, sp_i, usl_i])
        
        return features
    
    def _compute_cluster_diversity(self, team_id: str, venue_assignment: np.ndarray) -> float:
        """
        Compute venue-cluster diversity CC_i.
        
        Counts distinct FIFA clusters among the team's match venues.
        
        Args:
            team_id: Team identifier
            venue_assignment: Array a_ms
        
        Returns:
            Cluster diversity score (0-3)
        """
        team_matches = self.data.get_team_matches(team_id)
        clusters = set()
        
        for match_dict in team_matches:
            match_id = match_dict['match_id']
            match_idx = self.match_id_to_idx.get(match_id)
            
            if match_idx is not None:
                # Find assigned venue
                for venue_idx in range(venue_assignment.shape[1]):
                    if venue_assignment[match_idx, venue_idx] > 0.5:
                        venue_id = self.venue_idx_to_id.get(venue_idx)
                        if venue_id:
                            # Get cluster from venue data
                            venue_data = self.data.venues[self.data.venues['venue_id'] == venue_id]
                            if len(venue_data) > 0:
                                cluster = venue_data['cluster'].values[0] if 'cluster' in venue_data.columns else 'Unknown'
                                clusters.add(cluster)
                        break
        
        return float(len(clusters))
    
    def _compute_venue_spread(self, team_id: str, venue_assignment: np.ndarray) -> float:
        """
        Compute venue spread SP_i.
        
        Sum of pairwise distances between the team's match venues.
        
        Args:
            team_id: Team identifier
            venue_assignment: Array a_ms
        
        Returns:
            Venue spread (km)
        """
        team_matches = self.data.get_team_matches(team_id)
        venues = []
        
        for match_dict in team_matches:
            match_id = match_dict['match_id']
            match_idx = self.match_id_to_idx.get(match_id)
            
            if match_idx is not None:
                # Find assigned venue
                for venue_idx in range(venue_assignment.shape[1]):
                    if venue_assignment[match_idx, venue_idx] > 0.5:
                        venue_id = self.venue_idx_to_id.get(venue_idx)
                        if venue_id:
                            venues.append(venue_id)
                        break
        
        # Compute pairwise distances
        spread = 0.0
        dist_matrix = self.params.get('dist', {})
        
        for i in range(len(venues)):
            for j in range(i + 1, len(venues)):
                dist = dist_matrix.get(venues[i], {}).get(venues[j], 0)
                spread += dist
        
        return spread
    
    def _compute_us_load(self, team_id: str, venue_assignment: np.ndarray) -> float:
        """
        Compute US venue load USL_i.
        
        Number of matches scheduled in USA stadiums.
        
        Args:
            team_id: Team identifier
            venue_assignment: Array a_ms
        
        Returns:
            US match count (0-3)
        """
        team_matches = self.data.get_team_matches(team_id)
        us_count = 0
        
        for match_dict in team_matches:
            match_id = match_dict['match_id']
            match_idx = self.match_id_to_idx.get(match_id)
            
            if match_idx is not None:
                # Find assigned venue
                for venue_idx in range(venue_assignment.shape[1]):
                    if venue_assignment[match_idx, venue_idx] > 0.5:
                        venue_id = self.venue_idx_to_id.get(venue_idx)
                        if venue_id:
                            # Get country from venue data
                            venue_data = self.data.venues[self.data.venues['venue_id'] == venue_id]
                            if len(venue_data) > 0:
                                venue_country = venue_data['country'].values[0] if 'country' in venue_data.columns else None
                                if venue_country == 'USA':
                                    us_count += 1
                        break
        
        return us_count
    
    def predict_kpis(self, venue_assignment: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Predict base-camp-dependent KPI values using the surrogate.
        
        Args:
            venue_assignment: Array a_ms
        
        Returns:
            Dict with predicted KPI values:
            {
                'travel': {team_id -> predicted_value},
                'jet_lag': {team_id -> predicted_value},
                'border_crossings': {team_id -> predicted_value},
                'altitude': {team_id -> predicted_value}
            }
        """
        features = self.extract_features(venue_assignment)
        predictions = {
            'travel': {},
            'jet_lag': {},
            'border_crossings': {},
            'altitude': {}
        }
        
        for team_id, feature_vec in features.items():
            # Predict each KPI using linear model: y = w · x
            for kpi_name in ['travel', 'jet_lag', 'border_crossings', 'altitude']:
                pred_value = float(np.dot(self.kpi_weights[kpi_name], feature_vec))
                predictions[kpi_name][team_id] = max(0, pred_value)
        
        return predictions
    
    def update(self, venue_assignment: np.ndarray, simulated_kpis: Dict):
        """
        Update the surrogate model with new (features, simulated_KPI) observation.
        
        Called after each simulation to improve model fit.
        
        Args:
            venue_assignment: Array a_ms
            simulated_kpis: Dict with realized KPI values from simulation
        """
        features = self.extract_features(venue_assignment)
        
        # Store observation
        self.history.append((features, simulated_kpis))
        
        # Refit model on all history
        self._fit_regressions()
    
    def _fit_regressions(self):
        """
        Refit all KPI regressions on accumulated history using numpy least-squares.
        """
        if len(self.history) < 2:
            # Need at least 2 observations
            return
        
        # Extract X and y for each KPI
        for kpi_name in ['travel', 'jet_lag', 'border_crossings', 'altitude']:
            X_list = []
            y_list = []
            
            for features_dict, kpi_dict in self.history:
                for team_id, feature_vec in features_dict.items():
                    X_list.append(feature_vec)
                    y_list.append(kpi_dict[kpi_name].get(team_id, 0))
            
            if len(X_list) > 0:
                X = np.array(X_list)
                y = np.array(y_list)
                
                try:
                    # Solve least-squares: min ||Xw - y||^2
                    # Solution: w = (X^T X)^{-1} X^T y
                    weights = np.linalg.lstsq(X, y, rcond=None)[0]
                    self.kpi_weights[kpi_name] = weights
                except Exception as e:
                    print(f"Warning: Could not fit regression for {kpi_name}: {e}")
        
        self.is_fitted = True
    
    def get_surrogate_penalty(self, venue_assignment: np.ndarray,
                             weights: Dict[str, float]) -> float:
        """
        Compute normalized surrogate penalty for optimization objective.
        
        Args:
            venue_assignment: Array a_ms
            weights: Dict mapping KPI names to penalty weights
        
        Returns:
            Scalar penalty value
        """
        predictions = self.predict_kpis(venue_assignment)
        
        # Sum across teams and KPIs, normalized
        total_penalty = 0.0
        for kpi_name, weight in weights.items():
            if kpi_name in predictions:
                kpi_sum = sum(predictions[kpi_name].values())
                total_penalty += weight * kpi_sum
        
        return total_penalty
