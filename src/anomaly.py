"""
Novelty & Anomaly Detection (Exotic Exoplanet Hunter)
====================================================
Uses Isolation Forest to detect extreme, non-standard exoplanets
such as ultra-dense iron worlds, super-puff planets, and ultra-short-period
lava worlds that deviate from standard planetary evolution tracks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.ensemble import IsolationForest

class ExoplanetAnomalyDetector:
    """
    Unsupervised outlier detector identifying rare astronomical phenomena
    using an ensemble of isolation trees.
    """
    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=150
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """Fit Isolation Forest model."""
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary anomaly status:
        -1 for anomalies/outliers, 1 for standard exoplanets.
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector is not fitted.")
        return self.model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Raw anomaly score. Lower scores indicate more anomalous / exotic objects.
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector is not fitted.")
        return self.model.decision_function(X)

    def compute_anomaly_index(self, X: np.ndarray) -> np.ndarray:
        """
        Normalized Anomaly Index between 0 (standard) and 100 (extreme anomaly).
        """
        raw = self.decision_function(X)
        # Min-max inversion so highest score = most anomalous
        min_v = np.min(raw)
        max_v = np.max(raw)
        norm = 1.0 - (raw - min_v) / (max_v - min_v + 1e-12)
        return norm * 100.0

    def get_top_anomalies(self, df: pd.DataFrame, X: np.ndarray, top_n: int = 10) -> pd.DataFrame:
        """Extract the top N most scientifically anomalous exoplanets."""
        df_out = df.copy()
        df_out["anomaly_score"] = self.compute_anomaly_index(X)
        df_out["is_anomaly"] = self.predict(X) == -1
        return df_out.sort_values("anomaly_score", ascending=False).head(top_n)
