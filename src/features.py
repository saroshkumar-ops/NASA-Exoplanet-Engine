"""
Feature Preprocessing and Pipeline Engine
==========================================
Engineers standardized, scale-invariant feature matrices for
unsupervised clustering and anomaly detection algorithms.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler

DEFAULT_FEATURE_COLS = [
    "log_rade",     # Log10 Planet Radius [Earth radii]
    "log_insol",    # Log10 Insolation Flux [Earth solar flux]
    "log_bmasse",   # Log10 Planet Mass [Earth masses]
    "log_dens",     # Log10 Bulk Density [g/cm^3]
    "st_teff"       # Host Star Effective Temperature [K]
]

class ExoplanetFeaturePipeline:
    """
    Scikit-learn compliant feature transformer for astronomical data.
    Standardizes logarithmic and physical properties across multiple orders of magnitude.
    """
    def __init__(self, feature_cols: Optional[List[str]] = None, scaler_type: str = "standard"):
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit scaler on specified feature columns."""
        X = df[self.feature_cols].copy()
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform exoplanet feature matrix."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming data.")
        X = df[self.feature_cols].copy()
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform feature matrix."""
        return self.fit(df).transform(df)

    def inverse_transform(self, X_scaled: np.ndarray) -> pd.DataFrame:
        """Reverse scaling to restore original physical units."""
        raw = self.scaler.inverse_transform(X_scaled)
        return pd.DataFrame(raw, columns=self.feature_cols)

    def get_feature_names(self) -> List[str]:
        """Return the names of the active feature columns."""
        return self.feature_cols.copy()
