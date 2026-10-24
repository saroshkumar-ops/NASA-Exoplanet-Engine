"""
Probabilistic Planetary Clustering (Gaussian Mixture Models)
============================================================
Implements Gaussian Mixture Models (GMM) using Expectation-Maximization
to quantify fuzzy cluster boundaries, assignment probabilities, and
classification entropy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.mixture import GaussianMixture

def evaluate_gmm_bic_aic(X: np.ndarray, k_range: range = range(2, 8)) -> Dict[str, List[float]]:
    """
    Compute Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC)
    across candidate component counts (lower is better, penalizes complexity).
    """
    results = {
        "k_values": list(k_range),
        "bic": [],
        "aic": []
    }

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        gmm.fit(X)
        results["bic"].append(float(gmm.bic(X)))
        results["aic"].append(float(gmm.aic(X)))

    return results


class ExoplanetGMM:
    """
    Probabilistic soft-clustering for exoplanetary continuum transitions.
    Provides cluster posterior probabilities and classification entropy.
    """
    def __init__(self, n_components: int = 4, covariance_type: str = "full", random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model: Optional[GaussianMixture] = None

    def fit(self, X: np.ndarray):
        """Fit Gaussian Mixture Model."""
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=10
        )
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict hard cluster assignments (MAP estimate)."""
        if self.model is None:
            raise ValueError("GMM model is not fitted.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate soft membership probabilities across all archetypes."""
        if self.model is None:
            raise ValueError("GMM model is not fitted.")
        return self.model.predict_proba(X)

    def compute_classification_entropy(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Shannon entropy of cluster probabilities:
        H = - sum(p * log(p))
        High entropy denotes boundary planets (e.g., radius valley transition).
        """
        probs = np.clip(self.predict_proba(X), 1e-12, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return entropy
