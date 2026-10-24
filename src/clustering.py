"""
Unsupervised Planetary Clustering Engine
========================================
Implements K-Means clustering with automated hyperparameter optimization
(Elbow method, Silhouette analysis, Davies-Bouldin index) and physical
astronomical cluster profiling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_cluster_metrics(X: np.ndarray, k_range: range = range(2, 9)) -> Dict[str, List[float]]:
    """
    Evaluate clustering quality metrics across a spectrum of cluster counts:
    - Inertia (Within-Cluster Sum of Squares)
    - Silhouette Score (Separation & cohesion, -1 to 1, higher is better)
    - Davies-Bouldin Index (Lower is better)
    """
    metrics = {
        "k_values": list(k_range),
        "inertias": [],
        "silhouette_scores": [],
        "davies_bouldin_scores": []
    }

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = km.fit_predict(X)
        metrics["inertias"].append(float(km.inertia_))
        metrics["silhouette_scores"].append(float(silhouette_score(X, labels)))
        metrics["davies_bouldin_scores"].append(float(davies_bouldin_score(X, labels)))

    return metrics


class ExoplanetKMeans:
    """
    Production K-Means model for planetary archetype discovery.
    Provides automated k-selection, centroid physical profiling, and archetype tagging.
    """
    def __init__(self, n_clusters: Optional[int] = None, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model: Optional[KMeans] = None
        self.optimal_k: int = n_clusters or 4
        self.cluster_names: Dict[int, str] = {}

    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 9)) -> int:
        """Find k maximizing the Silhouette Score."""
        metrics = evaluate_cluster_metrics(X, k_range)
        best_idx = int(np.argmax(metrics["silhouette_scores"]))
        self.optimal_k = metrics["k_values"][best_idx]
        return self.optimal_k

    def fit(self, X: np.ndarray, df_context: Optional[pd.DataFrame] = None):
        """Fit K-Means clustering with chosen or optimized k."""
        k = self.n_clusters or self.optimal_k
        self.model = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
        labels = self.model.fit_predict(X)

        if df_context is not None:
            self._generate_cluster_names(df_context, labels)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
        if self.model is None:
            raise ValueError("Model is not fitted.")
        return self.model.predict(X)

    def _generate_cluster_names(self, df: pd.DataFrame, labels: np.ndarray):
        """
        Assign scientifically meaningful cluster archetype names based on
        median radius, insolation, and density of each cluster.
        """
        temp_df = df.copy()
        temp_df["cluster"] = labels

        agg_cols = ["pl_rade", "pl_insol"]
        if "pl_dens" in temp_df.columns:
            agg_cols.append("pl_dens")
        elif "log_dens" in temp_df.columns:
            temp_df["pl_dens"] = 10 ** temp_df["log_dens"]
            agg_cols.append("pl_dens")

        grouped = temp_df.groupby("cluster")[agg_cols].median()

        for c_id, row in grouped.iterrows():
            r = row["pl_rade"]
            insol = row["pl_insol"]
            dens = row.get("pl_dens", 3.0)

            if r >= 6.0 and insol >= 10.0:
                name = f"Cluster {c_id}: Hot Gas Giants (Jovian)"
            elif r >= 6.0 and insol < 10.0:
                name = f"Cluster {c_id}: Cold Gas Giants"
            elif 1.6 <= r < 6.0:
                name = f"Cluster {c_id}: Sub-Neptunes & Mini-Neptunes"
            elif r < 1.6 and 0.25 <= insol <= 2.5:
                name = f"Cluster {c_id}: Temperate Terrestrial / Earth-like"
            elif r < 1.6 and insol > 2.5:
                name = f"Cluster {c_id}: Ultra-Hot Rocky & Lava Worlds"
            else:
                name = f"Cluster {c_id}: Cold Terrestrial Outliers"

            self.cluster_names[c_id] = name

    def get_cluster_name(self, cluster_id: int) -> str:
        """Get the human-interpretable astronomical archetype name."""
        return self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
