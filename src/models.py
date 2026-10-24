"""
Unified Machine Learning Module Facade
======================================
Re-exports key machine learning architectures:
- ExoplanetFeaturePipeline (Preprocessing)
- ExoplanetKMeans (Unsupervised Clustering)
- ExoplanetGMM (Probabilistic Soft-Clustering)
- ExoplanetLatentSpace (PCA Dimensionality Reduction)
- ExoplanetAnomalyDetector (Isolation Forest Outlier Detection)
"""

from src.features import ExoplanetFeaturePipeline, DEFAULT_FEATURE_COLS
from src.clustering import ExoplanetKMeans, evaluate_cluster_metrics
from src.probabilistic_models import ExoplanetGMM, evaluate_gmm_bic_aic
from src.dimensionality import ExoplanetLatentSpace
from src.anomaly import ExoplanetAnomalyDetector

__all__ = [
    "ExoplanetFeaturePipeline",
    "DEFAULT_FEATURE_COLS",
    "ExoplanetKMeans",
    "evaluate_cluster_metrics",
    "ExoplanetGMM",
    "evaluate_gmm_bic_aic",
    "ExoplanetLatentSpace",
    "ExoplanetAnomalyDetector"
]
