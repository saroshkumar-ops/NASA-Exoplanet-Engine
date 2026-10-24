"""
Unit Tests for Machine Learning Models & Latent Space
=====================================================
Validates feature transformation, K-Means clustering, GMM, PCA,
and Isolation Forest anomaly detection.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import pytest
import numpy as np
import pandas as pd
from src.features import ExoplanetFeaturePipeline
from src.clustering import ExoplanetKMeans, evaluate_cluster_metrics
from src.probabilistic_models import ExoplanetGMM
from src.dimensionality import ExoplanetLatentSpace
from src.anomaly import ExoplanetAnomalyDetector

@pytest.fixture
def synthetic_features_df():
    """Generate synthetic exoplanet dataset for ML testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "pl_name": [f"Planet-{i}" for i in range(n)],
        "pl_rade": np.random.uniform(0.5, 15.0, size=n),
        "pl_insol": np.random.uniform(0.1, 1000.0, size=n),
        "pl_dens": np.random.uniform(0.5, 8.0, size=n),
        "log_rade": np.random.normal(0.2, 0.5, size=n),
        "log_insol": np.random.normal(1.0, 1.2, size=n),
        "log_bmasse": np.random.normal(0.8, 1.0, size=n),
        "log_dens": np.random.normal(0.5, 0.4, size=n),
        "st_teff": np.random.uniform(3000.0, 6500.0, size=n)
    })


def test_feature_pipeline(synthetic_features_df):
    """Test standard scaling and transformation shapes."""
    pipeline = ExoplanetFeaturePipeline()
    X_scaled = pipeline.fit_transform(synthetic_features_df)

    assert X_scaled.shape == (100, 5)
    # Check approximately zero mean and unit variance
    assert np.allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-2)
    assert np.allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-2)


def test_kmeans_clustering(synthetic_features_df):
    """Test K-Means model, metric evaluation, and archetype naming."""
    pipeline = ExoplanetFeaturePipeline()
    X = pipeline.fit_transform(synthetic_features_df)

    metrics = evaluate_cluster_metrics(X, k_range=range(2, 5))
    assert len(metrics["silhouette_scores"]) == 3
    assert all(-1 <= s <= 1 for s in metrics["silhouette_scores"])

    kmeans = ExoplanetKMeans(n_clusters=3)
    kmeans.fit(X, df_context=synthetic_features_df)
    preds = kmeans.predict(X)

    assert len(preds) == 100
    assert set(preds) == {0, 1, 2}
    assert len(kmeans.cluster_names) == 3


def test_gmm_probabilistic_clustering(synthetic_features_df):
    """Test Gaussian Mixture Model soft probabilities and entropy."""
    pipeline = ExoplanetFeaturePipeline()
    X = pipeline.fit_transform(synthetic_features_df)

    gmm = ExoplanetGMM(n_components=3)
    gmm.fit(X)
    probs = gmm.predict_proba(X)
    entropy = gmm.compute_classification_entropy(X)

    assert probs.shape == (100, 3)
    # Probabilities must sum to 1.0 across clusters
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)
    assert len(entropy) == 100
    assert (entropy >= 0.0).all()


def test_pca_dimensionality_reduction(synthetic_features_df):
    """Test PCA 2D/3D embeddings and explained variance."""
    pipeline = ExoplanetFeaturePipeline()
    X = pipeline.fit_transform(synthetic_features_df)

    pca = ExoplanetLatentSpace(n_components=3)
    X_pca = pca.fit_transform(X, pipeline.get_feature_names())

    assert X_pca.shape == (100, 3)
    assert len(pca.explained_variance_ratio) == 3
    assert 0.0 < pca.cumulative_explained_variance <= 1.0
    
    loadings = pca.get_loadings_dataframe()
    assert loadings.shape == (5, 3)


def test_isolation_forest_anomaly_detection(synthetic_features_df):
    """Test anomaly detection, score range, and top outlier extraction."""
    pipeline = ExoplanetFeaturePipeline()
    X = pipeline.fit_transform(synthetic_features_df)

    detector = ExoplanetAnomalyDetector(contamination=0.05)
    detector.fit(X)
    scores = detector.compute_anomaly_index(X)
    preds = detector.predict(X)

    assert len(scores) == 100
    assert (scores >= 0.0).all() and (scores <= 100.0).all()
    assert set(preds).issubset({-1, 1})

    top_anomalies = detector.get_top_anomalies(synthetic_features_df, X, top_n=5)
    assert len(top_anomalies) == 5
    assert "anomaly_score" in top_anomalies.columns
