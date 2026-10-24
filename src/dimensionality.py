"""
Latent Space Dimensionality Reduction
=====================================
Implements Principal Component Analysis (PCA) to compress high-dimensional
astrophysical parameter spaces into 2D and 3D latent embeddings.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.decomposition import PCA

class ExoplanetLatentSpace:
    """
    Computes 2D and 3D latent projections of exoplanetary parameter manifolds.
    Extracts explained variance ratios and physical feature loadings.
    """
    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit PCA on standardized feature matrix."""
        self.pca.fit(X)
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X.shape[1])]
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project high-dimensional observations into latent space."""
        if not self.is_fitted:
            raise ValueError("PCA model is not fitted.")
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform feature matrix."""
        return self.fit(X, feature_names).transform(X)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Percentage of variance captured by each principal component."""
        return self.pca.explained_variance_ratio_

    @property
    def cumulative_explained_variance(self) -> float:
        """Total variance captured across all retained components."""
        return float(np.sum(self.pca.explained_variance_ratio_))

    def get_loadings_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame showing how much each astrophysical variable
        contributes to each principal component.
        """
        if not self.is_fitted:
            raise ValueError("PCA model is not fitted.")
        
        cols = [f"PC{i+1}" for i in range(self.n_components)]
        loadings = self.pca.components_.T
        return pd.DataFrame(loadings, index=self.feature_names, columns=cols)
