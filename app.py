"""
ExoTerra: NASA Exoplanet Characterization & Discovery Engine
============================================================
Main Streamlit Application Entrypoint.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Streamlit Page Setup (must be first streamlit call)
st.set_page_config(
    page_title="ExoTerra | NASA Exoplanet Engine",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.data_loader import load_dataset
from src.features import ExoplanetFeaturePipeline
from src.clustering import ExoplanetKMeans
from src.probabilistic_models import ExoplanetGMM
from src.dimensionality import ExoplanetLatentSpace
from src.anomaly import ExoplanetAnomalyDetector
from src.dashboard_components import (
    inject_custom_css,
    render_header,
    render_kpi_cards,
    render_sidebar_filters
)
from src.views.explorer_view import render_explorer_view
from src.views.leaderboard_view import render_leaderboard_view
from src.views.ml_simulator_view import render_ml_lab_view, render_simulator_view


@st.cache_data(show_spinner="Connecting to NASA Exoplanet Archive...")
def get_cached_data() -> pd.DataFrame:
    """Fetch and cache enriched exoplanetary dataset."""
    df = load_dataset()
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner="Training Machine Learning Suite...")
def get_trained_models(df: pd.DataFrame):
    """Train and cache the scikit-learn feature pipeline, K-Means, GMM, PCA, and Isolation Forest."""
    pipeline = ExoplanetFeaturePipeline()
    X_scaled = pipeline.fit_transform(df)

    # 1. K-Means
    kmeans = ExoplanetKMeans(n_clusters=4)
    kmeans.fit(X_scaled, df_context=df)

    # 2. GMM
    gmm = ExoplanetGMM(n_components=4)
    gmm.fit(X_scaled)

    # 3. PCA
    pca = ExoplanetLatentSpace(n_components=3)
    pca.fit(X_scaled, feature_names=pipeline.get_feature_names())

    # 4. Isolation Forest
    detector = ExoplanetAnomalyDetector(contamination=0.015)
    detector.fit(X_scaled)

    return pipeline, X_scaled, kmeans, gmm, pca, detector


def main():
    # Inject CSS
    inject_custom_css()

    # Load Data
    df = get_cached_data()

    # Fit / Retrieve Models
    pipeline, X_scaled, kmeans, gmm, pca, detector = get_trained_models(df)

    # Attach cluster assignments to df if not present
    if "cluster_archetype" not in df.columns:
        clusters = kmeans.predict(X_scaled)
        df["cluster"] = clusters
        df["cluster_archetype"] = [kmeans.get_cluster_name(c) for c in clusters]

    # Render Header and Telemetry KPIs
    render_header()
    render_kpi_cards(df)

    # Render Sidebar Filters
    filtered_df = render_sidebar_filters(df)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌌 HZ & Composition Explorer",
        "🪐 Earth Twins Leaderboard",
        "🤖 ML & Latent Space Lab",
        "🧪 Real-Time Simulator"
    ])

    with tab1:
        render_explorer_view(filtered_df, df)

    with tab2:
        render_leaderboard_view(df)

    with tab3:
        render_ml_lab_view(df, pipeline, X_scaled, kmeans, gmm, pca, detector)

    with tab4:
        render_simulator_view(pipeline, kmeans, detector)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 12px 0;">
            ExoTerra Exoplanet Characterization Engine &bull; Built with Python, Streamlit, Scikit-Learn, Plotly & NASA Exoplanet Archive TAP API &bull; Created by Sarosh Kumar
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
