"""
Machine Learning Lab & Real-Time Planetary Simulator View
=========================================================
Interactive PCA latent manifold, K-Means & GMM clustering,
Isolation Forest anomaly hunter, and real-time candidate scanner.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from src.features import ExoplanetFeaturePipeline
from src.clustering import ExoplanetKMeans
from src.probabilistic_models import ExoplanetGMM
from src.dimensionality import ExoplanetLatentSpace
from src.anomaly import ExoplanetAnomalyDetector
from src.visualization import plot_latent_space_3d, plot_esi_radar_chart
from src.physics import (
    calculate_esi,
    determine_hz_status,
    classify_planet_regime,
    estimate_planetary_density,
    calculate_escape_velocity,
    EARTH_DENSITY_G_CM3,
    EARTH_TSURF_K,
    EARTH_TEQ_K
)


def render_ml_lab_view(df: pd.DataFrame, pipeline, X_scaled, kmeans, gmm, pca, detector):
    """Render Tab 3: Machine Learning & Latent Space."""
    st.markdown("### 🤖 Unsupervised Machine Learning & Latent Space")
    st.write(
        "Explore how unsupervised algorithms segment the cosmos into physical archetypes "
        "and isolate extreme astrophysical anomalies using multi-dimensional manifold learning."
    )

    t1, t2, t3 = st.tabs(["🌌 3D PCA Latent Space", "📊 Cluster Archetypes & GMM", "🚨 Exotic Anomaly Hunter"])

    with t1:
        st.markdown("#### Principal Component Analysis (3D Latent Manifold)")
        pca_fig = plot_latent_space_3d(df, pca.transform(X_scaled), color_col="cluster_archetype")
        st.plotly_chart(pca_fig, use_container_width=True)

        evr = pca.explained_variance_ratio
        st.info(
            f"**Information Compression:** PC1 captures `{evr[0]*100:.1f}%`, PC2 captures `{evr[1]*100:.1f}%`, "
            f"and PC3 captures `{evr[2]*100:.1f}%` of total astronomical variance "
            f"(Cumulative: `{pca.cumulative_explained_variance*100:.1f}%`)."
        )

        st.markdown("##### Principal Component Feature Loadings")
        st.dataframe(pca.get_loadings_dataframe().round(3), use_container_width=True)

    with t2:
        st.markdown("#### K-Means Discovered Planetary Archetypes")
        summary = df.groupby("cluster_archetype").agg(
            planet_count=("pl_name", "count"),
            median_radius=("pl_rade", "median"),
            median_mass=("pl_bmasse", "median"),
            median_insol=("pl_insol", "median"),
            median_esi=("esi_global", "median")
        ).reset_index()
        st.dataframe(summary.round(2), use_container_width=True)

        st.markdown("#### Probabilistic Soft-Clustering (GMM) Classification Entropy")
        st.write(
            "Higher Shannon entropy denotes transition planets located on continuous boundaries "
            "(e.g., Fulton radius valley between rocky Super-Earths and gaseous Sub-Neptunes)."
        )
        entropy = gmm.compute_classification_entropy(X_scaled)
        df_ent = df.copy()
        df_ent["entropy"] = entropy

        ent_fig = px.histogram(
            df_ent,
            x="entropy",
            nbins=40,
            template="plotly_dark",
            title="Classification Uncertainty Distribution (Shannon Entropy)",
            color_discrete_sequence=["#818cf8"]
        )
        st.plotly_chart(ent_fig, use_container_width=True)

    with t3:
        st.markdown("#### 🚨 Isolation Forest: Top 10 Most Exotic Outlier Exoplanets")
        st.write(
            "Isolates statistical anomalies in few partitions. Flags rare worlds such as "
            "ultra-dense iron chthonian cores, super-puff planets, and extreme short-period worlds."
        )
        anomalies = detector.get_top_anomalies(df, X_scaled, top_n=10)
        disp_anom = anomalies[[
            "pl_name", "anomaly_score", "pl_rade", "pl_bmasse",
            "pl_dens", "pl_insol", "st_teff", "discoverymethod"
        ]].copy()
        disp_anom.columns = [
            "Planet", "Anomaly Score (0-100)", "Radius (R⊕)", "Mass (M⊕)",
            "Density (g/cm³)", "Insolation (S⊕)", "Star Teff (K)", "Method"
        ]
        st.dataframe(disp_anom.round(2), use_container_width=True)


def render_simulator_view(pipeline, kmeans, detector):
    """Render Tab 4: Interactive Custom Planetary Simulator."""
    st.markdown("### 🧪 Real-Time Planetary Candidate Scanner")
    st.write(
        "Design a custom or hypothetical exoplanet, or input newly released observational data. "
        "The trained pipeline will execute real-time feature engineering, ESI calculation, "
        "habitable zone screening, and cluster classification."
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Planetary Parameters")
        sim_r = st.slider("Planet Radius (Earth Radii)", 0.2, 15.0, 1.05, 0.05)
        sim_m = st.slider("Planet Mass (Earth Masses)", 0.1, 50.0, 1.15, 0.1)
        sim_insol = st.slider("Stellar Insolation Flux (Earth = 1.0)", 0.05, 20.0, 1.02, 0.05)
        sim_teff = st.slider("Host Star Temperature (Kelvin)", 2600.0, 7200.0, 5750.0, 50.0)

        # Derived physical attributes
        sim_dens = estimate_planetary_density(sim_r, sim_m)
        sim_vesc = calculate_escape_velocity(sim_m, sim_r)
        sim_teq = EARTH_TEQ_K * (sim_insol ** 0.25)
        sim_tsurf = sim_teq * (EARTH_TSURF_K / EARTH_TEQ_K)

    with c2:
        st.markdown("#### ⚡ Real-Time ML & Astrobiological Assessment")

        # Feature transform
        sample_df = pd.DataFrame([{
            "log_rade": np.log10(sim_r),
            "log_insol": np.log10(sim_insol),
            "log_bmasse": np.log10(sim_m),
            "log_dens": np.log10(sim_dens),
            "st_teff": sim_teff
        }])

        X_sim = pipeline.transform(sample_df)
        pred_cluster = kmeans.predict(X_sim)[0]
        archetype_name = kmeans.get_cluster_name(pred_cluster)
        anomaly_score = detector.compute_anomaly_index(X_sim)[0]

        esi_glob, esi_int, esi_surf = calculate_esi(sim_r, sim_dens, sim_vesc, sim_tsurf)
        hz_status = determine_hz_status(sim_insol, sim_teff)
        regime = classify_planet_regime(sim_r)

        # Result badge
        hz_color = "#2ecc71" if "Conservative" in hz_status else "#f39c12" if "Optimistic" in hz_status else "#e74c3c"

        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <h3 style="margin: 0 0 10px 0; color: #38bdf8;">Scan Verdict: {archetype_name}</h3>
                <p style="margin: 4px 0;"><strong>Regime:</strong> {regime}</p>
                <p style="margin: 4px 0;"><strong>Habitable Zone:</strong> <span style="color: {hz_color}; font-weight: 700;">{hz_status}</span></p>
                <p style="margin: 4px 0;"><strong>Global Earth Similarity (ESI):</strong> <span style="font-size: 1.25rem; font-weight: 800; color: #4ade80;">{esi_glob:.4f}</span></p>
                <p style="margin: 4px 0;"><strong>Exotic Anomaly Score:</strong> {anomaly_score:.1f} / 100</p>
                <p style="margin: 4px 0;"><strong>Est. Bulk Density:</strong> {sim_dens:.2f} g/cm³ (Earth = 5.51)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Simulator radar chart
    sim_data = pd.DataFrame([{
        "pl_name": "Custom Simulation Candidate",
        "pl_rade": sim_r,
        "pl_dens": sim_dens,
        "pl_vesc_rel": sim_vesc,
        "pl_tsurf_est": sim_tsurf,
        "esi_global": esi_glob
    }])
    st.markdown("#### 📡 Real-Time Similarity Radar vs. Earth")
    sim_radar = plot_esi_radar_chart("Custom Simulation Candidate", sim_data)
    st.plotly_chart(sim_radar, use_container_width=True)
