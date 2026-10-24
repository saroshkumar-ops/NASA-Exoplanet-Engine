"""
Reusable Dashboard UI Components
================================
Provides header layouts, telemetry KPI metric cards, and dynamic sidebar controls.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSS_PATH = PROJECT_ROOT / "assets" / "style.css"


def inject_custom_css():
    """Inject custom observatory CSS styles into Streamlit."""
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header():
    """Render application header banner."""
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
                <span style="font-size: 2rem;">🪐</span>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    ExoTerra: NASA Exoplanet Characterization Engine
                </h1>
            </div>
            <p style="margin: 0; color: #94a3b8; font-size: 0.95rem;">
                Production Machine Learning, Habitable Zone Astrobiology, and Latent Space Explorer powered by the NASA Exoplanet Archive
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_kpi_cards(df: pd.DataFrame):
    """Render top summary telemetry cards."""
    total_planets = len(df)
    hz_planets = len(df[df["hz_status"].isin(["Conservative HZ", "Optimistic HZ (Inner)", "Optimistic HZ (Outer)"])])
    conservative_hz = len(df[df["hz_status"] == "Conservative HZ"])
    top_esi_candidate = df.sort_values("esi_global", ascending=False).iloc[0]
    solar_neighbors = len(df[df["sy_dist"] <= 20.0])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Catalog Observations</div>
                <div class="metric-value">{total_planets:,}</div>
                <div class="metric-caption">Confirmed NASA Exoplanets</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Habitable Zone Worlds</div>
                <div class="metric-value" style="color: #4ade80;">{hz_planets}</div>
                <div class="metric-caption">{conservative_hz} Conservative Goldilocks</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Highest ESI Candidate</div>
                <div class="metric-value" style="color: #60a5fa; font-size: 1.5rem;">{top_esi_candidate['pl_name']}</div>
                <div class="metric-caption">ESI: {top_esi_candidate['esi_global']:.3f} (Earth = 1.0)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Nearby Solar Neighbors</div>
                <div class="metric-value" style="color: #f472b6;">{solar_neighbors}</div>
                <div class="metric-caption">Within 20 Parsecs (~65 Light Years)</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered DataFrame."""
    st.sidebar.markdown("### 🔭 Exploration Filters")

    # Habitable Zone selector
    hz_options = ["All Observations", "Any Habitable Zone", "Conservative HZ Only"]
    selected_hz = st.sidebar.selectbox("Habitable Zone Screening", hz_options, index=0)

    # Radius slider
    r_max = float(min(df["pl_rade"].quantile(0.99), 25.0))
    radius_range = st.sidebar.slider(
        "Planet Radius (Earth Radii)",
        min_value=0.1,
        max_value=r_max,
        value=(0.5, 3.0),
        step=0.1
    )

    # Insolation slider (logarithmic scale)
    insol_range = st.sidebar.slider(
        "Stellar Insolation (Earth = 1.0)",
        min_value=0.01,
        max_value=100.0,
        value=(0.1, 10.0),
        step=0.1
    )

    # Discovery methods
    all_methods = sorted(df["discoverymethod"].dropna().unique().tolist())
    selected_methods = st.sidebar.multiselect(
        "Discovery Technique",
        all_methods,
        default=all_methods[:3] if len(all_methods) >= 3 else all_methods
    )

    # Apply filters
    filtered = df.copy()

    if selected_hz == "Any Habitable Zone":
        filtered = filtered[filtered["hz_status"].isin(["Conservative HZ", "Optimistic HZ (Inner)", "Optimistic HZ (Outer)"])]
    elif selected_hz == "Conservative HZ Only":
        filtered = filtered[filtered["hz_status"] == "Conservative HZ"]

    filtered = filtered[
        (filtered["pl_rade"] >= radius_range[0]) & (filtered["pl_rade"] <= radius_range[1]) &
        (filtered["pl_insol"] >= insol_range[0]) & (filtered["pl_insol"] <= insol_range[1])
    ]

    if selected_methods:
        filtered = filtered[filtered["discoverymethod"].isin(selected_methods)]

    st.sidebar.markdown(f"**Filtered Candidates:** `{len(filtered):,}` of `{len(df):,}`")
    return filtered
