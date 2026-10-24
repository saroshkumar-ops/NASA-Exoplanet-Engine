"""
Explorer View Module
====================
Interactive Habitable Zone diagram, Mass-Radius composition tracks,
regime distribution, and filtered candidate catalog exporter.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.visualization import plot_habitable_zone_chart, plot_mass_radius_diagram


def render_explorer_view(filtered_df: pd.DataFrame, full_df: pd.DataFrame):
    """Render Tab 1: Habitable Zone and Planetary Explorer."""
    st.markdown("### 🌌 Habitable Zone & Composition Explorer")
    st.write(
        "Screen astronomical candidates using peer-reviewed Kopparapu effective flux models "
        "and compare planetary bulk densities against theoretical composition tracks."
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### Kopparapu Habitable Zone Boundary Diagram")
        hz_fig = plot_habitable_zone_chart(filtered_df)
        st.plotly_chart(hz_fig, use_container_width=True)

    with col2:
        st.markdown("#### Planetary Regime Distribution")
        regime_counts = filtered_df["planet_regime"].value_counts().reset_index()
        regime_counts.columns = ["Regime", "Count"]

        donut_fig = px.pie(
            regime_counts,
            names="Regime",
            values="Count",
            hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_dark"
        )
        donut_fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, b=20, t=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(donut_fig, use_container_width=True)

        st.markdown("#### 🔬 Astro-Physics Filter Metrics")
        st.markdown(
            f"""
            - **Median Candidate Radius:** `{filtered_df['pl_rade'].median():.2f}` $R_\\oplus$
            - **Median Insolation Flux:** `{filtered_df['pl_insol'].median():.2f}` $S_\\oplus$
            - **Average Host Star Temp:** `{filtered_df['st_teff'].mean():.0f}` K
            - **Mean Surface Gravity:** `{filtered_df['pl_gravity_rel'].median():.2f}` $g_\\oplus$
            """
        )

    st.markdown("---")
    st.markdown("#### Theoretical Mass-Radius Composition Tracks")
    mr_fig = plot_mass_radius_diagram(filtered_df)
    st.plotly_chart(mr_fig, use_container_width=True)

    st.markdown("#### 📋 Screened Candidate Catalog")
    display_cols = [
        "pl_name", "planet_regime", "hz_status", "esi_global",
        "pl_rade", "pl_bmasse", "pl_insol", "st_teff", "sy_dist", "discoverymethod"
    ]
    st.dataframe(
        filtered_df[display_cols].sort_values("esi_global", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=320
    )

    # CSV Exporter
    csv_bytes = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Screened Candidates (CSV)",
        data=csv_bytes,
        file_name="exoterra_screened_candidates.csv",
        mime="text/csv"
    )
