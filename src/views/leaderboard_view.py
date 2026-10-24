"""
Earth Analogs & Habitability Leaderboard View
=============================================
Ranks candidates by Earth Similarity Index (ESI), provides
side-by-side radar analysis, and diagnostic deep-dive inspection.
"""

import streamlit as st
import pandas as pd
from src.visualization import plot_esi_radar_chart
from src.physics import explain_esi_breakdown


def render_leaderboard_view(df: pd.DataFrame):
    """Render Tab 2: Top Earth Analogs & Deep Dive Inspector."""
    st.markdown("### 🪐 Earth Analogs Leaderboard & Candidate Inspector")
    st.write(
        "Ranked candidates by Schulze-Makuch et al. (2011) Earth Similarity Index (ESI). "
        "Select any exoplanet to inspect multi-parameter radar similarity and astrophysical metrics."
    )

    # Top candidates
    top_candidates = df.sort_values("esi_global", ascending=False).head(20).reset_index(drop=True)

    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("#### 🏆 Top 20 Potential Earth Twins (by Global ESI)")
        display_df = top_candidates[[
            "pl_name", "esi_global", "hz_status", "planet_regime",
            "pl_rade", "pl_bmasse", "pl_insol", "sy_dist"
        ]].copy()
        display_df.columns = [
            "Planet", "Global ESI", "Habitable Zone", "Regime",
            "Radius (R⊕)", "Mass (M⊕)", "Insolation (S⊕)", "Distance (pc)"
        ]
        st.dataframe(display_df, use_container_width=True, height=450)

    with c_right:
        st.markdown("#### 🔍 Candidate Deep-Dive Inspector")
        planet_list = top_candidates["pl_name"].tolist()
        # Add popular landmark planets if not in top 20
        landmarks = ["TRAPPIST-1 e", "Kepler-452 b", "Proxima Centauri b", "TOI-700 d", "K2-72 e"]
        for p in landmarks:
            if p not in planet_list and not df[df["pl_name"].str.lower() == p.lower()].empty:
                planet_list.append(p)

        selected_planet = st.selectbox("Select Exoplanet to Inspect", planet_list, index=0)

        radar_fig = plot_esi_radar_chart(selected_planet, df)
        st.plotly_chart(radar_fig, use_container_width=True)

    # Detailed metrics card for selected planet
    matched = df[df["pl_name"].str.strip().str.lower() == selected_planet.strip().lower()]
    if not matched.empty:
        p = matched.iloc[0]
        st.markdown(f"#### 🛰️ Detailed Telemetry: **{p['pl_name']}**")
        m1, m2, m3, m4, m5, m6 = st.columns(6)

        m1.metric("Radius", f"{p['pl_rade']:.2f} R⊕", f"{(p['pl_rade']-1.0)*100:+.1f}% vs Earth")
        m2.metric("Mass", f"{p['pl_bmasse']:.2f} M⊕", f"{(p['pl_bmasse']-1.0)*100:+.1f}% vs Earth")
        m3.metric("Density", f"{p['pl_dens']:.2f} g/cm³", f"{(p['pl_dens']-5.51)*100/5.51:+.1f}% vs Earth")
        m4.metric("Gravity", f"{p['pl_gravity_rel']:.2f} g", f"{(p['pl_gravity_rel']-1.0)*100:+.1f}% vs Earth")
        m5.metric("Est. Surface Temp", f"{p['pl_tsurf_est']:.0f} K", f"{p['pl_tsurf_est']-288:+.0f} K vs Earth")
        m6.metric("Distance", f"{p['sy_dist']:.1f} pc" if pd.notna(p['sy_dist']) else "N/A", f"~{p['sy_dist']*3.26:.0f} ly" if pd.notna(p['sy_dist']) else "")

        # Explainability component table
        breakdown = explain_esi_breakdown(p["pl_rade"], p["pl_dens"], p["pl_vesc_rel"], p["pl_tsurf_est"])
        b_df = pd.DataFrame([
            {"Component": "Radius Similarity (Weight: 0.57)", "Similarity Score": breakdown["radius_similarity"]},
            {"Component": "Bulk Density Similarity (Weight: 1.07)", "Similarity Score": breakdown["density_similarity"]},
            {"Component": "Interior ESI (Geometric Mean)", "Similarity Score": breakdown["interior_esi"]},
            {"Component": "Escape Velocity Similarity (Weight: 0.70)", "Similarity Score": breakdown["escape_velocity_similarity"]},
            {"Component": "Surface Temperature Similarity (Weight: 5.58)", "Similarity Score": breakdown["temperature_similarity"]},
            {"Component": "Surface ESI (Geometric Mean)", "Similarity Score": breakdown["surface_esi"]},
            {"Component": "GLOBAL EARTH SIMILARITY INDEX", "Similarity Score": breakdown["global_esi"]},
        ])
        st.table(b_df)
