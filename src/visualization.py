"""
Astrophysics and Machine Learning Visualization Engine
======================================================
Produces publication-grade Plotly interactive visualizations:
1. Kopparapu Habitable Zone (Teff vs Insolation) with shaded thermal regimes
2. Mass-Radius diagram with theoretical physical composition curves
3. Multi-dimensional Radar/Spider chart for Earth Similarity breakdown
4. Interactive 3D PCA Latent Space visualization
5. K-Means Elbow & Silhouette optimization curve
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
from src.physics import (
    calculate_hz_flux_boundary,
    EARTH_DENSITY_G_CM3,
    EARTH_TSURF_K
)


def plot_habitable_zone_chart(df: pd.DataFrame, highlight_planet: Optional[str] = None) -> go.Figure:
    """
    Plotly interactive Habitable Zone diagram: Host Star Teff vs. Insolation Flux
    with Kopparapu et al. boundary curves and shaded habitable bands.
    """
    fig = go.Figure()

    # Temperature grid for boundary curves (2600 K to 7200 K)
    t_grid = np.linspace(2600, 7000, 150)
    s_recent_venus = calculate_hz_flux_boundary(t_grid, "recent_venus")
    s_runaway_gh = calculate_hz_flux_boundary(t_grid, "runaway_greenhouse")
    s_max_gh = calculate_hz_flux_boundary(t_grid, "maximum_greenhouse")
    s_early_mars = calculate_hz_flux_boundary(t_grid, "early_mars")

    # 1. Conservative Habitable Zone Band (Green)
    fig.add_trace(go.Scatter(
        x=np.concatenate([s_runaway_gh, s_max_gh[::-1]]),
        y=np.concatenate([t_grid, t_grid[::-1]]),
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.22)',
        line=dict(color='rgba(46, 204, 113, 0.6)', width=1.5),
        name='Conservative HZ',
        hoverinfo='skip'
    ))

    # 2. Optimistic HZ (Inner - Recent Venus to Runaway GH)
    fig.add_trace(go.Scatter(
        x=np.concatenate([s_recent_venus, s_runaway_gh[::-1]]),
        y=np.concatenate([t_grid, t_grid[::-1]]),
        fill='toself',
        fillcolor='rgba(241, 196, 15, 0.12)',
        line=dict(color='rgba(241, 196, 15, 0.5)', width=1, dash='dash'),
        name='Optimistic HZ (Inner)',
        hoverinfo='skip'
    ))

    # 3. Optimistic HZ (Outer - Max GH to Early Mars)
    fig.add_trace(go.Scatter(
        x=np.concatenate([s_max_gh, s_early_mars[::-1]]),
        y=np.concatenate([t_grid, t_grid[::-1]]),
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.12)',
        line=dict(color='rgba(52, 152, 219, 0.5)', width=1, dash='dash'),
        name='Optimistic HZ (Outer)',
        hoverinfo='skip'
    ))

    # Color mapping for exoplanet points
    color_map = {
        "Conservative HZ": "#2ecc71",
        "Optimistic HZ (Inner)": "#f39c12",
        "Optimistic HZ (Outer)": "#3498db",
        "Too Hot": "#e74c3c",
        "Too Cold": "#95a5a6"
    }

    # Plot exoplanets grouped by HZ status
    for status, color in color_map.items():
        subset = df[df["hz_status"] == status]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset["pl_insol"],
                y=subset["st_teff"],
                mode='markers',
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.75,
                    line=dict(width=0.5, color='white')
                ),
                name=status,
                text=subset["pl_name"],
                customdata=np.stack((
                    subset["pl_rade"],
                    subset["esi_global"],
                    subset["planet_regime"],
                    subset["sy_dist"].fillna(-1)
                ), axis=-1),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Insolation: %{x:.2f} S_earth<br>"
                    "Host Star Teff: %{y:.0f} K<br>"
                    "Radius: %{customdata[0]:.2f} R_earth<br>"
                    "ESI: %{customdata[1]:.3f}<br>"
                    "Regime: %{customdata[2]}<br>"
                    "Distance: %{customdata[3]:.1f} pc"
                    "<extra></extra>"
                )
            ))

    # Reference Earth Marker
    fig.add_trace(go.Scatter(
        x=[1.0],
        y=[5780.0],
        mode='markers+text',
        marker=dict(symbol='star', size=16, color='#f1c40f', line=dict(color='black', width=1.5)),
        name='Earth Reference',
        text=['Earth'],
        textposition='top right',
        hoverinfo='text'
    ))

    # Highlight specific planet if selected
    if highlight_planet:
        matched = df[df["pl_name"].str.lower() == highlight_planet.lower()]
        if not matched.empty:
            p = matched.iloc[0]
            fig.add_trace(go.Scatter(
                x=[p["pl_insol"]],
                y=[p["st_teff"]],
                mode='markers+text',
                marker=dict(symbol='circle-open', size=20, color='#ff00ff', line=dict(width=3)),
                name=f"Selected: {p['pl_name']}",
                text=[p["pl_name"]],
                textposition='bottom center'
            ))

    fig.update_layout(
        title="<b>Kopparapu Habitable Zone: Stellar Flux vs. Host Star Temperature</b>",
        xaxis=dict(
            title="Insolation Flux (Earth Solar Constant = 1.0)",
            type="log",
            autorange="reversed", # Astronomy standard: higher flux on left
            gridcolor="rgba(255, 255, 255, 0.1)"
        ),
        yaxis=dict(
            title="Host Star Effective Temperature (Kelvin)",
            gridcolor="rgba(255, 255, 255, 0.1)"
        ),
        template="plotly_dark",
        height=580,
        margin=dict(l=50, r=50, b=50, t=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_mass_radius_diagram(df: pd.DataFrame) -> go.Figure:
    """
    Mass vs. Radius log-log diagram illustrating theoretical compositional tracks.
    """
    fig = go.Figure()

    # Theoretical composition curves (Zapolsky & Salpeter 1969 / Seager et al.)
    mass_grid = np.logspace(-1, 3.5, 100)
    # Earth mantle (rocky) track: R ~ M^(1/3.7)
    r_rock = np.power(mass_grid, 1.0 / 3.7)
    # 100% Water / Ice track: R ~ 1.25 * M^(1/3.5)
    r_water = 1.25 * np.power(mass_grid, 1.0 / 3.5)
    # 100% Iron track: R ~ 0.75 * M^(1/3.7)
    r_iron = 0.75 * np.power(mass_grid, 1.0 / 3.7)

    fig.add_trace(go.Scatter(x=mass_grid, y=r_iron, mode='lines', line=dict(color='#e74c3c', dash='dot', width=1.5), name='100% Iron Core'))
    fig.add_trace(go.Scatter(x=mass_grid, y=r_rock, mode='lines', line=dict(color='#2ecc71', dash='dash', width=1.5), name='100% Silicate / Rock'))
    fig.add_trace(go.Scatter(x=mass_grid, y=r_water, mode='lines', line=dict(color='#3498db', dash='dashdot', width=1.5), name='100% Water / Ice World'))

    fig.add_trace(go.Scatter(
        x=df["pl_bmasse"],
        y=df["pl_rade"],
        mode='markers',
        marker=dict(
            size=6,
            color=df["esi_global"],
            colorscale='Viridis',
            colorbar=dict(title="Earth Similarity Index (ESI)"),
            opacity=0.7,
            showscale=True
        ),
        text=df["pl_name"],
        customdata=df["planet_regime"],
        hovertemplate="<b>%{text}</b><br>Mass: %{x:.2f} M_earth<br>Radius: %{y:.2f} R_earth<br>Regime: %{customdata}<extra></extra>",
        name='Exoplanets'
    ))

    # Reference Earth Marker
    fig.add_trace(go.Scatter(
        x=[1.0], y=[1.0], mode='markers+text',
        marker=dict(symbol='star', size=16, color='#f1c40f', line=dict(color='black', width=1.5)),
        name='Earth', text=['Earth'], textposition='top left'
    ))

    fig.update_layout(
        title="<b>Mass-Radius Diagram with Compositional Curves</b>",
        xaxis=dict(title="Planet Mass (Earth Mass = 1.0)", type="log", gridcolor="rgba(255, 255, 255, 0.1)"),
        yaxis=dict(title="Planet Radius (Earth Radius = 1.0)", type="log", gridcolor="rgba(255, 255, 255, 0.1)"),
        template="plotly_dark",
        height=550,
        margin=dict(l=50, r=50, b=50, t=60)
    )
    return fig


def plot_esi_radar_chart(planet_name: str, df: pd.DataFrame) -> go.Figure:
    """
    Generate an interactive 5-axis Radar / Spider chart comparing
    an exoplanet against Earth across all ESI physical components.
    """
    matched = df[df["pl_name"].str.strip().str.lower() == planet_name.strip().lower()]
    if matched.empty:
        # Default fallback to Earth
        p_row = {
            "pl_name": planet_name,
            "esi_r": 1.0, "esi_dens": 1.0, "esi_v": 1.0, "esi_t": 1.0, "esi_glob": 1.0
        }
    else:
        p = matched.iloc[0]
        # Calculate sub-component similarities
        r = p["pl_rade"]
        dens = p["pl_dens"]
        vesc = p["pl_vesc_rel"]
        temp = p["pl_tsurf_est"]

        esi_r = (1.0 - abs(r - 1.0) / (r + 1.0)) ** 0.57
        esi_dens = (1.0 - abs(dens - EARTH_DENSITY_G_CM3) / (dens + EARTH_DENSITY_G_CM3)) ** 1.07
        esi_v = (1.0 - abs(vesc - 1.0) / (vesc + 1.0)) ** 0.70
        esi_t = (1.0 - abs(temp - EARTH_TSURF_K) / (temp + EARTH_TSURF_K)) ** 5.58

        p_row = {
            "pl_name": p["pl_name"],
            "esi_r": float(np.clip(esi_r, 0, 1)),
            "esi_dens": float(np.clip(esi_dens, 0, 1)),
            "esi_v": float(np.clip(esi_v, 0, 1)),
            "esi_t": float(np.clip(esi_t, 0, 1)),
            "esi_glob": float(p["esi_global"])
        }

    categories = [
        "Radius Similarity",
        "Bulk Density Similarity",
        "Escape Velocity Similarity",
        "Surface Temp Similarity",
        "Global ESI"
    ]

    planet_values = [
        p_row["esi_r"], p_row["esi_dens"], p_row["esi_v"], p_row["esi_t"], p_row["esi_glob"]
    ]
    # Close polygon
    planet_values.append(planet_values[0])
    earth_values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cat_closed = categories + [categories[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=earth_values,
        theta=cat_closed,
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.15)',
        line=dict(color='#2ecc71', width=1.5, dash='dash'),
        name='Earth (Baseline = 1.0)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=planet_values,
        theta=cat_closed,
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.35)',
        line=dict(color='#3498db', width=2.5),
        name=p_row["pl_name"]
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05], gridcolor='rgba(255, 255, 255, 0.15)')
        ),
        template="plotly_dark",
        title=f"<b>Earth Similarity Breakdown: {p_row['pl_name']}</b>",
        height=450,
        margin=dict(l=40, r=40, b=40, t=60)
    )

    return fig


def plot_latent_space_3d(df: pd.DataFrame, X_pca: np.ndarray, color_col: str = "cluster_archetype") -> go.Figure:
    """
    Interactive 3D scatter plot of the exoplanetary manifold in PCA latent space.
    """
    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]
    df_plot["PC3"] = X_pca[:, 2]

    fig = px.scatter_3d(
        df_plot,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color_col,
        hover_name="pl_name",
        hover_data={"pl_rade": ":.2f", "pl_insol": ":.2f", "esi_global": ":.3f", "PC1": False, "PC2": False, "PC3": False},
        opacity=0.75,
        template="plotly_dark",
        title="<b>3D Exoplanet Latent Space Manifold (PCA)</b>"
    )

    fig.update_traces(marker=dict(size=3.5))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="PC1: Mass & Size"),
            yaxis=dict(title="PC2: Irradiation & Star Temp"),
            zaxis=dict(title="PC3: Bulk Density")
        ),
        height=620,
        margin=dict(l=20, r=20, b=20, t=50)
    )

    return fig
