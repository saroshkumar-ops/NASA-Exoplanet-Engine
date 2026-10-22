"""
Astrobiology and Planetary Physics Engine
==========================================
Implements scientific formulations for exoplanetary habitability and characterization:
1. Kopparapu et al. (2013, 2014) Habitable Zone Boundaries
2. Schulze-Makuch et al. (2011) Earth Similarity Index (ESI)
3. Astrophysical classification regimes (Terran, Super-Earth, Sub-Neptune, Jovian)
4. Surface gravity and escape velocity estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple

# Physical constants and reference values (in Earth units or standard SI)
EARTH_RADIUS_KM = 6371.0
EARTH_MASS_KG = 5.972e24
EARTH_DENSITY_G_CM3 = 5.514
EARTH_TEQ_K = 255.0         # Blackbody equilibrium temperature with albedo ~0.3
EARTH_TSURF_K = 288.0       # Mean surface temperature with greenhouse effect
EARTH_VESC_KM_S = 11.186    # Escape velocity in km/s
EARTH_INSOL = 1.0           # Solar constant at 1 AU

# Kopparapu et al. (2013, 2014) coefficients for Habitable Zone (HZ) boundaries
# S_eff = S_eff_sun + a*T* + b*T*^2 + c*T*^3 + d*T*^4, where T* = T_eff - 5780 K
HZ_COEFFICIENTS = {
    "recent_venus": {
        "S_sun": 1.776,
        "a": 2.136e-4,
        "b": 2.533e-8,
        "c": -1.332e-11,
        "d": -3.097e-15,
    },
    "runaway_greenhouse": {
        "S_sun": 1.107,
        "a": 1.332e-4,
        "b": 1.580e-8,
        "c": -8.308e-12,
        "d": -1.931e-15,
    },
    "maximum_greenhouse": {
        "S_sun": 0.356,
        "a": 6.171e-5,
        "b": 1.698e-9,
        "c": -3.198e-12,
        "d": -5.575e-16,
    },
    "early_mars": {
        "S_sun": 0.320,
        "a": 5.547e-5,
        "b": 1.526e-9,
        "c": -2.874e-12,
        "d": -5.011e-16,
    },
}

# Schulze-Makuch et al. (2011) Earth Similarity Index (ESI) weights
# ESI_param = (1 - |(x - x0) / (x + x0)|)^w
ESI_WEIGHTS = {
    "radius": {"x0": 1.0, "w": 0.57},              # Earth radii
    "density": {"x0": EARTH_DENSITY_G_CM3, "w": 1.07}, # g/cm^3
    "vesc": {"x0": 1.0, "w": 0.70},                # Earth relative escape velocity
    "temperature": {"x0": EARTH_TSURF_K, "w": 5.58} # Kelvin (surface / estimated temp)
}


def calculate_hz_flux_boundary(t_eff: Union[float, np.ndarray], boundary_type: str) -> Union[float, np.ndarray]:
    """
    Calculate the effective stellar flux (S_eff, in units of Earth solar constant)
    for a given Habitable Zone boundary using Kopparapu et al. (2013, 2014).
    Valid for 2600 K <= T_eff <= 7200 K.
    """
    params = HZ_COEFFICIENTS[boundary_type]
    t_star = np.clip(t_eff, 2600.0, 7200.0) - 5780.0
    
    s_eff = (
        params["S_sun"]
        + params["a"] * t_star
        + params["b"] * (t_star ** 2)
        + params["c"] * (t_star ** 3)
        + params["d"] * (t_star ** 4)
    )
    return np.maximum(s_eff, 0.05)


def determine_hz_status(insol: Union[float, np.ndarray], t_eff: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
    """
    Classify an exoplanet into a scientifically validated Habitable Zone regime:
    - 'Too Hot' (Interior to Recent Venus)
    - 'Optimistic HZ (Inner)' (Between Recent Venus & Runaway Greenhouse)
    - 'Conservative HZ' (Between Runaway Greenhouse & Maximum Greenhouse)
    - 'Optimistic HZ (Outer)' (Between Maximum Greenhouse & Early Mars)
    - 'Too Cold' (Beyond Early Mars)
    """
    s_recent_venus = calculate_hz_flux_boundary(t_eff, "recent_venus")
    s_runaway_gh = calculate_hz_flux_boundary(t_eff, "runaway_greenhouse")
    s_max_gh = calculate_hz_flux_boundary(t_eff, "maximum_greenhouse")
    s_early_mars = calculate_hz_flux_boundary(t_eff, "early_mars")

    if np.isscalar(insol) and np.isscalar(t_eff):
        if insol > s_recent_venus:
            return "Too Hot"
        elif insol >= s_runaway_gh:
            return "Optimistic HZ (Inner)"
        elif insol >= s_max_gh:
            return "Conservative HZ"
        elif insol >= s_early_mars:
            return "Optimistic HZ (Outer)"
        else:
            return "Too Cold"

    # Vectorized computation for NumPy / Pandas
    insol_arr = np.asarray(insol)
    result = np.full(insol_arr.shape, "Too Cold", dtype=object)
    result[insol_arr > s_recent_venus] = "Too Hot"
    result[(insol_arr <= s_recent_venus) & (insol_arr >= s_runaway_gh)] = "Optimistic HZ (Inner)"
    result[(insol_arr < s_runaway_gh) & (insol_arr >= s_max_gh)] = "Conservative HZ"
    result[(insol_arr < s_max_gh) & (insol_arr >= s_early_mars)] = "Optimistic HZ (Outer)"
    return result


def estimate_planetary_density(radius: Union[float, np.ndarray], mass: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate bulk density in g/cm^3 from radius (Earth radii) and mass (Earth masses).
    Density_earth = 5.514 g/cm^3.
    rho_rel = Mass / Radius^3
    rho = rho_rel * 5.514 g/cm^3
    """
    r_safe = np.maximum(radius, 0.05)
    m_safe = np.maximum(mass, 0.001)
    rho_rel = m_safe / (r_safe ** 3)
    return rho_rel * EARTH_DENSITY_G_CM3


def estimate_mass_from_radius(radius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Empirical Mass-Radius relation (Chen & Kipping 2017 / Wolfgang et al.):
    Used as physical imputation when mass measurement is unavailable.
    - Terran (R < 1.23): M = R^3.68
    - Super-Earth / Mini-Neptune (1.23 <= R < 14.2): M = 0.97 * R^1.70
    - Gas Giant (R >= 14.2): M ~ 317.8 (Jupiter-like plateau)
    """
    r = np.asarray(radius)
    m = np.zeros_like(r, dtype=float)
    
    mask_rocky = r < 1.23
    mask_intermediate = (r >= 1.23) & (r < 14.2)
    mask_giant = r >= 14.2
    
    m[mask_rocky] = np.power(r[mask_rocky], 3.68)
    m[mask_intermediate] = 0.97 * np.power(r[mask_intermediate], 1.70)
    m[mask_giant] = 317.8 * np.power(r[mask_giant] / 11.2, 0.88)
    
    return float(m) if np.isscalar(radius) else m


def calculate_escape_velocity(mass: Union[float, np.ndarray], radius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate escape velocity relative to Earth:
    v_esc / v_esc_earth = sqrt((M / M_earth) / (R / R_earth))
    """
    m_safe = np.maximum(mass, 0.001)
    r_safe = np.maximum(radius, 0.05)
    return np.sqrt(m_safe / r_safe)


def calculate_surface_gravity(mass: Union[float, np.ndarray], radius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate surface gravity in Earth gravities (g):
    g_rel = (M / M_earth) / (R / R_earth)^2
    """
    m_safe = np.maximum(mass, 0.001)
    r_safe = np.maximum(radius, 0.05)
    return m_safe / (r_safe ** 2)


def calculate_esi(
    radius: Union[float, np.ndarray],
    density: Union[float, np.ndarray],
    vesc: Union[float, np.ndarray],
    temperature: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Schulze-Makuch et al. (2011) Earth Similarity Index:
    Computes Interior ESI, Surface ESI, and Global ESI.
    
    Parameters:
    - radius: in Earth radii
    - density: in g/cm^3
    - vesc: relative to Earth (1.0 = 11.2 km/s)
    - temperature: in Kelvin (equilibrium or surface temp)
    
    Returns:
    - (esi_global, esi_interior, esi_surface)
    """
    def _param_esi(x, x0, w):
        x_val = np.maximum(np.asarray(x), 1e-4)
        diff = np.abs(x_val - x0) / (x_val + x0)
        # Avoid negative bases or NaN
        base = np.clip(1.0 - diff, 1e-6, 1.0)
        return np.power(base, w)

    # Interior parameters (n=2)
    esi_r = _param_esi(radius, ESI_WEIGHTS["radius"]["x0"], ESI_WEIGHTS["radius"]["w"])
    esi_rho = _param_esi(density, ESI_WEIGHTS["density"]["x0"], ESI_WEIGHTS["density"]["w"])
    esi_interior = np.power(esi_r * esi_rho, 1.0 / 2.0)

    # Surface parameters (n=2)
    esi_v = _param_esi(vesc, ESI_WEIGHTS["vesc"]["x0"], ESI_WEIGHTS["vesc"]["w"])
    esi_t = _param_esi(temperature, ESI_WEIGHTS["temperature"]["x0"], ESI_WEIGHTS["temperature"]["w"])
    esi_surface = np.power(esi_v * esi_t, 1.0 / 2.0)

    # Global ESI is the geometric mean of interior and surface ESI
    esi_global = np.sqrt(esi_interior * esi_surface)

    if np.isscalar(radius):
        return float(esi_global), float(esi_interior), float(esi_surface)
    return esi_global, esi_interior, esi_surface


def explain_esi_breakdown(
    radius: float,
    density: float,
    vesc: float,
    temperature: float
) -> Dict[str, float]:
    """
    Generate an explainable component-by-component breakdown of Earth Similarity Index (ESI).
    Useful for diagnostic dashboards and scientific candidate evaluation.
    """
    def _param_comp(x, x0, w):
        diff = abs(x - x0) / (x + x0)
        return float((1.0 - diff) ** w)

    comp_r = _param_comp(radius, ESI_WEIGHTS["radius"]["x0"], ESI_WEIGHTS["radius"]["w"])
    comp_rho = _param_comp(density, ESI_WEIGHTS["density"]["x0"], ESI_WEIGHTS["density"]["w"])
    comp_v = _param_comp(vesc, ESI_WEIGHTS["vesc"]["x0"], ESI_WEIGHTS["vesc"]["w"])
    comp_t = _param_comp(temperature, ESI_WEIGHTS["temperature"]["x0"], ESI_WEIGHTS["temperature"]["w"])
    
    esi_int = (comp_r * comp_rho) ** (1.0 / 2.0)
    esi_surf = (comp_v * comp_t) ** (1.0 / 2.0)
    esi_glob = (esi_int * esi_surf) ** 0.5

    return {
        "radius_similarity": round(comp_r, 4),
        "density_similarity": round(comp_rho, 4),
        "interior_esi": round(esi_int, 4),
        "escape_velocity_similarity": round(comp_v, 4),
        "temperature_similarity": round(comp_t, 4),
        "surface_esi": round(esi_surf, 4),
        "global_esi": round(esi_glob, 4),
    }


def classify_planet_regime(radius: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
    """
    Classify exoplanet based on established radius thresholds (Rogers 2015, Fulton et al. 2017):
    - Sub-Earth: R < 0.8 Earth radii
    - Terran (Earth-sized): 0.8 <= R < 1.25 Earth radii
    - Super-Earth: 1.25 <= R < 1.75 Earth radii (rocky transition limit ~1.6)
    - Sub-Neptune / Mini-Neptune: 1.75 <= R < 3.5 Earth radii
    - Neptune-like: 3.5 <= R < 6.0 Earth radii
    - Jovian / Gas Giant: R >= 6.0 Earth radii
    """
    r = np.asarray(radius)
    regimes = np.full(r.shape, "Jovian / Gas Giant", dtype=object)
    regimes[r < 0.8] = "Sub-Earth"
    regimes[(r >= 0.8) & (r < 1.25)] = "Terran (Rocky)"
    regimes[(r >= 1.25) & (r < 1.75)] = "Super-Earth"
    regimes[(r >= 1.75) & (r < 3.5)] = "Sub-Neptune"
    regimes[(r >= 3.5) & (r < 6.0)] = "Neptune-like"

    return str(regimes) if np.isscalar(radius) else regimes
