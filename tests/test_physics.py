import pytest
import numpy as np
from src.physics import (
    calculate_hz_flux_boundary,
    determine_hz_status,
    calculate_esi,
    estimate_planetary_density,
    calculate_escape_velocity,
    calculate_surface_gravity,
    classify_planet_regime,
    EARTH_DENSITY_G_CM3,
    EARTH_TSURF_K
)

def test_earth_esi_is_one():
    """Verify that Earth gets an exact ESI of 1.0."""
    esi_global, esi_int, esi_surf = calculate_esi(
        radius=1.0,
        density=EARTH_DENSITY_G_CM3,
        vesc=1.0,
        temperature=EARTH_TSURF_K
    )
    assert np.isclose(esi_global, 1.0, atol=1e-3)
    assert np.isclose(esi_int, 1.0, atol=1e-3)
    assert np.isclose(esi_surf, 1.0, atol=1e-3)

def test_earth_habitable_zone_status():
    """Earth at 1 AU around Sun (Teff = 5780 K, Insol = 1.0) must be in Conservative HZ."""
    status = determine_hz_status(insol=1.0, t_eff=5780.0)
    assert status == "Conservative HZ"

def test_kopparapu_solar_boundaries():
    """Check solar effective flux boundaries against Kopparapu et al. (2013) published values."""
    s_rv = calculate_hz_flux_boundary(5780.0, "recent_venus")
    s_rg = calculate_hz_flux_boundary(5780.0, "runaway_greenhouse")
    s_mg = calculate_hz_flux_boundary(5780.0, "maximum_greenhouse")
    s_em = calculate_hz_flux_boundary(5780.0, "early_mars")

    assert np.isclose(s_rv, 1.776, atol=1e-3)
    assert np.isclose(s_rg, 1.107, atol=1e-3)
    assert np.isclose(s_mg, 0.356, atol=1e-3)
    assert np.isclose(s_em, 0.320, atol=1e-3)

def test_mars_lower_esi():
    """Mars (radius ~0.53, mass ~0.107, T ~ 210K) should have ESI ~ 0.65 - 0.75."""
    r_mars = 0.532
    m_mars = 0.107
    dens_mars = estimate_planetary_density(r_mars, m_mars)
    vesc_mars = calculate_escape_velocity(m_mars, r_mars)
    esi_mars, _, _ = calculate_esi(r_mars, dens_mars, vesc_mars, 210.0)
    assert 0.60 <= esi_mars <= 0.80

def test_vectorized_computations():
    """Ensure functions handle NumPy arrays seamlessly."""
    radii = np.array([0.5, 1.0, 1.5, 2.5, 8.0])
    regimes = classify_planet_regime(radii)
    assert regimes[0] == "Sub-Earth"
    assert regimes[1] == "Terran (Rocky)"
    assert regimes[2] == "Super-Earth"
    assert regimes[3] == "Sub-Neptune"
    assert regimes[4] == "Jovian / Gas Giant"

def test_explain_esi_breakdown():
    """Verify explainability breakdown produces all components for Earth."""
    from src.physics import explain_esi_breakdown
    breakdown = explain_esi_breakdown(1.0, EARTH_DENSITY_G_CM3, 1.0, EARTH_TSURF_K)
    assert breakdown["global_esi"] == 1.0
    assert breakdown["interior_esi"] == 1.0
    assert breakdown["surface_esi"] == 1.0
    assert "temperature_similarity" in breakdown
