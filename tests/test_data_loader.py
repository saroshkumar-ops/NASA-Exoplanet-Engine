"""
Unit & Integration Tests for Data Ingestion Pipeline
===================================================
Tests data loading, TAP schema validation, physical imputation,
and feature enrichment constraints.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_cleaner import clean_and_enrich_data
from src.data_loader import NASA_COLUMNS, build_tap_query
from src.imputation import run_physical_imputation

@pytest.fixture
def sample_raw_data():
    """Create a synthetic mini DataFrame resembling raw NASA TAP query results."""
    return pd.DataFrame({
        "pl_name": ["Earth-Analog-1", "Hot-Jupiter-1", "Super-Earth-Missing-Mass", "Missing-Teq-Planet"],
        "hostname": ["Star-A", "Star-B", "Star-C", "Star-D"],
        "discoverymethod": ["Transit", "Radial Velocity", "Transit", "Transit"],
        "disc_year": [2020, 2015, 2022, 2023],
        "disc_facility": ["Kepler", "Keck", "TESS", "TESS"],
        "sy_dist": [10.5, 45.0, 120.0, 85.0],
        "sy_pnum": [3, 1, 2, 4],
        "pl_orbper": [365.25, 3.5, 25.0, 15.0],
        "pl_orbsmax": [1.0, 0.04, 0.15, 0.1],
        "pl_orbeccen": [0.016, 0.0, 0.05, 0.02],
        "pl_rade": [1.0, 11.2, 1.5, 0.9],
        "pl_bmasse": [1.0, 317.8, np.nan, 0.8], # Third planet has missing mass
        "pl_dens": [5.514, 1.33, np.nan, np.nan], # Missing densities
        "pl_insol": [1.0, 850.0, 1.2, np.nan],   # Fourth planet has missing insol (to be computed)
        "pl_eqt": [255.0, 1400.0, 270.0, np.nan],# Missing Teq
        "st_teff": [5780.0, 6000.0, 5200.0, 4800.0],
        "st_rad": [1.0, 1.2, 0.85, 0.75],
        "st_mass": [1.0, 1.1, 0.88, 0.72],
        "st_lum": [0.0, 0.35, -0.3, -0.6],
        "st_spectype": ["G2V", "F8V", "K2V", "M0V"],
        "ra": [180.0, 45.0, 270.0, 90.0],
        "dec": [15.0, -20.0, 30.0, -45.0]
    })


def test_tap_query_construction():
    """Ensure TAP query contains all essential columns."""
    query = build_tap_query()
    assert "select" in query
    assert "pscomppars" in query
    for col in ["pl_name", "pl_rade", "pl_insol", "st_teff"]:
        assert col in query


def test_physical_imputation_pipeline(sample_raw_data):
    """Test that physical imputation fills missing mass, density, and temperatures."""
    imputed = run_physical_imputation(sample_raw_data)
    assert imputed["pl_bmasse"].isna().sum() == 0
    assert imputed["pl_dens"].isna().sum() == 0
    assert imputed["pl_eqt"].isna().sum() == 0
    assert (imputed["pl_bmasse"] > 0).all()
    assert (imputed["pl_dens"] > 0).all()


def test_clean_and_enrich_data_pipeline(sample_raw_data):
    """Verify data cleaning, physical imputation, and feature creation."""
    processed = clean_and_enrich_data(sample_raw_data, force_refresh=True)

    # 1. Output must not be empty
    assert len(processed) > 0

    # 2. Key enriched columns must exist
    required_cols = [
        "hz_status", "esi_global", "esi_interior", "esi_surface",
        "planet_regime", "pl_tsurf_est", "pl_vesc_rel", "pl_gravity_rel",
        "log_rade", "log_insol", "log_bmasse", "log_dens"
    ]
    for col in required_cols:
        assert col in processed.columns, f"Missing required feature column: {col}"

    # 3. Imputation check: No NaN in essential physical fields
    assert processed["pl_bmasse"].isna().sum() == 0
    assert processed["pl_dens"].isna().sum() == 0
    assert processed["pl_eqt"].isna().sum() == 0

    # 4. Physical boundaries check
    assert (processed["pl_rade"] > 0).all()
    assert (processed["pl_insol"] > 0).all()
    assert (processed["pl_bmasse"] > 0).all()
    assert (processed["pl_dens"] > 0).all()

    # 5. ESI boundaries check: ESI must always be between 0.0 and 1.0
    assert (processed["esi_global"] >= 0.0).all()
    assert (processed["esi_global"] <= 1.0).all()

    # 6. Earth analog check: The first planet in fixture is calibrated as Earth
    earth_analog = processed.loc[processed["pl_name"] == "Earth-Analog-1"].iloc[0]
    assert np.isclose(earth_analog["esi_global"], 1.0, atol=0.01)
    assert earth_analog["hz_status"] == "Conservative HZ"
    assert earth_analog["planet_regime"] == "Terran (Rocky)"
