"""
Data Cleaning and Feature Enrichment Pipeline
=============================================
Orchestrates raw observational data filtering, physical imputation,
and habitability feature enrichment (ESI, HZ Status, Gravities, Log scales).
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.imputation import run_physical_imputation
from src.physics import (
    determine_hz_status,
    calculate_esi,
    calculate_escape_velocity,
    calculate_surface_gravity,
    classify_planet_regime
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_DIR / "exoplanets_processed.parquet"


def clean_and_enrich_data(raw_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """
    Clean raw NASA data and derive physical/astrobiological features:
    1. Filter unphysical rows (missing vital radius or stellar temp)
    2. Run physics-based imputation (mass, density, flux, Teq)
    3. Compute Habitable Zone classification (Kopparapu et al.)
    4. Compute Earth Similarity Index (Schulze-Makuch et al.)
    5. Compute surface gravity, escape velocity, and log-features
    6. Persist processed Parquet and CSV
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not force_refresh and PROCESSED_DATA_PATH.exists():
        logger.info(f"Loading cached processed data from {PROCESSED_DATA_PATH}")
        return pd.read_parquet(PROCESSED_DATA_PATH)

    logger.info("Cleaning raw observations and deriving astrophysical features...")
    df = raw_df.copy()

    # Drop entries missing vital planetary radius or host star effective temperature
    df = df.dropna(subset=["pl_rade", "st_teff"]).copy()

    # Filter out unphysical zero or negative values
    df = df[
        (df["pl_rade"] > 0) &
        (df["st_teff"] > 0)
    ].copy()

    # Run physics-based imputation for mass, density, and temperature
    df = run_physical_imputation(df)

    # Filter out non-positive or still-missing insolation flux
    df = df[df["pl_insol"] > 0].copy()

    # Calculate escape velocity and surface gravity
    df["pl_vesc_rel"] = calculate_escape_velocity(df["pl_bmasse"].values, df["pl_rade"].values)
    df["pl_gravity_rel"] = calculate_surface_gravity(df["pl_bmasse"].values, df["pl_rade"].values)

    # Calculate Habitable Zone status (Kopparapu et al.)
    df["hz_status"] = determine_hz_status(df["pl_insol"].values, df["st_teff"].values)

    # Calculate Earth Similarity Index (Global, Interior, Surface)
    esi_global, esi_int, esi_surf = calculate_esi(
        radius=df["pl_rade"].values,
        density=df["pl_dens"].values,
        vesc=df["pl_vesc_rel"].values,
        temperature=df["pl_tsurf_est"].values
    )
    df["esi_global"] = esi_global
    df["esi_interior"] = esi_int
    df["esi_surface"] = esi_surf

    # Classify planetary regimes
    df["planet_regime"] = classify_planet_regime(df["pl_rade"].values)

    # Logarithmic features for scale-invariant analysis
    df["log_rade"] = np.log10(df["pl_rade"])
    df["log_insol"] = np.log10(df["pl_insol"])
    df["log_bmasse"] = np.log10(df["pl_bmasse"])
    df["log_dens"] = np.log10(df["pl_dens"])

    logger.info(f"Data enrichment complete. Processed dataset size: {len(df)} exoplanets.")

    # Save to parquet and csv
    df.to_parquet(PROCESSED_DATA_PATH, index=False)
    df.to_csv(DATA_DIR / "exoplanets_processed.csv", index=False)

    return df
