"""
Physical Imputation Engine for Exoplanetary Attributes
======================================================
Imputes missing observational parameters using astrophysically grounded
relationships rather than arbitrary statistical means:
1. Mass-Radius power laws (Chen & Kipping 2017)
2. Equilibrium temperature from stellar flux (Stefan-Boltzmann scaling)
3. Bulk density from mass and volumetric radius
"""

import numpy as np
import pandas as pd
from typing import Union
from src.physics import (
    estimate_mass_from_radius,
    estimate_planetary_density,
    EARTH_TEQ_K,
    EARTH_TSURF_K
)

def impute_missing_masses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing or non-positive planetary masses using empirical mass-radius relations.
    """
    df = df.copy()
    missing_mask = df["pl_bmasse"].isna() | (df["pl_bmasse"] <= 0)
    if missing_mask.any():
        df.loc[missing_mask, "pl_bmasse"] = estimate_mass_from_radius(
            df.loc[missing_mask, "pl_rade"].values
        )
    return df


def impute_missing_densities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing or non-positive bulk densities from planetary mass and radius.
    """
    df = df.copy()
    missing_mask = df["pl_dens"].isna() | (df["pl_dens"] <= 0)
    if missing_mask.any():
        df.loc[missing_mask, "pl_dens"] = estimate_planetary_density(
            df.loc[missing_mask, "pl_rade"].values,
            df.loc[missing_mask, "pl_bmasse"].values
        )
    return df


def impute_temperatures_and_flux(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing insolation flux and equilibrium temperatures using stellar parameters.
    """
    df = df.copy()
    
    # Estimate missing insolation flux from host star radius, temperature, and semi-major axis
    missing_insol = df["pl_insol"].isna() | (df["pl_insol"] <= 0)
    can_calc_insol = (
        missing_insol &
        df["st_rad"].notna() &
        df["st_teff"].notna() &
        df["pl_orbsmax"].notna() &
        (df["pl_orbsmax"] > 0)
    )
    if can_calc_insol.any():
        df.loc[can_calc_insol, "pl_insol"] = (
            (df.loc[can_calc_insol, "st_rad"] ** 2) *
            ((df.loc[can_calc_insol, "st_teff"] / 5780.0) ** 4) /
            (df.loc[can_calc_insol, "pl_orbsmax"] ** 2)
        )

    # Impute equilibrium temperature from insolation: T_eq ~ 255 * (S)^(1/4)
    missing_teq = df["pl_eqt"].isna() | (df["pl_eqt"] <= 0)
    valid_insol = df["pl_insol"] > 0
    teq_impute_mask = missing_teq & valid_insol
    if teq_impute_mask.any():
        df.loc[teq_impute_mask, "pl_eqt"] = EARTH_TEQ_K * np.power(
            df.loc[teq_impute_mask, "pl_insol"].values, 0.25
        )

    # Estimated surface temperature with greenhouse effect
    df["pl_tsurf_est"] = df["pl_eqt"] * (EARTH_TSURF_K / EARTH_TEQ_K)
    return df


def run_physical_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute full physical imputation pipeline sequentially:
    Mass -> Density -> Flux & Temperature.
    """
    df = impute_missing_masses(df)
    df = impute_temperatures_and_flux(df)
    df = impute_missing_densities(df)
    return df
