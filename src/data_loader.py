"""
NASA Exoplanet Archive TAP API Ingestion Engine
================================================
Handles automated querying of the NASA TAP API, network resilience,
query construction, and Parquet caching.
"""

import logging
import urllib.parse
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "exoplanets_raw.parquet"

# NASA Exoplanet Archive TAP API endpoint
NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Key astronomical attributes to retrieve from the composite parameters table
NASA_COLUMNS = [
    "pl_name",
    "hostname",
    "discoverymethod",
    "disc_year",
    "disc_facility",
    "sy_dist",
    "sy_pnum",
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen",
    "pl_rade",
    "pl_bmasse",
    "pl_dens",
    "pl_insol",
    "pl_eqt",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_lum",
    "st_spectype",
    "ra",
    "dec"
]


def build_tap_query() -> str:
    """Build the SQL-compatible TAP query string."""
    cols_str = ",".join(NASA_COLUMNS)
    return f"select {cols_str} from pscomppars"


def fetch_raw_nasa_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Query the NASA Exoplanet Archive TAP API for the composite parameters table.
    Caches the result locally in Parquet format.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not force_refresh and RAW_DATA_PATH.exists():
        logger.info(f"Loading cached raw NASA data from {RAW_DATA_PATH}")
        return pd.read_parquet(RAW_DATA_PATH)

    logger.info("Connecting to NASA Exoplanet Archive TAP API...")
    query = build_tap_query()
    url = f"{NASA_TAP_URL}?query={urllib.parse.quote(query)}&format=csv"

    try:
        df = pd.read_csv(url)
        logger.info(f"Successfully retrieved {len(df)} exoplanet records from NASA.")
        df.to_parquet(RAW_DATA_PATH, index=False)
        df.to_csv(DATA_DIR / "exoplanets_raw.csv", index=False)
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch from NASA TAP API: {e}")
        if RAW_DATA_PATH.exists():
            logger.info("Falling back to existing cached Parquet file.")
            return pd.read_parquet(RAW_DATA_PATH)
        raise RuntimeError("Could not connect to NASA API and no local cache was found.") from e


def load_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """
    Convenience orchestrator: Fetches raw data from NASA and returns
    the cleaned and feature-enriched dataset.
    """
    from src.data_cleaner import clean_and_enrich_data
    raw = fetch_raw_nasa_data(force_refresh=force_refresh)
    return clean_and_enrich_data(raw, force_refresh=force_refresh)


if __name__ == "__main__":
    df = load_dataset()
    print(f"Loaded {len(df)} planets successfully.")
    print("Top 5 candidates by Earth Similarity Index (ESI):")
    print(df.sort_values("esi_global", ascending=False)[["pl_name", "pl_rade", "pl_insol", "hz_status", "esi_global"]].head())
