"""
preprocess.py

Load a saved lightcurve (FITS or CSV), clean NaNs, remove outliers
and optionally normalize. Saves cleaned lightcurve to data/.

Usage:
    python src/preprocess.py --input data/lightcurve_raw_TIC_25155310.csv
"""

import argparse
from pathlib import Path
import logging

import numpy as np
import lightkurve as lk
from astropy.timeseries import TimeSeries

from src.utils import DATA_DIR, configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def load_lightcurve(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    logger.info("Loading %s", path)
    if path.suffix.lower() in (".fits", ".fz"):
        lc = lk.LightCurve.read(path)
    elif path.suffix.lower() in (".csv", ".txt"):
        import pandas as pd
        df = pd.read_csv(path)
        # expect columns like time, flux or time, flux_normalized
        # Flexible mapping:
        time_col = None
        flux_col = None
        for c in df.columns:
            c_lower = c.lower()
            if "time" in c_lower and time_col is None:
                time_col = c
            if ("flux" in c_lower or "pdcsap_flux" in c_lower or "sap_flux" in c_lower) and flux_col is None:
                flux_col = c
        if time_col is None or flux_col is None:
            raise ValueError("CSV must contain time and flux columns")
        lc = lk.LightCurve(time=df[time_col].values, flux=df[flux_col].values)
    else:
        raise ValueError("Unsupported file extension: " + path.suffix)
    return lc

def clean_lightcurve(lc: lk.LightCurve, sigma_clip: float = 5.0, normalize: bool = True):
    """Remove NaNs, outliers using sigma clipping. Optionally normalize flux to 1.0."""
    logger.info("Initial points: %d", len(lc.time.value))
    lc = lc.remove_nans()
    logger.info("After remove_nans: %d", len(lc.time.value))
    try:
        lc = lc.remove_outliers(sigma=sigma_clip)
    except Exception:
        # manual sigma clipping fallback
        logger.warning("remove_outliers failed; applying manual sigma clipping")
        flux = lc.flux
        med = np.nanmedian(flux)
        std = np.nanstd(flux)
        mask = np.abs(flux - med) < sigma_clip * std
        lc = lc[mask]
    logger.info("After outlier removal: %d", len(lc.time.value))
    if normalize:
        lc = lc.normalize()
        logger.info("Normalized lightcurve")
    return lc

def save_cleaned(lc: lk.LightCurve, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lc.to_fits(out_path.with_suffix(".fits"), overwrite=True)
    lc.to_pandas().to_csv(out_path.with_suffix(".csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input lightcurve FITS/CSV path")
    parser.add_argument("--out", default=str(DATA_DIR / "lightcurve_clean"), help="Output prefix")
    parser.add_argument("--sigma", type=float, default=5.0, help="Sigma for outlier removal")
    args = parser.parse_args()

    lc = load_lightcurve(Path(args.input))
    lc_clean = clean_lightcurve(lc, sigma_clip=args.sigma, normalize=True)
    save_cleaned(lc_clean, Path(args.out))

if __name__ == "__main__":
    main()

