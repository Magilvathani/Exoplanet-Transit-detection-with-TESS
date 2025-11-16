"""
search.py

Search for transits using the Box Least Squares (BLS) algorithm from astropy.
Produces a BLS periodogram CSV and returns best candidate period.

Usage:
    python src/search.py --input data/lightcurve_detrended.detrended.fits --min-period 0.5 --max-period 10 --n-periods 20000
"""

import argparse
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

import lightkurve as lk

from src.utils import DATA_DIR, configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def run_bls(lc: lk.LightCurve, min_period: float = 0.5, max_period: float = 10.0, n_periods: int = 20000, duration_fraction: float = 0.05):
    """
    Run BLS over a period grid between min_period and max_period.
    duration_fraction is the guess for typical transit duration as fraction of period (used to set durations grid).
    """
    t = lc.time.value
    y = lc.flux.value
    # Remove NaNs
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) < 10:
        raise ValueError("Not enough valid points for BLS")

    logger.info("BLS: periods grid from %.3f to %.3f (%d points)", min_period, max_period, n_periods)
    periods = np.linspace(min_period, max_period, n_periods)
    durations = period_to_duration_guess(periods, duration_fraction)
    bls = BoxLeastSquares(t, y)
    result = bls.power(periods, durations)
    return periods, durations, result

def period_to_duration_guess(periods, fraction=0.05):
    # duration scale: fraction of period, clipped to a min and max (days)
    durations = fraction * periods
    durations = np.clip(durations, 0.005, 0.5)  # clamp durations to reasonable values
    return durations

def save_bls_results(periods, power, out_csv: Path):
    df = pd.DataFrame({"period": periods, "power": power})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

def pick_best(periods, power):
    idx = np.nanargmax(power)
    best_period = periods[idx]
    best_power = power[idx]
    return best_period, best_power

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input detrended lightcurve (FITS/CSV)")
    parser.add_argument("--min-period", type=float, default=0.5)
    parser.add_argument("--max-period", type=float, default=10.0)
    parser.add_argument("--n-periods", type=int, default=20000)
    parser.add_argument("--out", default=str(DATA_DIR / "bls_results.csv"))
    args = parser.parse_args()

    # read with lightkurve convenience
    lc = lk.LightCurve.read(args.input)
    periods, durations, result = run_bls(lc, args.min_period, args.max_period, args.n_periods)
    save_bls_results(periods, result.power, Path(args.out))
    best_period, best_power = pick_best(periods, result.power)
    logger.info("Best period: %.6f days (power=%.4f)", best_period, best_power)
    # Save summary
    summary = {"best_period": float(best_period), "best_power": float(best_power)}
    from src.utils import write_json
    write_json(Path(args.out).with_suffix(".summary.json"), summary)

if __name__ == "__main__":
    main()

