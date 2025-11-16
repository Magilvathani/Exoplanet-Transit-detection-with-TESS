"""
detrend.py

Detrend (flatten) a lightcurve using lightkurve's flatten (Savitzky-Golay by default)
or by using a specified method.

Saves detrended lightcurve and trend to files.

Usage:
    python src/detrend.py --input data/lightcurve_clean.csv --window 401
"""

import argparse
from pathlib import Path
import logging

import lightkurve as lk
from src.utils import DATA_DIR, PLOTS_DIR, configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def detrend_lightcurve(lc: lk.LightCurve, window_length: int = 401, polyorder: int = 2, return_trend: bool = False):
    """
    Detrend using LightKurve's flatten (which defaults to Savitzky-Golay)
    window_length should be an odd integer (samples). If lc.time spacing is large,
    adapt window_length accordingly.
    """
    logger.info("Detrending with window_length=%s, polyorder=%s", window_length, polyorder)
    flattened = lc.flatten(window_length=window_length, polyorder=polyorder, return_trend=return_trend)
    return flattened

def save_outputs(flattened, out_prefix: Path):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(flattened, tuple):
        lc_flat, trend = flattened
        lc_flat.to_fits(out_prefix.with_suffix(".detrended.fits"), overwrite=True)
        lc_flat.to_pandas().to_csv(out_prefix.with_suffix(".detrended.csv"), index=False)
        # Save trend separately
        trend_table = trend.to_table()
        from astropy.table import Table
        Table(trend_table).write(out_prefix.with_suffix(".trend.fits"), overwrite=True)
    else:
        flattened.to_fits(out_prefix.with_suffix(".detrended.fits"), overwrite=True)
        flattened.to_pandas().to_csv(out_prefix.with_suffix(".detrended.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input cleaned lightcurve (FITS or CSV)")
    parser.add_argument("--out", default=str(DATA_DIR / "lightcurve_detrended"), help="Output prefix")
    parser.add_argument("--window", type=int, default=401, help="Window length (odd integer) for smoothing")
    parser.add_argument("--polyorder", type=int, default=2, help="Polynomial order for Savitzky-Golay")
    args = parser.parse_args()

    # load using lightkurve convenience
    lc = lk.LightCurve.read(args.input) if Path(args.input).suffix.lower() in (".fits", ".lc") else lk.read(args.input)
    flattened = detrend_lightcurve(lc, window_length=args.window, polyorder=args.polyorder, return_trend=False)
    save_outputs(flattened, Path(args.out))

if __name__ == "__main__":
    main()

