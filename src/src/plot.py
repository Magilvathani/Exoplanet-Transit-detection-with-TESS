"""
plot.py

Produce publication-quality plots:
- Raw/detrended lightcurve
- BLS periodogram
- Phase-folded transit plot for the best period

Usage:
    python src/plot.py --lc data/lightcurve_detrended.detrended.fits --bls data/bls_results.csv --period 1.274
"""

import argparse
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightkurve as lk

from src.utils import PLOTS_DIR, configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def plot_lightcurve(lc: lk.LightCurve, out_path: Path, title: str = "Lightcurve"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(lc.time.value, lc.flux, ".", markersize=2)
    plt.xlabel("Time (BTJD or JD)")
    plt.ylabel("Flux (normalized)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info("Saved plot %s", out_path)

def plot_bls(periods: np.ndarray, power: np.ndarray, out_path: Path):
    plt.figure(figsize=(10, 4))
    plt.plot(periods, power, lw=0.6)
    plt.xlabel("Period (days)")
    plt.ylabel("BLS Power")
    plt.title("BLS Periodogram")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info("Saved plot %s", out_path)

def plot_phase_fold(lc: lk.LightCurve, period: float, out_path: Path, bins: int = 200):
    folded = lc.fold(period=period)
    plt.figure(figsize=(6, 5))
    plt.plot(folded.phase, folded.flux, ".", markersize=2, alpha=0.6, label="data")
    # binned
    phases = np.linspace(-0.5, 0.5, bins)
    bin_means = []
    bin_centers = (phases[:-1] + phases[1:]) / 2
    import numpy as np
    inds = np.digitize(folded.phase, phases)
    for i in range(1, len(phases)):
        sel = folded.flux[inds == i]
        if len(sel):
            bin_means.append(np.nanmean(sel))
        else:
            bin_means.append(np.nan)
    plt.plot(bin_centers, bin_means, "-", lw=1.5, label="binned")
    plt.xlabel("Phase")
    plt.ylabel("Flux")
    plt.title(f"Phase-folded (P={period:.6f} d)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info("Saved plot %s", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc", required=True, help="Lightcurve file (FITS/CSV)")
    parser.add_argument("--bls", required=False, help="BLS CSV file for plotting periodogram")
    parser.add_argument("--period", type=float, default=None, help="Period for folding (if provided)")
    parser.add_argument("--out-prefix", default=str(PLOTS_DIR / "figure"), help="Output prefix")
    args = parser.parse_args()

    lc = lk.LightCurve.read(args.lc)
    plot_lightcurve(lc, Path(args.out_prefix + "_lightcurve.png"))
    if args.bls:
        df = pd.read_csv(args.bls)
        plot_bls(df["period"].values, df["power"].values, Path(args.out_prefix + "_bls.png"))
    if args.period:
        plot_phase_fold(lc, args.period, Path(args.out_prefix + "_phase.png"))

if __name__ == "__main__":
    main()

