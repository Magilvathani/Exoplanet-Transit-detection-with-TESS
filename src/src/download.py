"""
download.py

Search and download TESS light curves using lightkurve.
Saves stitched lightcurve as FITS and CSV into data/.

Usage:
    python src/download.py --target "TIC 25155310" --max_search 20
"""

import argparse
from pathlib import Path
from typing import Optional
import logging

import lightkurve as lk
from lightkurve import search_lightcurve
from astropy.table import Table

from src.utils import DATA_DIR, configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def search_and_download(target: str, mission: str = "TESS", limit: Optional[int] = None):
    """
    Search for and download all lightcurves matching `target`.
    Returns stitched LightCurve if possible.
    """
    logger.info("Searching for target: %s (mission=%s)", target, mission)
    search_result = search_lightcurve(target, mission=mission)
    logger.info("Found %d results", len(search_result))
    if len(search_result) == 0:
        raise ValueError(f"No lightcurve results for {target}")

    # Optionally limit the number of results to download
    if limit:
        logger.info("Limiting download to first %d search results", limit)
        search_result = search_result[:limit]

    logger.info("Downloading lightcurves (this may take time depending on network)")
    lc_collection = search_result.download_all()
    if lc_collection is None:
        raise RuntimeError("download_all() returned None")

    logger.info("Stitching lightcurves into a single LightCurve")
    try:
        stitched = lc_collection.stitch()
    except Exception:
        # fallback: try to append manually
        logger.warning("Stitch failed; trying manual append")
        stitched = lc_collection[0]
        for lc in lc_collection[1:]:
            stitched = stitched.append(lc)

    return stitched

def save_lightcurve(lc, out_prefix: str):
    """Save lightcurve to FITS and CSV."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fits_path = out_prefix.with_suffix(".fits")
    csv_path = out_prefix.with_suffix(".csv")

    logger.info("Saving FITS -> %s", fits_path)
    lc.to_fits(fits_path, overwrite=True)

    logger.info("Saving CSV -> %s", csv_path)
    # convert to astropy table then pandas for clean CSV
    try:
        df = lc.to_pandas()
        df.to_csv(csv_path, index=False)
    except Exception:
        # fallback - save astropy table
        tbl = lc.to_table()
        Table(tbl).write(csv_path, format="csv", overwrite=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help='Target name (e.g., "TIC 25155310" or "WASP-121")')
    parser.add_argument("--out", default=str(DATA_DIR / "lightcurve_raw"), help="Output prefix (without extension)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of search results to download")
    args = parser.parse_args()

    lc = search_and_download(args.target, limit=args.limit)
    save_lightcurve(lc, args.out + f"_{args.target.replace(' ', '_')}")

if __name__ == "__main__":
    main()

