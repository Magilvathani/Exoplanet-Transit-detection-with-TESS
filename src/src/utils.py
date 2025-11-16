"""
Utility helpers for the TESS transit pipeline.

Contains:
- path helpers
- save/load helpers
- simple logging setup
"""

from pathlib import Path
import json
import logging
from typing import Any, Dict

# Project root is repository root (one level up from src/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PLOTS_DIR = REPO_ROOT / "plots"

# Ensure directories exist
for d in (DATA_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def configure_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON file with indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def read_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

