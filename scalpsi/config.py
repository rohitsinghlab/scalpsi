"""
Centralized path configuration for scalpsi.
"""

from pathlib import Path

# Root of the installed package (scalpsi/)
_PKG_ROOT = Path(__file__).parent

# Root of the repo (one level up from scalpsi/)
REPO_ROOT = _PKG_ROOT.parent

# Reference data directory (data/ in repo — gene_info.tsv, splits, etc.)
DATA_DIR = REPO_ROOT / "data"

# Default output directory for preprocessed datasets (h5ad + DEG files).
# Created automatically by the preprocess step.
# All downstream scripts (run, evaluate) read from here by default.
DATASET2_DIR = REPO_ROOT / "preprocessed"

# Method scripts directory (myGears.py, myCPA.py, etc.)
SCRIPT_DIR = _PKG_ROOT / "methods" / "scripts"

# Split files directory
SPLIT_DIR = DATA_DIR / "splits"
