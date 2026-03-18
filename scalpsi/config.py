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
