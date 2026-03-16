"""
Centralized path configuration for scalpsi.

Override any path via environment variables before running:

    export SCALPSI_DATA_DIR=/path/to/DataSet2       # processed datasets output
    export SCALPSI_BASE_DIR=/path/to/Pertb_benchmark # parent of DataSet2
    export SCALPSI_SCRIPT_DIR=/path/to/method/scripts # GEARS, scGPT, etc.

If SCALPSI_DATA_DIR is not set, defaults to ./DataSet2 relative to the
current working directory.
"""

import os
from pathlib import Path

# Root of the installed package (scalpsi/)
_PKG_ROOT = Path(__file__).parent

# Root of the repo (one level up from scalpsi/)
REPO_ROOT = _PKG_ROOT.parent

# Reference data directory (data/ in repo — gene_info.tsv, splits, etc.)
DATA_DIR = REPO_ROOT / "data"

# Processed datasets directory — override with SCALPSI_DATA_DIR
# Default: ./DataSet2 relative to current working directory
DATASET2_DIR = Path(os.environ.get("SCALPSI_DATA_DIR", "DataSet2"))

# Benchmark base directory — override with SCALPSI_BASE_DIR
# Default: parent of DATASET2_DIR
BASE_DIR = Path(os.environ.get("SCALPSI_BASE_DIR", str(DATASET2_DIR.parent)))

# Method scripts directory (GEARS, scGPT, etc.) — override with SCALPSI_SCRIPT_DIR
SCRIPT_DIR = Path(
    os.environ.get(
        "SCALPSI_SCRIPT_DIR",
        str(BASE_DIR / "manuscript2" / "genetic"),
    )
)
