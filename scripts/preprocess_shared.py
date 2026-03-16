"""
CLI: Preprocess multiple datasets keeping only shared perturbations.

Usage:
    scalpsi-preprocess-shared \\
        --datasets HEK_filtered.h5ad:HEK293T HCT_filtered.h5ad:HCT116 K562_filtered.h5ad:K562 \\
        --output-dir /path/to/DataSet2

Set SCALPSI_DATA_DIR to change the default output directory.
"""

import argparse
import sys

from scalpsi import config
from scalpsi.preprocess.shared import preprocess_shared_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess multiple datasets with shared perturbations only"
    )
    parser.add_argument("--datasets", nargs='+', required=True,
                        help="path:name pairs (e.g. HEK_filtered.h5ad:HEK293T)")
    parser.add_argument("--min-cells", type=int, default=0,
                        help="Minimum cells per perturbation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Base output directory (default: $SCALPSI_DATA_DIR or {config.DATASET2_DIR})")
    parser.add_argument("--save-datasets-for", nargs='*', type=int, default=None,
                        help="HVG thresholds to save h5ad for (default: 5000 only)")

    args = parser.parse_args()

    datasets = []
    for d in args.datasets:
        parts = d.split(':')
        if len(parts) != 2:
            print(f"ERROR: Expected path:name format, got '{d}'")
            return 1
        datasets.append((parts[0], parts[1]))

    output_dir = args.output_dir or str(config.DATASET2_DIR)

    try:
        preprocess_shared_datasets(
            datasets=datasets,
            min_cells=args.min_cells,
            output_dir=output_dir,
            save_datasets_for=args.save_datasets_for,
        )
    except (FileExistsError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
