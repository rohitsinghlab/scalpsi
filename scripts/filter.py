"""
CLI: Filter a raw perturbation dataset to cells with genes in the CV splits.

Usage:
    scalpsi-filter --dataset K562 --input rawdata/perturbSeq/K562_raw_sc.h5ad --output filtered/K562_filtered.h5ad
    scalpsi-filter --dataset HepG2 --input rawdata/perturbSeq/hepg2_raw_sc.h5ad --output filtered/HepG2_filtered.h5ad
"""

import argparse
import sys

from scalpsi.filter.core import VALID_DATASETS, filter_dataset


def main():
    valid = ", ".join(VALID_DATASETS.keys())
    parser = argparse.ArgumentParser(
        description=f"Filter raw perturbation h5ad to cells with genes in CV splits. Supported datasets: {valid}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(VALID_DATASETS.keys()),
                        help="Dataset name")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to raw h5ad file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write filtered h5ad file")
    parser.add_argument("--splits-dir", type=str, default=None,
                        help="Directory containing split*.json files (default: data/splits/ in repo)")
    parser.add_argument("--max-controls", type=int, default=0,
                        help="Maximum number of non-targeting control cells to keep (0 = keep all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for control subsampling")

    args = parser.parse_args()

    filter_dataset(
        dataset=args.dataset,
        input_path=args.input,
        output_path=args.output,
        splits_dir=args.splits_dir,
        max_controls=args.max_controls,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
