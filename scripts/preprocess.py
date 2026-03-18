"""
CLI: Preprocess a perturbation dataset.

Usage:
    scalpsi-preprocess --path toy.h5ad
    scalpsi-preprocess --path toy.h5ad --name toy --output-dir /path/to/DataSet2

Pass --output-dir to override the default preprocessed directory.
"""

import argparse
import os
import sys

from scalpsi import config
from scalpsi.preprocess.core import preprocess_and_save


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess perturbation datasets with normalization, log1p, and HVG selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--path", type=str, required=True,
                        help="Path to input h5ad file")
    parser.add_argument("--name", type=str, default=None,
                        help="Dataset name (default: derived from filename)")
    parser.add_argument("--min-cells", type=int, default=50,
                        help="Minimum cells per perturbation to keep")
    parser.add_argument("--max-pert", type=int, default=0,
                        help="Max cells per perturbation (0 = no downsampling)")
    parser.add_argument("--max-ctrl", type=int, default=0,
                        help="Max control cells (0 = no downsampling)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Base output directory (default: {config.DATASET2_DIR})")

    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"ERROR: File not found: {args.path}")
        return 1

    if args.name is None:
        basename = os.path.basename(args.path)
        dataset_name = basename.replace('.h5ad', '').replace('.H5AD', '')
        print(f"Auto-detected dataset name: {dataset_name}")
    else:
        dataset_name = args.name

    output_dir = args.output_dir or str(config.DATASET2_DIR)

    print("="*60)
    print("Preprocessing Configuration")
    print("="*60)
    print(f"Input file:         {args.path}")
    print(f"Dataset name:       {dataset_name}")
    print(f"Min cells/pert:     {args.min_cells}")
    print(f"Max cells/pert:     {args.max_pert if args.max_pert > 0 else 'No limit'}")
    print(f"Max control cells:  {args.max_ctrl if args.max_ctrl > 0 else 'No limit'}")
    print(f"Output directory:   {output_dir}")
    print("="*60 + "\n")

    preprocess_and_save(
        input_file=args.path,
        dataset_name=dataset_name,
        minNums=args.min_cells,
        domaxNumsPerturb=args.max_pert,
        domaxNumsControl=args.max_ctrl,
        output_base_dir=output_dir,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
