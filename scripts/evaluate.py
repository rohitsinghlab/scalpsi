"""
CLI: Evaluate perturbation prediction results.

Usage:
    scalpsi-evaluate --dataset hepg2 jurkat rpe1 --hvg 5000 --methods GEARS scGPT CPA
    scalpsi-evaluate --dataset toy --hvg 1000 --methods GEARS --splits 0 1 2

Set SCALPSI_DATA_DIR to point to your DataSet2 directory.
"""

import argparse
import sys

from scalpsi import config
from scalpsi.evaluation.performance import run_evaluation, DEFAULT_METRICS


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate perturbation prediction results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, nargs='+', required=True,
                        help='Dataset name(s)')
    parser.add_argument('--hvg', type=int, default=1000,
                        help='HVG threshold')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to evaluate (default: all found in directory)')
    parser.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Splits to evaluate')
    parser.add_argument('--base-dir', type=str, default=None,
                        help=f'DataSet2 directory (default: $SCALPSI_DATA_DIR or {config.DATASET2_DIR})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers')

    args = parser.parse_args()

    base_dir = args.base_dir or str(config.DATASET2_DIR)

    print(f"{'='*60}")
    print(f"Performance Evaluation")
    print(f"{'='*60}")
    print(f"Datasets:  {args.dataset}")
    print(f"HVG:       {args.hvg}")
    print(f"Splits:    {args.splits}")
    print(f"Workers:   {args.workers}")
    print(f"Base dir:  {base_dir}")
    print(f"{'='*60}\n")

    try:
        run_evaluation(
            datasets=args.dataset,
            hvg=args.hvg,
            methods=args.methods,
            splits=args.splits,
            base_dir=base_dir,
            output=args.output,
            workers=args.workers,
        )
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
