"""
CLI: Gene-level performance evaluation on benchmark results.

Usage:
    scalpsi-evaluate-genes --dataset hepg2 jurkat rpe1 --hvg 5000 --methods GEARS scGPT
    scalpsi-evaluate-genes --dataset hepg2 --hvg 5000 --methods GEARS --splits 0 1 2 --num-deg 1000

Pass --base-dir to override the default preprocessed directory.
"""

import argparse
import sys

from scalpsi import config
from scalpsi.evaluation.gene_performance import run_gene_evaluation


def main():
    parser = argparse.ArgumentParser(
        description='Gene-level performance evaluation',
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
                        help=f'Preprocessed data directory (default: {config.DATASET2_DIR})')
    parser.add_argument('--num-deg', type=int, default=2000,
                        help='Number of top DEGs per perturbation')
    parser.add_argument('--all-genes', action='store_true',
                        help='Evaluate all genes (skips DEG_hvg*.pkl loading)')
    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Output file prefix')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers')

    args = parser.parse_args()

    base_dir = args.base_dir or str(config.DATASET2_DIR)

    print(f"{'='*60}")
    print(f"Gene-Level Performance Evaluation")
    print(f"{'='*60}")
    print(f"Datasets:  {args.dataset}")
    print(f"HVG:       {args.hvg}")
    if args.all_genes:
        print("Gene set:  ALL available genes")
    else:
        print(f"Top DEGs:  {args.num_deg}")
    print(f"Splits:    {args.splits}")
    print(f"Workers:   {args.workers}")
    print(f"Base dir:  {base_dir}")
    print(f"{'='*60}\n")

    try:
        run_gene_evaluation(
            datasets=args.dataset,
            hvg=args.hvg,
            methods=args.methods,
            splits=args.splits,
            base_dir=base_dir,
            output_prefix=args.output_prefix,
            num_deg=args.num_deg,
            use_all_genes=args.all_genes,
            workers=args.workers,
        )
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
