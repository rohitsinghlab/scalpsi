"""
CLI: Run prediction methods on a dataset.

Usage (inside Apptainer container):
    python scripts/run_methods.py --dataset K562 --methods GEARS --split-index 0
    python scripts/run_methods.py --dataset K562 --methods GEARS CPA scGPT --split-index 0
    python scripts/run_methods.py --dataset K562 --split-index 0    # runs all methods
"""

import argparse
import sys

from scalpsi import config
from scalpsi.methods.runner import METHODS, check_dataset_exists, run_method


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction methods with JSON splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (must exist in preprocessed/)')
    parser.add_argument('--methods', type=str, nargs='+', default=['all'],
                        help=f'Methods to run. Options: all, {", ".join(METHODS.keys())}')
    parser.add_argument('--split-index', type=int, default=0, choices=list(range(10)),
                        help='Split index (0-4) from JSON split files')
    parser.add_argument('--hvg', type=int, default=5000,
                        choices=[100, 1000, 2000, 5000],
                        help='Number of highly variable genes')
    parser.add_argument('--data-dir', type=str, default=None,
                        help=f'Preprocessed data directory (default: {config.DATASET2_DIR})')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for output directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run without executing')
    parser.add_argument('--list-methods', action='store_true',
                        help='List available methods and exit')

    args = parser.parse_args()

    if args.list_methods:
        print("Available methods:")
        print("=" * 60)
        for method, info in METHODS.items():
            print(f"  {method:20s} - {info['script']:25s} (env: {info['conda_env']})")
        return 0

    data_dir = args.data_dir or str(config.DATASET2_DIR)

    if not check_dataset_exists(args.dataset, args.hvg, data_dir):
        return 1

    if 'all' in args.methods:
        methods_to_run = list(METHODS.keys())
    else:
        methods_to_run = args.methods

    # Ensure GEARS runs first if included with other methods
    if len(methods_to_run) > 1 and 'GEARS' in methods_to_run:
        idx = methods_to_run.index('GEARS')
        if idx != 0:
            methods_to_run.remove('GEARS')
            methods_to_run.insert(0, 'GEARS')
            print("NOTE: Moved GEARS to run first\n")

    print(f"Dataset:       {args.dataset}")
    print(f"HVG threshold: {args.hvg}")
    print(f"Methods:       {', '.join(methods_to_run)}")
    print(f"Split index:   {args.split_index}")
    print(f"Data dir:      {data_dir}")
    if args.suffix:
        print(f"Suffix:        {args.suffix}")
    print("")

    results = {}
    for method in methods_to_run:
        success = run_method(method, args.dataset, args.hvg, args.split_index,
                             data_dir=data_dir, dry_run=args.dry_run, suffix=args.suffix)
        results[method] = success

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method, success in results.items():
        status = "Success" if success else "Failed"
        print(f"  {method:20s}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
