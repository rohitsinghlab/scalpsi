"""
CLI: Run genetic perturbation prediction methods on a dataset.

Usage:
    scalpsi-run --dataset toy --methods GEARS --hvg 1000 --split-index 0
    scalpsi-run --dataset toy --methods GEARS scFoundation --hvg 1000 --split-index 0
    scalpsi-run --dataset toy --hvg 1000 --split-index 0    # runs all methods

Set SCALPSI_SCRIPT_DIR to point to your method scripts (myGears.py, myscGPT1.py, etc.).
Set SCALPSI_BASE_DIR to point to your benchmark base directory.
"""

import argparse
import sys

from scalpsi import config
from scalpsi.methods.runner import METHODS, check_dataset_exists, run_method


def main():
    parser = argparse.ArgumentParser(
        description="Run genetic perturbation prediction methods with JSON splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (must exist in DataSet2/)')
    parser.add_argument('--methods', type=str, nargs='+', default=['all'],
                        help=f'Methods to run. Options: all, {", ".join(METHODS.keys())}')
    parser.add_argument('--split-index', type=int, default=0, choices=list(range(10)),
                        help='Split index (0-4) from JSON split files')
    parser.add_argument('--hvg', type=int, default=1000,
                        choices=[100, 1000, 2000, 5000],
                        help='Number of highly variable genes')
    parser.add_argument('--base-dir', type=str, default=None,
                        help=f'Base directory (default: $SCALPSI_BASE_DIR or {config.BASE_DIR})')
    parser.add_argument('--script-dir', type=str, default=None,
                        help=f'Method scripts directory (default: $SCALPSI_SCRIPT_DIR or {config.SCRIPT_DIR})')
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
            dep = " (needs GEARS first)" if info['needs_gears'] else ""
            print(f"  {method:15s} - {info['script']:25s} (env: {info['conda_env']}){dep}")
        return 0

    base_dir = args.base_dir or str(config.BASE_DIR)
    script_dir = args.script_dir or str(config.SCRIPT_DIR)

    if not check_dataset_exists(args.dataset, args.hvg, base_dir):
        return 1

    if 'all' in args.methods:
        methods_to_run = list(METHODS.keys())
    else:
        methods_to_run = args.methods

    # Ensure GEARS runs first
    if len(methods_to_run) > 1 and 'GEARS' in methods_to_run:
        idx = methods_to_run.index('GEARS')
        if idx != 0:
            methods_to_run.remove('GEARS')
            methods_to_run.insert(0, 'GEARS')
            print("NOTE: Moved GEARS to run first (other methods depend on it)\n")

    print(f"Dataset:       {args.dataset}")
    print(f"HVG threshold: {args.hvg}")
    print(f"Methods:       {', '.join(methods_to_run)}")
    print(f"Split index:   {args.split_index}")
    print(f"Base dir:      {base_dir}")
    print(f"Script dir:    {script_dir}")
    if args.suffix:
        print(f"Suffix:        {args.suffix}")
    print("")

    results = {}
    for method in methods_to_run:
        success = run_method(method, args.dataset, args.hvg, args.split_index,
                             base_dir, script_dir, args.dry_run, args.suffix)
        results[method] = success

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method, success in results.items():
        status = "Success" if success else "Failed"
        print(f"  {method:15s}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
