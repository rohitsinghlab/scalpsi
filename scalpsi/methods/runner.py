"""
Core logic for running prediction methods on a dataset.

Method scripts live in scalpsi/methods/scripts/.
They are run inside an Apptainer container via conda environments.
"""

import os
import subprocess
from scalpsi import config

# Method configurations: name -> script file, conda env
METHODS = {
    'GEARS': {
        'script': 'myGears.py',
        'conda_env': 'gears',
    },
    'scGPT': {
        'script': 'myscGPT1.py',
        'conda_env': 'scGPT',
    },
    'GenePert': {
        'script': 'myGenePert.py',
        'conda_env': 'gears',
    },
    'linearModel_mean': {
        'script': 'linearModel_mean.py',
        'conda_env': 'gears',
    },
    'CPA': {
        'script': 'myCPA.py',
        'conda_env': 'cpa',
    },
}


def check_dataset_exists(dataset_name, hvg_threshold, data_dir=None):
    """Check if preprocessed dataset files exist."""
    if data_dir is None:
        data_dir = str(config.DATASET2_DIR)
    data_path = os.path.join(data_dir, dataset_name, f'hvg{hvg_threshold}')
    required = [
        os.path.join(data_path, f'filter_hvg{hvg_threshold}_logNor.h5ad'),
        os.path.join(data_path, 'filter_hvgall_logNor.h5ad'),
    ]
    missing = [f for f in required if not os.path.isfile(f)]
    if missing:
        print("ERROR: Required data files not found:")
        for f in missing:
            print(f"  - {f}")
        print(f"\nRun preprocessing first:")
        print(f"  python scripts/preprocess.py --path filtered_datasets/{dataset_name}_filtered.h5ad --name {dataset_name}")
        return False
    return True


def run_method(method_name, dataset_name, hvg_threshold=5000,
               split_index=0, data_dir=None, script_dir=None,
               split_dir=None, dry_run=False, suffix=''):
    """
    Run a specific method on the dataset with JSON-based splits.
    """
    if data_dir is None:
        data_dir = str(config.DATASET2_DIR)
    if script_dir is None:
        script_dir = str(config.SCRIPT_DIR)
    if split_dir is None:
        split_dir = str(config.SPLIT_DIR)

    if method_name not in METHODS:
        print(f"ERROR: Unknown method '{method_name}'")
        print(f"Available: {', '.join(METHODS.keys())}")
        return False

    method_info = METHODS[method_name]
    script_path = os.path.join(script_dir, method_info['script'])

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return False

    print("=" * 60)
    print(f"Running {method_name}")
    print("=" * 60)
    print(f"Dataset:      {dataset_name}")
    print(f"HVG thresh:   {hvg_threshold}")
    print(f"Script:       {method_info['script']}")
    print(f"Conda env:    {method_info['conda_env']}")
    print(f"Split index:  {split_index} (JSON-based)")
    print("=" * 60)

    if dry_run:
        print("[DRY RUN] Would execute this method")
        return True

    # Build command arguments
    cmd_args = [
        f"--split-index {split_index}",
        f"--dataset {dataset_name}",
        f"--data-dir {data_dir}",
        f"--split-dir {split_dir}",
    ]
    if suffix:
        cmd_args.append(f"--suffix {suffix}")

    try:
        env = os.environ.copy()
        cache_dir = '/tmp/numba_cache'
        os.makedirs(cache_dir, exist_ok=True)
        env['NUMBA_CACHE_DIR'] = cache_dir
        env['PYTHONWARNINGS'] = 'ignore'

        cmd = (
            f"cd {script_dir} && conda run --no-capture-output "
            f"-n {method_info['conda_env']} python {method_info['script']} "
            f"{' '.join(cmd_args)}"
        )
        print(f"Executing: {cmd}")
        print(f"{'=' * 60}\n")

        result = subprocess.run(cmd, shell=True, env=env)

        print(f"\n{'=' * 60}")
        if result.returncode == 0:
            print(f"  {method_name} completed successfully")
            return True
        else:
            print(f"  {method_name} failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print(f"  {method_name} failed with exception: {e}")
        return False
