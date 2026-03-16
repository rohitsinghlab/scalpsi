"""
Core logic for running all genetic perturbation methods on a dataset.

Method scripts (myGears.py, myscGPT1.py, etc.) must be located in
SCALPSI_SCRIPT_DIR (default: $SCALPSI_BASE_DIR/manuscript2/genetic).

Set environment variable SCALPSI_SCRIPT_DIR to point to your method scripts.
"""

import os
import shutil
from scalpsi import config

# Method configurations: name -> script file, conda env, GEARS dependency
METHODS = {
    'GEARS': {
        'script': 'myGears.py',
        'conda_env': 'gears',
        'needs_gears': False,
    },
    'scFoundation': {
        'script': 'myscFoundation.py',
        'conda_env': 'gears',
        'needs_gears': True,
    },
    'scGPT': {
        'script': 'myscGPT1.py',
        'conda_env': 'scGPT',
        'needs_gears': False,
    },
    'GenePert': {
        'script': 'myGenePert.py',
        'conda_env': 'gears',
        'needs_gears': False,
    },
    'GeneCompass': {
        'script': 'myGeneCompass.py',
        'conda_env': 'gears',
        'needs_gears': True,
    },
    'AttentionPert': {
        'script': 'myAttentionPert.py',
        'conda_env': 'gears',
        'needs_gears': True,
    },
    'scELMo': {
        'script': 'myscELMo.py',
        'conda_env': 'gears',
        'needs_gears': True,
    },
    'scouter': {
        'script': 'myscouter.py',
        'conda_env': 'cpa',
        'needs_gears': True,
    },
    'baseMLP': {
        'script': 'genetic_baseMLP.py',
        'conda_env': 'cpa',
        'needs_gears': True,
    },
    'baseReg': {
        'script': 'genetic_baseReg.py',
        'conda_env': 'cpa',
        'needs_gears': True,
    },
    'linearModel': {
        'script': 'linearModel.py',
        'conda_env': 'linearModel',
        'needs_gears': True,
    },
    'linearModel_mean': {
        'script': 'linearModel_mean.py',
        'conda_env': 'gears',
        'needs_gears': False,
    },
    'CPA': {
        'script': 'myCPA.py',
        'conda_env': 'cpa',
        'needs_gears': False,
    }
}


def check_dataset_exists(dataset_name, hvg_threshold, base_dir=None):
    """Check if preprocessed dataset files exist."""
    if base_dir is None:
        base_dir = str(config.BASE_DIR)
    data_dir = os.path.join(base_dir, 'DataSet2', dataset_name, f'hvg{hvg_threshold}')
    required = [
        os.path.join(data_dir, f'filter_hvg{hvg_threshold}_logNor.h5ad'),
        os.path.join(data_dir, 'filter_hvgall_logNor.h5ad'),
    ]
    missing = [f for f in required if not os.path.isfile(f)]
    if missing:
        print("ERROR: Required data files not found:")
        for f in missing:
            print(f"  - {f}")
        print(f"\nRun preprocessing first:")
        print(f"  scalpsi-preprocess --path your_data.h5ad --name {dataset_name}")
        return False
    return True


def check_gears_data(dataset_name, hvg_threshold, split_index, base_dir=None):
    """Check if GEARS has been run (needed by most other methods)."""
    if base_dir is None:
        base_dir = str(config.BASE_DIR)
    gears_dir = os.path.join(base_dir, 'DataSet2', dataset_name,
                             f'hvg{hvg_threshold}', 'GEARS')
    splits_dir = os.path.join(gears_dir, 'data', 'train', 'splits')
    result_file = os.path.join(gears_dir, f'savedModels_split{split_index}', 'result.h5ad')

    issues = []
    if not os.path.isdir(splits_dir):
        issues.append(f"GEARS splits dir missing: {splits_dir}")
    if not os.path.isfile(result_file):
        issues.append(f"GEARS result missing: {result_file}")
    return issues


def _find_gears_data(dataset_dir, preferred_hvg):
    """Find GEARS data directory at any HVG threshold."""
    for hvg in [preferred_hvg, 5000, 2000, 1000, 100]:
        candidate = os.path.join(dataset_dir, f'hvg{hvg}', 'GEARS', 'data')
        if os.path.exists(os.path.join(candidate, 'gene2go_all.pkl')):
            return candidate
    return None


def setup_gears_data(method_name, dataset_name, hvg_threshold, base_dir=None):
    """
    Copy GEARS data files to method directories that need them.
    Most methods need gene2go_all.pkl and train/test splits.
    scFoundation needs gene2go.pkl (renamed from gene2go_all.pkl).
    """
    if base_dir is None:
        base_dir = str(config.BASE_DIR)
    dataset_dir = os.path.join(base_dir, 'DataSet2', dataset_name)
    hvg_dir = os.path.join(dataset_dir, f'hvg{hvg_threshold}')

    standard_gears_methods = ['GEARS', 'GeneCompass', 'AttentionPert', 'scELMo']

    if method_name in standard_gears_methods and method_name != 'GEARS':
        target_data = os.path.join(hvg_dir, method_name, 'data')
        if not os.path.exists(os.path.join(target_data, 'gene2go_all.pkl')):
            source = _find_gears_data(dataset_dir, hvg_threshold)
            if source:
                print(f"Copying GEARS data to {method_name}...")
                if os.path.exists(target_data):
                    shutil.rmtree(target_data)
                os.makedirs(os.path.dirname(target_data), exist_ok=True)
                shutil.copytree(source, target_data)
                print("Done\n")

    if method_name == 'scFoundation':
        target_data = os.path.join(hvg_dir, method_name, 'data')
        gene2go_path = os.path.join(target_data, 'gene2go.pkl')
        needs_fix = not os.path.exists(gene2go_path) or os.path.islink(gene2go_path)

        if needs_fix:
            source = _find_gears_data(dataset_dir, hvg_threshold)
            if source:
                print("Setting up scFoundation data files...")
                if os.path.exists(target_data):
                    shutil.rmtree(target_data)
                shutil.copytree(source, target_data)
                src = os.path.join(target_data, 'gene2go_all.pkl')
                dst = os.path.join(target_data, 'gene2go.pkl')
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                go_csv_src = os.path.join(source, 'train', 'go.csv')
                go_csv_dst = os.path.join(target_data, 'train', 'go.csv')
                if os.path.isfile(go_csv_src):
                    os.makedirs(os.path.dirname(go_csv_dst), exist_ok=True)
                    shutil.copy2(go_csv_src, go_csv_dst)
                print("Done\n")
            else:
                print("WARNING: No GEARS data found. Run GEARS first.")


def run_method(method_name, dataset_name, hvg_threshold=5000,
               split_index=0, base_dir=None, script_dir=None,
               dry_run=False, suffix=''):
    """
    Run a specific method on the dataset with JSON-based splits.

    Parameters
    ----------
    method_name : str
        One of the keys in METHODS.
    dataset_name : str
        Dataset name (must exist in DataSet2/).
    hvg_threshold : int
        HVG threshold (100, 1000, 2000, 5000).
    split_index : int
        Split index (0-4) from JSON split files.
    base_dir : str or None
        Base directory. Defaults to config.BASE_DIR.
    script_dir : str or None
        Directory containing method scripts. Defaults to config.SCRIPT_DIR.
    dry_run : bool
        If True, print the command without executing.
    suffix : str
        Optional suffix for output directories.
    """
    import subprocess

    if base_dir is None:
        base_dir = str(config.BASE_DIR)
    if script_dir is None:
        script_dir = str(config.SCRIPT_DIR)

    if method_name not in METHODS:
        print(f"ERROR: Unknown method '{method_name}'")
        print(f"Available: {', '.join(METHODS.keys())}")
        return False

    method_info = METHODS[method_name]
    script_path = os.path.join(script_dir, method_info['script'])

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        print(f"Set SCALPSI_SCRIPT_DIR to point to your method scripts directory.")
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

    # Check GEARS dependency
    if method_info['needs_gears']:
        issues = check_gears_data(dataset_name, hvg_threshold, split_index, base_dir)
        if issues:
            print(f"\nNOTE: GEARS data not found for split{split_index}:")
            for issue in issues:
                print(f"  - {issue}")
            print("Run GEARS first for this dataset.\n")

    # Set up shared data files
    setup_gears_data(method_name, dataset_name, hvg_threshold, base_dir)

    # Build command arguments
    cmd_args = [
        f"--split-index {split_index}",
        f"--dataset {dataset_name}"
    ]
    if suffix:
        cmd_args.append(f"--suffix {suffix}")
    if method_name in ['baseMLP', 'baseReg']:
        cmd_args.append("--mode single")

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
