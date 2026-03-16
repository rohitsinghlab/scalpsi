"""
Cross-dataset preprocessing: find shared perturbations across cell types,
then preprocess each dataset with only the shared perturbations.
"""

import os
from scalpsi import config
from scalpsi.preprocess.core import compute_shared_perturbations, preprocess_and_save


def preprocess_shared_datasets(datasets, min_cells=0, output_dir=None, save_datasets_for=None):
    """
    Preprocess multiple datasets retaining only perturbations shared across all.

    Parameters
    ----------
    datasets : list of (path, name) tuples
        e.g. [("HEK_filtered.h5ad", "HEK293T"), ("K562_filtered.h5ad", "K562")]
    min_cells : int
        Minimum cells per perturbation to keep (default: 0)
    output_dir : str or None
        Base output directory. Defaults to config.DATASET2_DIR.
    save_datasets_for : list of int or None
        Which HVG thresholds to save h5ad files for. Default: [5000].

    Returns
    -------
    shared_perts : set
        The set of shared perturbations used.
    """
    if output_dir is None:
        output_dir = str(config.DATASET2_DIR)

    # Check that no output directories already exist
    for path, name in datasets:
        out_path = os.path.join(output_dir, name)
        if os.path.exists(out_path):
            raise FileExistsError(
                f"Output directory already exists: {out_path}\n"
                "Aborting to avoid overwriting. Rename or remove it first."
            )

    # Step 1: Compute shared perturbations across all datasets
    shared_perts = compute_shared_perturbations(datasets)

    if len(shared_perts) == 0:
        raise ValueError("No shared perturbations found across datasets!")

    # Step 2: Preprocess each dataset with only shared perturbations
    for path, name in datasets:
        print(f"\n{'='*60}")
        print(f"Preprocessing {name} with {len(shared_perts)} shared perturbations...")
        print(f"{'='*60}\n")
        preprocess_and_save(
            input_file=path,
            dataset_name=name,
            minNums=min_cells,
            allowed_perturbations=shared_perts,
            output_base_dir=output_dir,
            save_datasets_for=save_datasets_for,
        )

    print(f"\n{'='*60}")
    print("All datasets preprocessed with shared perturbations!")
    print(f"{'='*60}")
    return shared_perts
