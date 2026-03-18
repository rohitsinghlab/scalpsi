"""
Unified split utilities for loading train/val/test splits from JSON files.
"""

import json
import os
from typing import Dict, List, Tuple, Set


def load_split_json(split_index: int, split_dir: str = "data/splits") -> Dict[str, List[str]]:
    """
    Load train/val/test split from JSON file.

    Args:
        split_index: Index 0-4 for split0.json to split4.json
        split_dir: Directory containing split JSON files

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing gene name lists
    """
    if not 0 <= split_index <= 9:
        raise ValueError(f"split_index must be 0-9, got {split_index}")

    split_file = os.path.join(split_dir, f"split{split_index}.json")

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, 'r') as f:
        split_dict = json.load(f)

    # Validate structure
    required_keys = {'train', 'val', 'test'}
    if not required_keys.issubset(split_dict.keys()):
        raise ValueError(f"Split must contain {required_keys}, got {split_dict.keys()}")

    return split_dict


def get_split_perturbations(
    split_index: int,
    available_perturbations: List[str],
    split_dir: str = "data/splits",
    verbose: bool = True
) -> Tuple[List[str], List[str], List[str], Dict]:
    """
    Load split and map gene names to perturbations in the dataset.

    Args:
        split_index: Split index 0-4
        available_perturbations: All perturbations in adata.obs['perturbation'].unique()
        split_dir: Directory with split JSON files
        verbose: Print progress and warnings

    Returns:
        (train_perts, val_perts, test_perts, statistics_dict)
    """
    # Load JSON
    split_dict = load_split_json(split_index, split_dir)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Loading Split {split_index} from {split_dir}/split{split_index}.json")
        print(f"{'='*70}")
        print(f"JSON split contains:")
        print(f"  Train: {len(split_dict['train'])} genes")
        print(f"  Val:   {len(split_dict['val'])} genes")
        print(f"  Test:  {len(split_dict['test'])} genes")

    # Convert available perturbations to set for fast lookup
    available_set = set(available_perturbations)

    # Match genes to perturbations
    train_perts = []
    val_perts = []
    test_perts = []
    train_missing = set()
    val_missing = set()
    test_missing = set()

    # Process train
    for gene in split_dict['train']:
        if gene == 'non-targeting':
            continue  # Skip non-targeting from JSON
        if gene in available_set:
            train_perts.append(gene)
        else:
            train_missing.add(gene)

    # Process val
    for gene in split_dict['val']:
        if gene == 'non-targeting':
            continue
        if gene in available_set:
            val_perts.append(gene)
        else:
            val_missing.add(gene)

    # Process test
    for gene in split_dict['test']:
        if gene == 'non-targeting':
            continue
        if gene in available_set:
            test_perts.append(gene)
        else:
            test_missing.add(gene)

    stats = {
        'json_train': len([g for g in split_dict['train'] if g != 'non-targeting']),
        'json_val': len([g for g in split_dict['val'] if g != 'non-targeting']),
        'json_test': len([g for g in split_dict['test'] if g != 'non-targeting']),
        'matched_train': len(train_perts),
        'matched_val': len(val_perts),
        'matched_test': len(test_perts),
        'missing_train': len(train_missing),
        'missing_val': len(val_missing),
        'missing_test': len(test_missing),
    }

    if verbose:
        print(f"\nMatched to dataset:")
        print(f"  Train: {stats['matched_train']}/{stats['json_train']} perturbations")
        print(f"  Val:   {stats['matched_val']}/{stats['json_val']} perturbations")
        print(f"  Test:  {stats['matched_test']}/{stats['json_test']} perturbations")

        if train_missing or val_missing or test_missing:
            print(f"\nMissing from dataset (genes in JSON but not in data):")
            if train_missing:
                print(f"  Train: {len(train_missing)} genes - {sorted(list(train_missing))[:5]}...")
            if val_missing:
                print(f"  Val: {len(val_missing)} genes - {sorted(list(val_missing))[:5]}...")
            if test_missing:
                print(f"  Test: {len(test_missing)} genes - {sorted(list(test_missing))[:5]}...")

        print(f"{'='*70}\n")

    return train_perts, val_perts, test_perts, stats


def assert_split_correctness(
    train_perts: List[str],
    val_perts: List[str],
    test_perts: List[str],
    split_index: int,
    split_dir: str = "data/splits"
) -> None:
    """
    Assert that splits are correctly applied from JSON.

    Validates:
    1. No overlap between train/val/test
    2. All perturbations come from the correct JSON split

    Raises AssertionError if validation fails.
    """
    # Load JSON
    split_dict = load_split_json(split_index, split_dir)

    # Convert to sets
    train_set = set(train_perts)
    val_set = set(val_perts)
    test_set = set(test_perts)

    # Get JSON genes (excluding non-targeting)
    json_train = {g for g in split_dict['train'] if g != 'non-targeting'}
    json_val = {g for g in split_dict['val'] if g != 'non-targeting'}
    json_test = {g for g in split_dict['test'] if g != 'non-targeting'}

    # Check 1: No overlap
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    assert not overlap_tv, f"❌ Train/Val overlap detected: {overlap_tv}"
    assert not overlap_tt, f"❌ Train/Test overlap detected: {overlap_tt}"
    assert not overlap_vt, f"❌ Val/Test overlap detected: {overlap_vt}"

    # Check 2: All perturbations are in correct JSON split
    assert train_set.issubset(json_train), \
        f"❌ Train has genes not in JSON train set: {train_set - json_train}"
    assert val_set.issubset(json_val), \
        f"❌ Val has genes not in JSON val set: {val_set - json_val}"
    assert test_set.issubset(json_test), \
        f"❌ Test has genes not in JSON test set: {test_set - json_test}"

    print(f"\n{'='*70}")
    print(f"✓ SPLIT VALIDATION PASSED for split{split_index}")
    print(f"{'='*70}")
    print(f"  ✓ No overlap between train/val/test")
    print(f"  ✓ Train: {len(train_set)}/{len(json_train)} genes from JSON")
    print(f"  ✓ Val:   {len(val_set)}/{len(json_val)} genes from JSON")
    print(f"  ✓ Test:  {len(test_set)}/{len(json_test)} genes from JSON")
    print(f"{'='*70}\n")
