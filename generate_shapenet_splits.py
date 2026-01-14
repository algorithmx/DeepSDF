#!/usr/bin/env python3
"""
Generate train/test split JSON files for DeepSDF from ShapeNet dataset.

This script parses the ShapeNetCore.v2/all.csv file and generates split files
in the format expected by DeepSDF's preprocess_data.py script.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set


# ShapeNet taxonomy mapping: synsetId -> class name
# Source: https://gist.github.com/tejaskhot/15ae62827d6e43b91a4b0c5c850c168e
SHAPENET_TAXONOMY = {
    "02691156": "airplane",
    "02773838": "bag",
    "02747177": "bin",
    "02791171": "bus",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorbike",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "vessel",
    "04554684": "washer",
}


def parse_shapenet_csv(csv_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse the ShapeNet all.csv file.

    Returns:
        Dict mapping synsetId -> split ('train'/'test') -> list of modelIds
    """
    models_by_synset = defaultdict(lambda: defaultdict(list))

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            synset_id = row["synsetId"]
            model_id = row["modelId"]
            split = row["split"]

            # Map split values to 'train' or 'test'
            # ShapeNet uses 'train'/'test'/'val' - we'll treat 'val' as 'test'
            if split in ("train", "test", "val"):
                split_key = split if split != "val" else "test"
                models_by_synset[synset_id][split_key].append(model_id)

    return dict(models_by_synset)


def verify_models_exist(
    shapenet_dir: str, models_by_synset: Dict[str, Dict[str, List[str]]]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Verify that model folders exist in the dataset directory.

    Returns filtered dict with only existing models.
    """
    verified = defaultdict(lambda: defaultdict(list))

    for synset_id, splits in models_by_synset.items():
        synset_path = Path(shapenet_dir) / synset_id

        if not synset_path.exists():
            print(f"Warning: Synset folder not found: {synset_path}")
            continue

        for split, model_ids in splits.items():
            for model_id in model_ids:
                model_path = synset_path / model_id
                if model_path.exists():
                    verified[synset_id][split].append(model_id)

    return dict(verified)


def create_custom_split(
    models: List[str], test_ratio: float, seed: int = 42
) -> Dict[str, List[str]]:
    """Create a random train/test split from a list of models."""
    random.seed(seed)
    shuffled = models.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_ratio))
    return {
        "train": shuffled[:split_idx],
        "test": shuffled[split_idx:],
    }


def generate_split_json(
    models_by_synset: Dict[str, Dict[str, List[str]]],
    source_name: str = "ShapeNetV2",
) -> Dict:
    """
    Generate DeepSDF-format split JSON.

    Args:
        models_by_synset: Dict mapping synsetId -> split -> list of modelIds
        source_name: Name of the data source (default: "ShapeNetV2")

    Returns:
        Dict in DeepSDF split format
    """
    # DeepSDF format: {source_name: {synsetId: [modelIds]}}
    result = {source_name: {}}

    for synset_id, splits in models_by_synset.items():
        # We're generating a single split file (either train OR test)
        # So we only include one split type
        if "train" in splits:
            result[source_name][synset_id] = splits["train"]
        elif "test" in splits:
            result[source_name][synset_id] = splits["test"]

    return result


def resolve_class_names(
    class_names: List[str], available_synsets: Set[str]
) -> Set[str]:
    """
    Resolve class names to synsetIds.

    Args:
        class_names: List of class names or synsetIds
        available_synsets: Set of available synsetIds in the dataset

    Returns:
        Set of synsetIds
    """
    resolved = set()

    for name in class_names:
        name_lower = name.lower().strip()

        # Check if it's already a synsetId
        if name_lower in available_synsets:
            resolved.add(name_lower)
            continue

        # Check if it's "all"
        if name_lower == "all":
            return available_synsets

        # Try to find by class name
        found = False
        for synsetId, class_name in SHAPENET_TAXONOMY.items():
            if class_name.lower() == name_lower or name_lower in class_name.lower():
                if synsetId in available_synsets:
                    resolved.add(synsetId)
                    found = True
                    break

        if not found:
            print(f"Warning: Class '{name}' not found in taxonomy or dataset")

    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Generate DeepSDF split files from ShapeNet dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--shapenet_dir",
        default="data/ShapeNetCore.v2",
        help="Path to ShapeNetCore.v2 directory (default: data/ShapeNetCore.v2)",
    )
    parser.add_argument(
        "--output_dir",
        default="examples/splits",
        help="Output directory for split files (default: examples/splits)",
    )
    parser.add_argument(
        "--classes",
        default="all",
        help="Comma-separated class names or synsetIds (e.g., 'lamp,chair,sofa') or 'all' (default: all)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Test split ratio for custom splits (default: 0.2, ignored if --use_official)",
    )
    parser.add_argument(
        "--use_official",
        action="store_true",
        default=True,
        help="Use official ShapeNet train/test splits from CSV (default: True)",
    )
    parser.add_argument(
        "--no_use_official",
        action="store_false",
        dest="use_official",
        help="Create custom splits instead of using official ones",
    )
    parser.add_argument(
        "--source_name",
        default="ShapeNetV2",
        help="Data source name in output JSON (default: ShapeNetV2)",
    )
    parser.add_argument(
        "--prefix",
        default="sv2_",
        help="Output filename prefix (default: sv2_)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for custom splits (default: 42)",
    )
    parser.add_argument(
        "--skip_verification",
        action="store_true",
        help="Skip verification that model folders exist",
    )

    args = parser.parse_args()

    # Validate arguments
    csv_path = os.path.join(args.shapenet_dir, "all.csv")
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        print(f"Please ensure --shapenet_dir points to a valid ShapeNetCore.v2 directory")
        return 1

    # Parse the ShapeNet CSV
    print(f"Parsing ShapeNet CSV: {csv_path}")
    models_by_synset = parse_shapenet_csv(csv_path)

    available_synsets = set(models_by_synset.keys())
    print(f"Found {len(available_synsets)} synsets in CSV")

    # Verify models exist if not skipped
    if not args.skip_verification:
        print("Verifying model folders exist...")
        models_by_synset = verify_models_exist(args.shapenet_dir, models_by_synset)
        available_synsets = set(models_by_synset.keys())
        print(f"Verified {len(available_synsets)} synsets with existing models")

    # Resolve class names to synsetIds
    class_names = [c.strip() for c in args.classes.split(",")]
    selected_synsets = resolve_class_names(class_names, available_synsets)

    if not selected_synsets:
        print("Error: No valid classes selected")
        return 1

    print(f"Selected {len(selected_synsets)} classes:")
    for synsetId in sorted(selected_synsets):
        class_name = SHAPENET_TAXONOMY.get(synsetId, "Unknown")
        total = sum(len(v) for v in models_by_synset[synsetId].values())
        print(f"  {synsetId} ({class_name}): {total} models")

    # Filter to selected synsets
    filtered = {k: v for k, v in models_by_synset.items() if k in selected_synsets}

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate and write split files
    if args.use_official:
        # Use official splits from CSV
        for synsetId, splits in filtered.items():
            class_name = SHAPENET_TAXONOMY.get(synsetId, synsetId)

            for split_type, model_ids in splits.items():
                if not model_ids:
                    continue

                split_data = {
                    args.source_name: {
                        synsetId: sorted(model_ids),
                    }
                }

                output_path = os.path.join(
                    args.output_dir, f"{args.prefix}{class_name}_{split_type}.json"
                )
                with open(output_path, "w") as f:
                    json.dump(split_data, f, indent=2)
                print(f"Created: {output_path} ({len(model_ids)} models)")
    else:
        # Create custom splits
        for synsetId, splits in filtered.items():
            class_name = SHAPENET_TAXONOMY.get(synsetId, synsetId)

            # Combine all models for this synset
            all_models = []
            for model_list in splits.values():
                all_models.extend(model_list)

            if not all_models:
                continue

            # Create custom split
            custom_splits = create_custom_split(
                all_models, args.test_ratio, args.seed
            )

            for split_type, model_ids in custom_splits.items():
                split_data = {
                    args.source_name: {
                        synsetId: sorted(model_ids),
                    }
                }

                output_path = os.path.join(
                    args.output_dir, f"{args.prefix}{class_name}_{split_type}.json"
                )
                with open(output_path, "w") as f:
                    json.dump(split_data, f, indent=2)
                print(f"Created: {output_path} ({len(model_ids)} models)")

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
