"""Profile locally available disease datasets without displaying individual records."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from xai_dashboard.core.data.diseases import DiseaseConfig, load_disease_configs
from xai_dashboard.core.data.loaders import DataSchemaError, load_dataset


def profile_dataset(config: DiseaseConfig, dataset_path: Path) -> dict[str, Any]:
    features, target = load_dataset(config, dataset_path)
    missing = {column: int(count) for column, count in features.isna().sum().items()}
    target_counts = pd.Series(target).value_counts(sort=False, dropna=False)
    return {
        "disease_id": config.id,
        "dataset_path": str(dataset_path),
        "row_count": int(len(features)),
        "feature_count": int(len(config.feature_columns)),
        "missing_by_feature": missing,
        "target_counts": {str(label): int(count) for label, count in target_counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--disease", help="Profile one configured disease; defaults to all")
    parser.add_argument("--data", type=Path, help="Override the configured path for one disease")
    args = parser.parse_args()
    configs = load_disease_configs(args.configs)
    selected = configs
    if args.disease:
        config = configs.get(args.disease)
        if config is None:
            parser.error(f"Unknown disease {args.disease!r}; configured IDs: {sorted(configs)}")
        selected = {config.id: config}
    if args.data and len(selected) != 1:
        parser.error("--data can only be used when profiling one disease")

    failures = 0
    for config in selected.values():
        path = args.data or Path(config.dataset_path)
        if not path.is_file():
            print(f"{config.id}: not profiled; dataset not found at {path}")
            continue
        try:
            print(json.dumps(profile_dataset(config, path), sort_keys=True))
        except DataSchemaError as exc:
            failures += 1
            print(f"{config.id}: schema validation failed: {exc}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
