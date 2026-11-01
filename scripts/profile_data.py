"""Profile configured local disease datasets without inventing model metrics."""

import argparse
import json
from pathlib import Path
from typing import Any

from xai_dashboard.core.data.diseases import DiseaseConfig, load_disease_configs
from xai_dashboard.core.data.loader import DatasetSchemaError, load_dataset, profile_dataset

ROOT = Path(__file__).resolve().parents[1]


def profile_config(config: DiseaseConfig) -> dict[str, Any]:
    path = Path(config.dataset_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return {
            "disease_id": config.id,
            "status": "not_downloaded",
            "dataset_path": str(path),
        }
    try:
        frame = load_dataset(config, path)
    except (OSError, ValueError, DatasetSchemaError) as exc:
        return {
            "disease_id": config.id,
            "status": "invalid",
            "dataset_path": str(path),
            "error": str(exc),
        }
    return {"status": "ready", "dataset_path": str(path), **profile_dataset(frame, config)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", help="Profile one configured disease; otherwise profile all.")
    parser.add_argument("--configs", type=Path, default=ROOT / "configs")
    args = parser.parse_args()
    configs = load_disease_configs(args.configs)
    if args.disease:
        config = configs.get(args.disease)
        if config is None:
            parser.error(f"Unknown disease '{args.disease}'. Configure it in {args.configs}.")
        results = [profile_config(config)]
    else:
        results = [profile_config(config) for config in configs.values()]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
