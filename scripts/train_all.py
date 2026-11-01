"""Train and register configured disease models, printing stored evaluation summaries."""

import argparse
from pathlib import Path
from typing import Any

from xai_dashboard.core.data.diseases import DiseaseConfig, load_disease_configs
from xai_dashboard.services.training import print_training_summary, train_and_register_models

ROOT = Path(__file__).resolve().parents[1]


def _dataset_path(config: DiseaseConfig) -> Path:
    path = Path(config.dataset_path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=ROOT / "configs")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip configured disease datasets that have not been downloaded yet.",
    )
    args = parser.parse_args()
    configs = load_disease_configs(args.configs)
    if not configs:
        parser.error(f"No disease YAML configs found in {args.configs}.")

    missing = [
        (config, _dataset_path(config))
        for config in configs.values()
        if not _dataset_path(config).is_file()
    ]
    for config, path in missing:
        print(f"{config.id}: dataset not found at {path}")
    if missing and not args.skip_missing:
        raise SystemExit(
            "No training started for missing data. Download authorized CSV files "
            "or pass --skip-missing."
        )

    available = [
        config for config in configs.values() if _dataset_path(config).is_file()
    ]
    if not available:
        raise SystemExit("No configured datasets are available to train.")

    failures: list[str] = []
    for config in available:
        dataset = _dataset_path(config)
        try:
            result: dict[str, Any] = train_and_register_models(
                config,
                dataset,
                args.artifacts,
                seed=args.seed,
                test_size=args.test_size,
                cv_splits=args.cv_splits,
                cv_repeats=args.cv_repeats,
            )
            print_training_summary(result)
        except Exception as exc:
            failures.append(config.id)
            print(f"{config.id}: training failed: {exc}")
    if failures:
        raise SystemExit(f"Training failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
