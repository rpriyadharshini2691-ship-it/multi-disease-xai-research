"""Train configured disease models and persist evaluation artifacts."""

import argparse
from pathlib import Path

from xai_dashboard.core.data.diseases import load_disease_config
from xai_dashboard.services.training import print_training_summary, train_and_register_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("disease", help="Disease ID matching a YAML file in --configs")
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    config = load_disease_config(args.configs / f"{args.disease}.yaml")
    dataset = args.data or Path(config.dataset_path)
    result = train_and_register_models(config, dataset, args.artifacts)
    print_training_summary(result)


if __name__ == "__main__":
    main()
