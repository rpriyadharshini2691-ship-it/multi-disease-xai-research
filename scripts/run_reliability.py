"""Run the specified reliability experiments for a disease model."""

import argparse
from pathlib import Path

from xai_dashboard.core.data.diseases import load_disease_config
from xai_dashboard.core.reliability.runner import run_reliability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("disease", help="Disease ID matching a YAML file in --configs")
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    config = load_disease_config(args.configs / f"{args.disease}.yaml")
    dataset = args.data or Path(config.dataset_path)
    result = run_reliability(config, dataset, args.artifacts, repetitions=args.repetitions)
    print(f"Reliability artifact written: {result['disease_id']} ({result['generated_at']})")


if __name__ == "__main__":
    main()
