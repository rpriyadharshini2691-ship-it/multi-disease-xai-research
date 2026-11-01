"""Run reproducible Phase 3 reliability experiments for one or all diseases."""

import argparse
from pathlib import Path

from xai_dashboard.core.data.diseases import load_disease_config
from xai_dashboard.core.reliability.runner import (
    DEFAULT_RANDOM_SEEDS,
    DEFAULT_REPETITIONS,
    DEFAULT_SAMPLE_SIZE,
    run_reliability,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("disease", choices=("all", "diabetes", "heart", "ckd"))
    parser.add_argument("--configs", type=Path, default=ROOT / "configs")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_RANDOM_SEEDS)
    )
    args = parser.parse_args()
    diseases = ("diabetes", "heart", "ckd") if args.disease == "all" else (args.disease,)
    for disease_id in diseases:
        config = load_disease_config(args.configs / f"{disease_id}.yaml")
        dataset = args.data or Path(config.dataset_path)
        if not dataset.is_absolute():
            dataset = ROOT / dataset
        result = run_reliability(
            config,
            dataset,
            args.artifacts,
            repetitions=args.repetitions,
            sample_size=args.sample_size,
            random_seeds=args.seeds,
        )
        print(
            f"{disease_id}: {len(result['sample_results'])} sample comparisons, "
            f"artifact={result['artifact_path']}, warnings={len(result['warnings'])}"
        )


if __name__ == "__main__":
    main()
