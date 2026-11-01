"""Download only an explicitly selected dataset URL or show current local data status."""

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from xai_dashboard.core.data.diseases import DiseaseConfig, load_disease_configs
from xai_dashboard.core.data.loader import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def _download(url: str, output: Path, config: DiseaseConfig) -> None:
    scheme = urlparse(url).scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("Dataset URL must use http or https.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.part")
    try:
        request = Request(url, headers={"User-Agent": "multi-disease-xai-research/0.1"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
        load_dataset(config, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", help="Disease ID from the YAML configs")
    parser.add_argument("--url", help="Dataset URL authorized for your intended use")
    parser.add_argument(
        "--output", type=Path, help="Destination CSV path; defaults to config dataset_path"
    )
    parser.add_argument("--configs", type=Path, default=ROOT / "configs")
    args = parser.parse_args()
    configs = load_disease_configs(args.configs)
    if not args.url:
        print("No download requested; provide --disease and an authorized --url to download data.")
        for config in configs.values():
            path = ROOT / config.dataset_path
            status = "available" if path.is_file() else "not downloaded"
            print(f"{config.id}: {status} ({path})")
        return
    if not args.disease:
        parser.error("--disease is required when --url is provided")
    config = configs.get(args.disease)
    if config is None:
        parser.error(f"Unknown disease '{args.disease}'. Configure it in {args.configs}.")
    output = args.output or ROOT / config.dataset_path
    _download(args.url, output, config)
    print(f"Downloaded the user-selected dataset for {config.id} to {output}")


if __name__ == "__main__":
    main()
