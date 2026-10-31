"""Download a dataset only from a URL explicitly supplied by the user."""

import argparse
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from xai_dashboard.core.data.diseases import load_disease_configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Dataset URL authorized for your intended use")
    parser.add_argument("--disease", help="Configured disease ID; uses its dataset_path")
    parser.add_argument("--output", type=Path, help="Output path for a user-selected dataset URL")
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    args = parser.parse_args()
    configs = load_disease_configs(args.configs)

    if not args.url:
        if args.disease or args.output:
            parser.error("--url is required when --disease or --output is provided")
        print("No dataset downloaded. Supply an explicitly authorized --url to download.")
        for config in configs.values():
            print(f"{config.id}: expected local dataset path {config.dataset_path}")
        return

    if args.disease:
        config = configs.get(args.disease)
        if config is None:
            parser.error(f"Unknown disease {args.disease!r}; configured IDs: {sorted(configs)}")
        output = args.output or Path(config.dataset_path)
    elif args.output:
        output = args.output
    else:
        parser.error("provide --output or a configured --disease when using --url")

    if urlparse(args.url).scheme not in {"http", "https"}:
        parser.error("--url must use http or https")
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(args.url, timeout=60) as response, output.open("wb") as destination:
        destination.write(response.read())
    print(f"Downloaded user-selected dataset to {output}")


if __name__ == "__main__":
    main()
