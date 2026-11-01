"""Verify raw datasets against UCI, write cleaned copies to data/processed and quality reports.

Usage: python scripts/prepare_data.py [--reference-dir data/reference_uci] [--disease ID ...]
Raw files in data/ are never modified.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai_dashboard.core.data.diseases import load_disease_config  # noqa: E402
from xai_dashboard.core.data.quality import (  # noqa: E402
    prepare_dataset,
    save_report,
    write_prepared,
)
from xai_dashboard.core.data.verification import (  # noqa: E402
    verify_ckd,
    verify_diabetes,
    verify_heart,
)

VERIFIERS = {
    "diabetes": ("529", verify_diabetes),
    "heart": ("45", verify_heart),
    "ckd": ("336", verify_ckd),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data" / "reference_uci")
    parser.add_argument("--disease", nargs="*", default=list(VERIFIERS))
    args = parser.parse_args()
    for disease_id in args.disease:
        config = load_disease_config(ROOT / "configs" / f"{disease_id}.yaml")
        raw_path = ROOT / config.raw_dataset_path
        prepared = prepare_dataset(config, raw_path)
        uci_id, verifier = VERIFIERS[disease_id]
        reference = args.reference_dir / uci_id
        if reference.is_dir():
            prepared.report["verification"] = verifier(raw_path, reference)
        else:
            prepared.report["verification"] = {
                "status": "unverified",
                "note": (
                    f"Reference files not found at {reference}. Download UCI dataset {uci_id} "
                    f"from {config.source.get('url')} and re-run."
                ),
            }
        report = write_prepared(config, prepared, ROOT / config.dataset_path)
        json_path, _ = save_report(report, ROOT / "reports")
        print(
            f"{disease_id}: {report['original_rows']} -> {report['final_rows']} rows, "
            f"verification={report['verification']['status']}, report={json_path.name}"
        )


if __name__ == "__main__":
    main()
