"""Dataset preparation: target mapping, missing-target handling, duplicate policy, reporting.

Raw files are never modified. A cleaned copy is written separately and every operation is
recorded in a per-disease data-quality report. Duplicate handling happens here, before any
train/test split, so identical rows cannot appear on both sides of the split.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.core.data.loader import DatasetSchemaError


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    report: dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def apply_target_mapping(
    series: pd.Series, mapping: dict[str, Any]
) -> tuple[pd.Series, dict[str, Any]]:
    """Map a raw target to {0, 1}; missing targets stay missing (<NA>), never imputed."""
    if not mapping:
        return series, {"type": "none"}
    kind = mapping.get("type")
    if kind == "positive_if_greater_than":
        threshold = float(mapping.get("threshold", 0))
        numeric = pd.to_numeric(series, errors="coerce")
        unparsable = series.notna() & numeric.isna()
        if unparsable.any():
            examples = series[unparsable].astype(str).drop_duplicates().head(5).tolist()
            raise DatasetSchemaError(f"Target has non-numeric values: {examples}")
        mapped = (numeric > threshold).astype("Int64").where(numeric.notna(), pd.NA)
        return mapped, {"type": kind, "rule": f"value > {threshold:g} -> 1, otherwise 0"}
    if kind == "label_map":
        labels = {str(key): int(value) for key, value in dict(mapping["labels"]).items()}
        passthrough = {str(value) for value in labels.values()}
        cleaned = series.astype("string").str.strip()
        unknown = cleaned.notna() & ~cleaned.isin(set(labels) | passthrough)
        if unknown.any():
            examples = cleaned[unknown].drop_duplicates().head(5).tolist()
            raise DatasetSchemaError(
                f"Target contains labels outside the configured map {sorted(labels)}: {examples}"
            )
        resolved = cleaned.map(lambda value: pd.NA if pd.isna(value) else
                               labels.get(value, int(value) if value in passthrough else pd.NA))
        return resolved.astype("Int64"), {"type": kind, "labels": labels}
    raise DatasetSchemaError(f"Unsupported target_mapping type: {kind!r}")


def prepare_dataset(config: DiseaseConfig, raw_path: Path) -> PreparedDataset:
    """Return the cleaned frame (features + 0/1 target) and its data-quality report."""
    raw = pd.read_csv(raw_path, na_values=["?"])
    required = list(config.feature_columns) + [config.target_column]
    missing_columns = sorted(set(required) - set(raw.columns))
    if missing_columns:
        raise DatasetSchemaError(f"Raw dataset is missing required columns: {missing_columns}")
    operations: list[dict[str, Any]] = []

    frame = raw.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype("string").str.strip()
    extra = [column for column in frame.columns if column not in required]
    if extra:
        frame = frame.drop(columns=extra)
        operations.append({"operation": "drop_unconfigured_columns", "detail": extra})

    labels_before = _counts(frame[config.target_column])
    mapped, mapping_detail = apply_target_mapping(
        frame[config.target_column], config.target_mapping
    )
    frame[config.target_column] = mapped
    labels_after = _counts(frame[config.target_column])
    operations.append(
        {
            "operation": "map_target",
            "detail": mapping_detail,
            "labels_before": labels_before,
            "labels_after": labels_after,
        }
    )
    class_before = _counts(frame[config.target_column].dropna())

    missing_target = int(frame[config.target_column].isna().sum())
    if missing_target and config.drop_missing_target:
        frame = frame.loc[frame[config.target_column].notna()].copy()
    operations.append(
        {
            "operation": "drop_rows_with_missing_target",
            "rows_affected": missing_target if config.drop_missing_target else 0,
            "detail": "Missing targets are never imputed; rows are dropped before training.",
        }
    )
    frame[config.target_column] = frame[config.target_column].astype("int64")

    feature_columns = list(config.feature_columns)
    duplicate_rows = int(frame.duplicated().sum())
    groups = frame.groupby(feature_columns, dropna=False)[config.target_column].agg(
        ["size", "nunique"]
    )
    duplicate_analysis = {
        "exact_duplicate_rows": duplicate_rows,
        "distinct_rows": int(len(frame) - duplicate_rows),
        "feature_groups_with_repeats": int((groups["size"] > 1).sum()),
        "largest_repeat_group": int(groups["size"].max()) if len(groups) else 0,
        "feature_groups_with_conflicting_labels": int((groups["nunique"] > 1).sum()),
        "policy": config.duplicate_policy,
        "rationale": config.duplicate_rationale,
    }
    if config.duplicate_policy == "drop_exact" and duplicate_rows:
        frame = frame.drop_duplicates(keep="first").copy()
    operations.append(
        {
            "operation": "duplicate_handling",
            "rows_affected": duplicate_rows if config.duplicate_policy == "drop_exact" else 0,
            "detail": duplicate_analysis,
        }
    )
    frame = frame.reset_index(drop=True)

    report: dict[str, Any] = {
        "disease_id": config.id,
        "dataset_name": config.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": config.source,
        "raw_file": str(raw_path).replace("\\", "/"),
        "raw_sha256": file_sha256(raw_path),
        "original_rows": int(len(raw)),
        "original_columns": int(raw.shape[1]),
        "original_duplicate_rows": int(raw.duplicated().sum()),
        "original_missing_cells": int(raw.isna().sum().sum()),
        "original_missing_by_column": {
            str(key): int(value) for key, value in raw.isna().sum().items() if value
        },
        "target_column": config.target_column,
        "target_labels_before_mapping": labels_before,
        "target_labels_after_mapping": labels_after,
        "class_distribution_before_cleaning": class_before,
        "class_distribution_final": _counts(frame[config.target_column]),
        "duplicate_analysis": duplicate_analysis,
        "cleaning_operations": operations,
        "final_rows": int(len(frame)),
        "final_columns": int(frame.shape[1]),
        "final_feature_count": len(feature_columns),
        "final_missing_cells": int(frame.isna().sum().sum()),
        "missing_values_note": (
            "Remaining missing feature values are imputed inside the sklearn pipelines, "
            "fitted on training rows only."
        ),
    }
    return PreparedDataset(frame=frame, report=report)


def write_prepared(
    config: DiseaseConfig, prepared: PreparedDataset, output_path: Path
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.frame.to_csv(output_path, index=False)
    prepared.report["processed_file"] = str(output_path).replace("\\", "/")
    prepared.report["processed_sha256"] = file_sha256(output_path)
    return prepared.report


def render_markdown(report: dict[str, Any]) -> str:
    source = report.get("source", {})
    verification = report.get("verification", {})
    duplicate = report["duplicate_analysis"]
    lines = [
        f"# Data quality report: {report['dataset_name']}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Source",
        f"- Source: {source.get('name', 'unspecified')} "
        f"(UCI dataset ID {source.get('uci_id', '?')})",
        f"- URL: {source.get('url', 'unspecified')}",
        f"- Verification status: **{verification.get('status', 'not_verified')}**",
        "",
        "## Sizes",
        f"- Original: {report['original_rows']} rows x {report['original_columns']} columns",
        f"- Final: {report['final_rows']} rows, {report['final_feature_count']} features",
        f"- Original missing cells: {report['original_missing_cells']} "
        f"({report['original_missing_by_column'] or 'none'})",
        f"- Final missing cells: {report['final_missing_cells']}",
        "",
        "## Target",
        f"- Labels before mapping: {report['target_labels_before_mapping']}",
        f"- Labels after mapping: {report['target_labels_after_mapping']}",
        f"- Class distribution before cleaning: {report['class_distribution_before_cleaning']}",
        f"- Class distribution final: {report['class_distribution_final']}",
        "",
        "## Duplicates",
        f"- Exact duplicate rows: {duplicate['exact_duplicate_rows']}",
        f"- Feature groups with conflicting labels: "
        f"{duplicate['feature_groups_with_conflicting_labels']}",
        f"- Policy: {duplicate['policy']}",
        f"- Rationale: {duplicate['rationale'] or 'n/a'}",
        "",
        "## Cleaning operations",
    ]
    for operation in report["cleaning_operations"]:
        lines.append(
            f"- {operation['operation']}: rows affected = {operation.get('rows_affected', 'n/a')}"
        )
    if verification:
        lines += ["", "## Verification evidence"]
        lines += [f"- {key}: {value}" for key, value in verification.items() if key != "status"]
    lines += [
        "",
        "## Fingerprints",
        f"- Raw SHA-256: {report['raw_sha256']}",
        f"- Processed SHA-256: {report.get('processed_sha256', 'n/a')}",
        "",
    ]
    return "\n".join(lines)


def save_report(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"data_quality_{report['disease_id']}.json"
    md_path = reports_dir / f"data_quality_{report['disease_id']}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
