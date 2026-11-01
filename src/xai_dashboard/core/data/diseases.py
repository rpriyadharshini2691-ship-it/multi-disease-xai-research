"""YAML-driven disease metadata and input validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DiseaseConfig:
    id: str
    name: str
    dataset_path: str
    target_column: str
    positive_label: Any
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    binary_columns: tuple[str, ...]
    display_names: dict[str, str]
    units: dict[str, str]
    valid_ranges: dict[str, tuple[float, float]]
    raw_dataset_path: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    target_mapping: dict[str, Any] = field(default_factory=dict)
    drop_missing_target: bool = True
    duplicate_policy: str = "keep"
    duplicate_rationale: str = ""

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.categorical_columns


def load_disease_config(path: Path) -> DiseaseConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Disease config must be a YAML mapping: {path}")
    numeric = _column_list(raw, "numeric_columns", path)
    categorical = _column_list(raw, "categorical_columns", path)
    if not raw.get("id") or not raw.get("target_column") or not raw.get("name"):
        raise ValueError(f"Disease config is missing required identifiers: {path}")
    if len(set(numeric + categorical)) != len(numeric + categorical):
        raise ValueError(f"Feature columns must be unique: {path}")
    target_column = str(raw["target_column"])
    if target_column in numeric + categorical:
        raise ValueError(f"Target column '{target_column}' cannot also be a feature: {path}")
    binary = _column_list(raw, "binary_columns", path)
    unknown_binary = set(binary) - set(numeric) - set(categorical)
    if unknown_binary:
        raise ValueError(
            f"Binary columns are not configured features: {sorted(unknown_binary)}: {path}"
        )
    valid_ranges = raw.get("valid_ranges", {})
    if not isinstance(valid_ranges, dict):
        raise ValueError(f"valid_ranges must be a mapping: {path}")
    parsed_ranges: dict[str, tuple[float, float]] = {}
    for column, bounds in valid_ranges.items():
        if column not in numeric or not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Invalid numeric valid range for '{column}': {path}")
        minimum, maximum = float(bounds[0]), float(bounds[1])
        if minimum > maximum:
            raise ValueError(f"Invalid min/max order for valid range '{column}': {path}")
        parsed_ranges[str(column)] = (minimum, maximum)
    duplicate_policy = str(raw.get("duplicate_policy", "keep"))
    if duplicate_policy not in {"keep", "drop_exact"}:
        raise ValueError(f"duplicate_policy must be 'keep' or 'drop_exact': {path}")
    target_mapping = raw.get("target_mapping", {}) or {}
    if not isinstance(target_mapping, dict):
        raise ValueError(f"target_mapping must be a mapping: {path}")
    source = raw.get("source", {}) or {}
    if not isinstance(source, dict):
        raise ValueError(f"source must be a mapping: {path}")
    return DiseaseConfig(
        id=str(raw["id"]),
        name=str(raw["name"]),
        dataset_path=str(raw.get("dataset_path", "")),
        target_column=target_column,
        positive_label=raw.get("positive_label", 1),
        numeric_columns=numeric,
        categorical_columns=categorical,
        binary_columns=binary,
        display_names=dict(raw.get("display_names", {})),
        units=dict(raw.get("units", {})),
        valid_ranges=parsed_ranges,
        raw_dataset_path=str(raw.get("raw_dataset_path", raw.get("dataset_path", ""))),
        source=dict(source),
        target_mapping=dict(target_mapping),
        drop_missing_target=bool(raw.get("drop_missing_target", True)),
        duplicate_policy=duplicate_policy,
        duplicate_rationale=str(raw.get("duplicate_rationale", "")),
    )


def _column_list(raw: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be a list of non-empty column names: {path}")
    if len(set(value)) != len(value):
        raise ValueError(f"{key} contains duplicate column names: {path}")
    return tuple(value)


def load_disease_configs(directory: Path) -> dict[str, DiseaseConfig]:
    paths = [path for path in sorted(directory.glob("*.yaml")) if path.stem != "app"]
    configs = {item.id: item for path in paths if (item := load_disease_config(path))}
    if len(configs) != len(paths):
        raise ValueError("Disease config IDs must be unique")
    return configs
