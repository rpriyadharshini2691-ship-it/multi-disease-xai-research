"""YAML-driven disease metadata and input validation."""

from dataclasses import dataclass
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

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.categorical_columns


def load_disease_config(path: Path) -> DiseaseConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Disease config must be a YAML mapping: {path}")
    numeric = tuple(raw.get("numeric_columns", ()))
    categorical = tuple(raw.get("categorical_columns", ()))
    if not raw.get("id") or not raw.get("target_column") or not raw.get("name"):
        raise ValueError(f"Disease config is missing required identifiers: {path}")
    if len(set(numeric + categorical)) != len(numeric + categorical):
        raise ValueError(f"Feature columns must be unique: {path}")
    unconfigured_binary = set(raw.get("binary_columns", ())) - set(numeric + categorical)
    if unconfigured_binary:
        raise ValueError(
            f"Binary columns must also be numeric or categorical features: "
            f"{sorted(unconfigured_binary)} in {path}"
        )
    return DiseaseConfig(
        id=str(raw["id"]),
        name=str(raw["name"]),
        dataset_path=str(raw.get("dataset_path", "")),
        target_column=str(raw["target_column"]),
        positive_label=raw.get("positive_label", 1),
        numeric_columns=numeric,
        categorical_columns=categorical,
        binary_columns=tuple(raw.get("binary_columns", ())),
        display_names=dict(raw.get("display_names", {})),
        units=dict(raw.get("units", {})),
        valid_ranges={key: tuple(value) for key, value in raw.get("valid_ranges", {}).items()},
    )


def load_disease_configs(directory: Path) -> dict[str, DiseaseConfig]:
    paths = [path for path in sorted(directory.glob("*.yaml")) if path.stem != "app"]
    configs = {item.id: item for path in paths if (item := load_disease_config(path))}
    if len(configs) != len(paths):
        raise ValueError("Disease config IDs must be unique")
    return configs
