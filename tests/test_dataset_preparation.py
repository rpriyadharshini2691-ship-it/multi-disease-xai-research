"""Phase 2: dataset preparation, target mapping, duplicate handling and leakage safeguards."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.core.data.loader import DatasetSchemaError, stratified_split
from xai_dashboard.core.data.quality import (
    apply_target_mapping,
    file_sha256,
    prepare_dataset,
    save_report,
    write_prepared,
)


def _config(**overrides: object) -> DiseaseConfig:
    values: dict[str, object] = {
        "id": "toy",
        "name": "Toy",
        "dataset_path": "",
        "target_column": "target",
        "positive_label": 1,
        "numeric_columns": ("a",),
        "categorical_columns": ("b",),
        "binary_columns": (),
        "display_names": {},
        "units": {},
        "valid_ranges": {},
        "source": {"name": "toy", "uci_id": 0},
    }
    values.update(overrides)
    return DiseaseConfig(**values)  # type: ignore[arg-type]


def _write(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)
    return path


def test_heart_style_mapping_zero_stays_negative_and_one_to_four_positive(tmp_path) -> None:
    raw = pd.DataFrame(
        {"a": range(7), "b": list("xyxyxyx"), "target": [0, 1, 2, 3, 4, 0, None]}
    )
    config = _config(target_mapping={"type": "positive_if_greater_than", "threshold": 0})
    prepared = prepare_dataset(config, _write(tmp_path, raw))
    assert prepared.frame["target"].tolist() == [0, 1, 1, 1, 1, 0]
    assert prepared.report["target_labels_before_mapping"]["4.0"] == 1
    operations = {op["operation"]: op for op in prepared.report["cleaning_operations"]}
    assert operations["drop_rows_with_missing_target"]["rows_affected"] == 1


def test_missing_targets_are_not_imputed_as_negative() -> None:
    mapped, _ = apply_target_mapping(
        pd.Series([0, 3, None]), {"type": "positive_if_greater_than", "threshold": 0}
    )
    assert mapped.isna().tolist() == [False, False, True]


def test_ckd_label_map_and_unknown_label_rejected() -> None:
    mapped, _ = apply_target_mapping(
        pd.Series(["ckd", "notckd", " ckd\t", None]),
        {"type": "label_map", "labels": {"ckd": 1, "notckd": 0}},
    )
    assert mapped.tolist()[:3] == [1, 0, 1] and pd.isna(mapped.iloc[3])
    with pytest.raises(DatasetSchemaError, match="outside the configured map"):
        apply_target_mapping(
            pd.Series(["ckd", "maybe"]), {"type": "label_map", "labels": {"ckd": 1, "notckd": 0}}
        )


def test_duplicates_dropped_only_when_policy_says_and_report_is_complete(tmp_path) -> None:
    raw = pd.DataFrame(
        {"a": [1, 1, 2, 3, 3, 3], "b": list("xxyzzz"), "target": [1, 1, 0, 1, 1, 1]}
    )
    path = _write(tmp_path, raw)
    kept = prepare_dataset(_config(duplicate_policy="keep"), path)
    dropped = prepare_dataset(_config(duplicate_policy="drop_exact"), path)
    assert len(kept.frame) == 6 and len(dropped.frame) == 3
    report = dropped.report
    for key in (
        "source", "original_rows", "original_columns", "original_missing_cells",
        "target_labels_before_mapping", "target_labels_after_mapping",
        "class_distribution_before_cleaning", "class_distribution_final",
        "duplicate_analysis", "cleaning_operations", "final_rows", "final_feature_count",
        "raw_sha256",
    ):  # fmt: skip
        assert key in report
    assert report["duplicate_analysis"]["exact_duplicate_rows"] == 3
    assert report["final_rows"] == 3 and report["final_feature_count"] == 2
    assert report["class_distribution_before_cleaning"] == {"1": 5, "0": 1}


def test_raw_file_is_not_modified_and_cleaned_copy_is_separate(tmp_path) -> None:
    raw = pd.DataFrame({"a": [1, 1, 2], "b": list("xxy"), "target": [1, 1, 0]})
    path = _write(tmp_path, raw)
    before = file_sha256(path)
    config = _config(duplicate_policy="drop_exact")
    prepared = prepare_dataset(config, path)
    output = tmp_path / "processed" / "toy.csv"
    report = write_prepared(config, prepared, output)
    json_path, md_path = save_report(report, tmp_path / "reports")
    assert file_sha256(path) == before
    assert output.is_file() and len(pd.read_csv(output)) == 2
    assert json_path.is_file() and md_path.is_file()
    assert report["processed_sha256"] == file_sha256(output)


def test_duplicates_removed_before_split_leave_no_train_test_overlap(tmp_path) -> None:
    rng = np.random.default_rng(0)
    feature_a = rng.integers(0, 6, 80)
    feature_b = rng.choice(list("xy"), 80)
    base = pd.DataFrame(
        {
            "a": feature_a,
            "b": feature_b,
            "target": (feature_a + (feature_b == "x")) % 2,
        }
    )
    path = _write(tmp_path, pd.concat([base, base.iloc[:40]], ignore_index=True))
    config = _config(duplicate_policy="drop_exact")
    frame = prepare_dataset(config, path).frame
    split = stratified_split(frame, config, test_size=0.25, seed=3)
    train_keys = set(map(tuple, split.x_train.astype(str).values))
    test_keys = set(map(tuple, split.x_test.astype(str).values))
    assert not train_keys & test_keys


def test_real_configs_declare_source_mapping_and_policy() -> None:
    from xai_dashboard.core.data.diseases import load_disease_config

    for disease, uci_id in (("diabetes", 529), ("heart", 45), ("ckd", 336)):
        config = load_disease_config(Path("configs") / f"{disease}.yaml")
        assert config.source["uci_id"] == uci_id
        assert config.dataset_path.startswith("data/processed/")
        assert config.raw_dataset_path == f"data/{disease}.csv"
        assert config.target_mapping
    diabetes = load_disease_config(Path("configs/diabetes.yaml"))
    assert diabetes.duplicate_policy == "drop_exact" and diabetes.duplicate_rationale
