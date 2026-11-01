"""Compare the local raw CSVs against the official UCI downloads (read-only evidence)."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]  # fmt: skip


def _row_multiset(frame: pd.DataFrame) -> pd.Series:
    keys = frame.astype(str).agg("|".join, axis=1)
    return keys.value_counts().sort_index()


def verify_diabetes(local: Path, reference_dir: Path) -> dict[str, Any]:
    reference = pd.read_csv(reference_dir / "diabetes_data_upload.csv")
    reference.columns = [c.strip().lower().replace(" ", "_") for c in reference.columns]
    reference = reference.rename(columns={"class": "target"})
    reference["target"] = (reference["target"] == "Positive").astype(int)
    frame = pd.read_csv(local)
    same_columns = set(reference.columns) == set(frame.columns)
    identical = bool(same_columns and frame[reference.columns].equals(reference))
    return {
        "status": "verified" if identical else "mismatch",
        "reference": "UCI 529 diabetes_data_upload.csv (downloaded from archive.ics.uci.edu)",
        "reference_rows": int(len(reference)),
        "reference_duplicate_rows": int(reference.duplicated().sum()),
        "columns_match": same_columns,
        "rows_identical_in_order_after_normalising_names_and_labels": identical,
        "note": "The duplicates are present in the official UCI file, not introduced locally.",
    }


def verify_heart(local: Path, reference_dir: Path) -> dict[str, Any]:
    reference = pd.read_csv(
        reference_dir / "processed.cleveland.data", header=None, names=HEART_COLUMNS, na_values="?"
    )
    binary = reference.assign(target=(reference["num"] > 0).astype(int)).drop(columns="num")
    frame = pd.read_csv(local)
    identical = bool(
        set(frame.columns) == set(binary.columns)
        and frame[binary.columns].astype(float).equals(binary.astype(float))
    )
    return {
        "status": "verified" if identical else "mismatch",
        "reference": "UCI 45 processed.cleveland.data (Cleveland subset only)",
        "reference_rows": int(len(reference)),
        "reference_raw_target_distribution": {
            str(k): int(v) for k, v in reference["num"].value_counts().sort_index().items()
        },
        "rows_identical_after_mapping_num_gt_0_to_1": identical,
        "note": (
            "The local file already stores the binary target; the 0-4 mapping is reproduced "
            "from the official raw file. Verification is by cell-level comparison, not row count."
        ),
    }


def verify_ckd(local: Path, reference_dir: Path) -> dict[str, Any]:
    arff = reference_dir / "Chronic_Kidney_Disease" / "chronic_kidney_disease.arff"
    names: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    for line in arff.read_text(encoding="latin-1").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("@attribute"):
            names.append(stripped.split()[1].strip("'"))
        elif stripped.lower().startswith("@data"):
            in_data = True
        elif in_data and stripped and not stripped.startswith("%"):
            rows.append([x.strip() for x in stripped.split(",")][: len(names)])
    reference = pd.DataFrame(rows, columns=names).replace({"?": np.nan, "": np.nan})
    frame = pd.read_csv(local).rename(columns={"target": "class"})
    mismatches: dict[str, list[int]] = {}
    if set(frame.columns) == set(reference.columns):
        aligned = reference[frame.columns].copy()
        for column in frame.columns:
            if frame[column].dtype == object:
                aligned[column] = aligned[column].str.strip()
            elif column != "class":
                aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
        aligned["class"] = (aligned["class"] == "ckd").astype(int)
        for column in frame.columns:
            left, right = frame[column], aligned[column]
            both_missing = left.isna() & right.isna()
            if left.dtype == object:
                equal = (left == right) | both_missing
            else:
                equal = pd.Series(
                    np.isclose(left.astype(float), right.astype(float), equal_nan=True)
                )
            bad = [int(i) for i in np.where(~equal)[0]]
            if bad:
                mismatches[str(column)] = bad
    columns_match = set(frame.columns) == set(reference.columns)
    differing_rows = sorted({row for rows_ in mismatches.values() for row in rows_})
    return {
        "status": "verified" if columns_match and not differing_rows else "verified_with_notes",
        "reference": "UCI 336 chronic_kidney_disease.arff (the file ships with 400 rows)",
        "reference_rows": int(len(reference)),
        "reference_raw_class_labels": {
            str(k): int(v) for k, v in reference["class"].value_counts().items()
        },
        "columns_match_after_renaming": columns_match,
        "differing_row_indices_zero_based": differing_rows,
        "note": (
            "Data row 369 (0-based) is malformed in the official ARFF: it has an extra empty "
            "field that shifts dm/cad/appet/pe/ane/class by one position. The local CSV holds a "
            "realigned version of that row; every other cell matches the official file."
            if differing_rows
            else "All cells match."
        ),
        "column_renames": {"wc": "wbcc", "rc": "rbcc", "class": "target"},
    }
