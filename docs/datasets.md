# Dataset Provenance and Cleaning

The local CSVs and reference copies are verified against UCI records and are not redistributed by this project. The references below describe the exact cohorts and transformations used by the current configs; the generated reports include row counts, fingerprints, missingness, duplicate analysis, and cell-level verification evidence.

| Disease | Verified UCI source | DOI | Raw rows / features |
| --- | --- | --- | --- |
| Diabetes | [UCI 529: Early Stage Diabetes Risk Prediction](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset) | [10.24432/C5VG8H](https://doi.org/10.24432/C5VG8H) | 520 / 16 |
| Heart | [UCI 45: Heart Disease, Cleveland cohort](https://archive.ics.uci.edu/dataset/45/heart+disease) | [10.24432/C52P4X](https://doi.org/10.24432/C52P4X) | 303 / 13 |
| CKD | [UCI 336: Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) | [10.24432/C5G020](https://doi.org/10.24432/C5G020) | 400 / 24 |

All three UCI records list CC BY 4.0. Attribute the original dataset authors when publishing results, and recheck the UCI record and institutional requirements before reuse.

## Diabetes (UCI 529)

The local 520-row CSV matches UCI's `diabetes_data_upload.csv` cell-for-cell and in row order after normalizing column names and mapping `Positive` to 1 and `Negative` to 0. The official file contains 269 exact repeated rows, leaving 251 distinct full rows. There are 107 feature groups with repeats, the largest group occurs 16 times, and no repeated feature group has conflicting labels.

The raw source is preserved unchanged. The configured training copy uses `drop_exact`: exact copies are not treated as independent observations because they can otherwise cross the train/test boundary and inflate holdout results. This is a conservative evaluation choice, not a claim that the source repetitions are accidental; their origin cannot be determined from the file. The decision and counts are recorded in the diabetes quality report.

## Heart (UCI 45)

The local 303-row file matches UCI's `processed.cleveland.data` cohort only, not the combined Cleveland, Hungary, Switzerland, and VA cohorts. The official target `num` is mapped as 0 -> 0 and 1-4 -> 1; the source distribution is 164, 55, 36, 35, and 13 records respectively. The local CSV stores the already-mapped binary target. It has six missing feature cells and no exact duplicate rows, so no rows are removed.

## Chronic Kidney Disease (UCI 336)

The local file has the 400 UCI records and 24 configured features. The target mapping is `ckd` -> 1 and `notckd` -> 0; the local distribution is 250 positive and 150 negative records. The official ARFF's zero-based row 369 has 26 comma-separated fields where the schema has 25: an empty field shifts values after `htn`, and the parsed class becomes `no`. The local CSV preserves the intended alignment (`dm=no`, `cad=no`, `appet=good`, `pe=no`, `ane=no`) and the trailing `notckd` class. This single-record repair is documented in the report; no records are dropped. The reference ARFF remains unchanged.

There are 1,012 missing feature cells and no exact duplicate rows. Missing feature values are imputed within model pipelines fitted on training data only.

## Target Mapping and Reports

The mappings are declared in `configs/diabetes.yaml`, `configs/heart.yaml`, and `configs/ckd.yaml`. `scripts/prepare_data.py` compares each raw CSV with `data/reference_uci/`, writes a separate `data/processed/` copy, and regenerates the JSON and Markdown reports under `reports/`. Raw CSVs are never modified by that process. The reports record SHA-256 fingerprints so a future input change can be distinguished from this verified state.

The processed Diabetes set has 251 rows; Heart has 303; CKD has 400. Performance generated from the deduplicated Diabetes set is a new experiment and should not be compared directly with results trained on all 520 repeated rows.

## Data Handling

The application does not download datasets or place input features in prediction logs. A prediction score is logged only when an authenticated user explicitly consents. Keep downloaded research data local, follow the UCI license and attribution terms, and observe institutional ethics requirements.
