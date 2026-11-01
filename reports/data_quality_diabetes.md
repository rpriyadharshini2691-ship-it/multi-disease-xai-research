# Data quality report: Diabetes

Generated: 2026-11-01T08:37:28.618289+00:00

## Source
- Source: Early Stage Diabetes Risk Prediction Dataset (UCI dataset ID 529)
- URL: https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
- Verification status: **verified**

## Sizes
- Original: 520 rows x 17 columns
- Final: 251 rows, 16 features
- Original missing cells: 0 (none)
- Final missing cells: 0

## Target
- Labels before mapping: {'1': 320, '0': 200}
- Labels after mapping: {'1': 320, '0': 200}
- Class distribution before cleaning: {'1': 320, '0': 200}
- Class distribution final: {'1': 173, '0': 78}

## Duplicates
- Exact duplicate rows: 269
- Feature groups with conflicting labels: 0
- Policy: drop_exact
- Rationale: The official UCI file itself contains 269 exact repeats (251 distinct rows), no identical feature vector has conflicting labels, and repeats are scattered rather than adjacent, so they cannot be proven accidental or genuine. Exact copies would otherwise fall on both sides of a train/test split and inflate hold-out metrics, so exact duplicates are removed before splitting. The raw file is kept unchanged.

## Cleaning operations
- map_target: rows affected = n/a
- drop_rows_with_missing_target: rows affected = 0
- duplicate_handling: rows affected = 269

## Verification evidence
- reference: UCI 529 diabetes_data_upload.csv (downloaded from archive.ics.uci.edu)
- reference_rows: 520
- reference_duplicate_rows: 269
- columns_match: True
- rows_identical_in_order_after_normalising_names_and_labels: True
- note: The duplicates are present in the official UCI file, not introduced locally.

## Fingerprints
- Raw SHA-256: ddeb4f140afd7add362ae63ea5f678ad28f37667254481ff9897587e93c58b9b
- Processed SHA-256: f84bfc7cb336e14593eded89935423b5f349eb0a3f5f0615aba85c832d5bee5d
