# Data quality report: Heart Disease

Generated: 2026-11-01T08:37:28.681455+00:00

## Source
- Source: Heart Disease (Cleveland subset) (UCI dataset ID 45)
- URL: https://archive.ics.uci.edu/dataset/45/heart+disease
- Verification status: **verified**

## Sizes
- Original: 303 rows x 14 columns
- Final: 303 rows, 13 features
- Original missing cells: 6 ({'ca': 4, 'thal': 2})
- Final missing cells: 6

## Target
- Labels before mapping: {'0': 164, '1': 139}
- Labels after mapping: {'0': 164, '1': 139}
- Class distribution before cleaning: {'0': 164, '1': 139}
- Class distribution final: {'0': 164, '1': 139}

## Duplicates
- Exact duplicate rows: 0
- Feature groups with conflicting labels: 0
- Policy: keep
- Rationale: No exact duplicate rows exist in the Cleveland data.

## Cleaning operations
- map_target: rows affected = n/a
- drop_rows_with_missing_target: rows affected = 0
- duplicate_handling: rows affected = 0

## Verification evidence
- reference: UCI 45 processed.cleveland.data (Cleveland subset only)
- reference_rows: 303
- reference_raw_target_distribution: {'0': 164, '1': 55, '2': 36, '3': 35, '4': 13}
- rows_identical_after_mapping_num_gt_0_to_1: True
- note: The local file already stores the binary target; the 0-4 mapping is reproduced from the official raw file. Verification is by cell-level comparison, not row count.

## Fingerprints
- Raw SHA-256: 2c47d9ee8b2e0d30e81c113a079baec0bb3814b1cef093e675610de45df54ffc
- Processed SHA-256: 2c47d9ee8b2e0d30e81c113a079baec0bb3814b1cef093e675610de45df54ffc
