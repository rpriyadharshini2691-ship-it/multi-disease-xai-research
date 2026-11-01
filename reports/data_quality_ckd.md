# Data quality report: Chronic Kidney Disease

Generated: 2026-11-01T08:37:28.751260+00:00

## Source
- Source: Chronic Kidney Disease (UCI dataset ID 336)
- URL: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- Verification status: **verified_with_notes**

## Sizes
- Original: 400 rows x 25 columns
- Final: 400 rows, 24 features
- Original missing cells: 1012 ({'age': 9, 'bp': 12, 'sg': 47, 'al': 46, 'su': 49, 'rbc': 152, 'pc': 65, 'pcc': 4, 'ba': 4, 'bgr': 44, 'bu': 19, 'sc': 17, 'sod': 87, 'pot': 88, 'hemo': 52, 'pcv': 71, 'wbcc': 106, 'rbcc': 131, 'htn': 2, 'dm': 2, 'cad': 2, 'appet': 1, 'pe': 1, 'ane': 1})
- Final missing cells: 1012

## Target
- Labels before mapping: {'1': 250, '0': 150}
- Labels after mapping: {'1': 250, '0': 150}
- Class distribution before cleaning: {'1': 250, '0': 150}
- Class distribution final: {'1': 250, '0': 150}

## Duplicates
- Exact duplicate rows: 0
- Feature groups with conflicting labels: 0
- Policy: keep
- Rationale: No exact duplicate rows exist in the CKD data.

## Cleaning operations
- map_target: rows affected = n/a
- drop_rows_with_missing_target: rows affected = 0
- duplicate_handling: rows affected = 0

## Verification evidence
- reference: UCI 336 chronic_kidney_disease.arff (the file ships with 400 rows)
- reference_rows: 400
- reference_raw_class_labels: {'ckd': 250, 'notckd': 149, 'no': 1}
- columns_match_after_renaming: True
- differing_row_indices_zero_based: [369]
- note: Data row 369 (0-based) is malformed in the official ARFF: it has an extra empty field that shifts dm/cad/appet/pe/ane/class by one position. The local CSV holds a realigned version of that row; every other cell matches the official file.
- column_renames: {'wc': 'wbcc', 'rc': 'rbcc', 'class': 'target'}

## Fingerprints
- Raw SHA-256: 89e386a897c22bb5c42ff976917977bee61ae1426f6e0d3e4ff344a1b4a09b90
- Processed SHA-256: 89e386a897c22bb5c42ff976917977bee61ae1426f6e0d3e4ff344a1b4a09b90
