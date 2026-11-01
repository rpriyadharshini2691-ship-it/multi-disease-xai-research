# Dataset setup

No health datasets are included in this repository. Obtain each dataset directly from its source, review the source's current license and conditions, and place a CSV matching the configured column names in this directory (or pass a custom path to the training command).

Configured files are `diabetes.csv`, `heart.csv`, and `ckd.csv`. Dataset schema and target-column assumptions are in `configs/`. Running `python scripts/download_data.py` without arguments only lists local dataset availability; it never downloads automatically. To download, specify the disease and an explicitly authorized URL:

Example after confirming the source terms:

```powershell
python scripts/download_data.py --disease heart --url "<authorized-dataset-download-url>"
```

Profile all configured datasets with `python scripts/profile_data.py`. Missing datasets are reported as `not_downloaded`; existing data is schema-validated before counts and missingness are reported.

Do not put identifiable or sensitive personal health information in this repository. Use only data you are authorized to process.
