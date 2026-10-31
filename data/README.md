# Dataset setup

No health datasets are included in this repository. Obtain each dataset directly from its source, review the source's current license and conditions, and place a CSV matching the configured column names in this directory (or pass a custom path to the training command).

Expected defaults are `diabetes.csv`, `heart.csv`, and `ckd.csv`. Dataset schema and target-column assumptions are in `configs/`.

Run the downloader with no arguments to list expected paths; this does not contact the network. To download, provide an explicitly authorized HTTP(S) URL and either a configured disease or output path:

Example after confirming the source terms:

```powershell
python scripts/download_data.py --disease heart --url "<authorized-dataset-download-url>"
python scripts/download_data.py --url "<authorized-dataset-download-url>" --output data/custom.csv
```

Profile datasets that are already present and pass their configured schema:

```powershell
python scripts/profile_data.py
python scripts/profile_data.py --disease diabetes
```

Do not put identifiable or sensitive personal health information in this repository. Use only data you are authorized to process.
