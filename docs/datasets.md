# Dataset sources and license checks

This project does not redistribute data. Dataset names below identify candidate public research datasets; verify the current source record, license, attribution, and any restrictions before downloading or using a copy. Dataset-specific terms can change, and availability does not itself grant permission for every use.

| Module | Candidate source | License / use note |
| --- | --- | --- |
| Diabetes | Pima Indians Diabetes Database, National Institute of Diabetes and Digestive and Kidney Diseases; commonly distributed by secondary repositories | The original distribution's reusable license is not reliably stated by all mirrors. Obtain an authorized copy and confirm its terms with the source before use; do not assume a mirror's license applies to the original data. |
| Heart disease | UCI Machine Learning Repository, Heart Disease dataset (`https://archive.ics.uci.edu/dataset/45/heart+disease`) | Consult the current UCI dataset page for the governing license, citation, and source-specific notes before use. |
| CKD | UCI Machine Learning Repository, Chronic Kidney Disease dataset (`https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease`) | Consult the current UCI dataset page for the governing license, citation, and source-specific notes before use. |

## Schema expectations

The YAML files in `configs/` define feature names, target columns, positive labels, feature types, display names, units, and input ranges. Obtain a CSV whose schema matches that config or create a local config for a properly licensed compatible dataset. The training code fails with a clear missing-column error rather than silently guessing a schema.

## Data handling

Raw data is not downloaded by the application, committed to version control, or copied into prediction logs. Prediction features are not stored. A prediction score is logged only when the authenticated user explicitly consents. Keep downloaded research data local and follow its terms and your institution's ethics requirements.
