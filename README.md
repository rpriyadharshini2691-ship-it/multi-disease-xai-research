# Multi-Disease XAI Research Dashboard

Research and education dashboard for configured Diabetes, Heart Disease, and Chronic Kidney Disease models. It combines JWT authentication and role checks, disease prediction, SHAP explanations, model evaluation, reliability experiments, and PDF reports.

> This is not a clinical device. A score is a model-estimated positive-class score, not a clinical probability or diagnosis. Reliability results measure consistency under the tested conditions only and do not imply medical correctness.

## Setup

Requires Python 3.11 and a local MySQL 8 instance for shared deployments. SQLite is the default for local development and tests.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set a random `SECRET_KEY` in `.env`. For MySQL, set `DATABASE_URL` to a SQLAlchemy URL such as `mysql+pymysql://user:password@localhost:3306/xai_dashboard`. Run migrations with `alembic upgrade head`.

Start the API and UI in separate terminals from the repository root:

```powershell
uvicorn xai_dashboard.api.main:app --reload
streamlit run src/xai_dashboard/ui/Home.py
```

Open `http://127.0.0.1:8502`. Register as `researcher` or `clinician`. Public registration cannot create `admin` accounts. Bootstrap an initial administrator locally with `python scripts/create_admin.py`.

## Data and models

No datasets or pretrained models are bundled. Review `docs/datasets.md`, confirm dataset permissions, and prepare local CSV files matching the YAML configs. Column names, target labels, units, and input ranges are defined in `configs/`.

Train a model and create evaluation artifacts:

```powershell
python scripts/train.py diabetes
```

Run the three reliability experiments after training:

```powershell
python scripts/run_reliability.py diabetes
```

Alternatively, authenticated `researcher` and `admin` users can start these jobs from the UI/API. Results and figures are written to `artifacts/<disease>/`; absent artifacts are shown as not generated. Dataset download is never automatic; `scripts/download_data.py` requires an explicitly provided URL and output path.

## API

The API is versioned under `/api/v1`. Interactive API documentation is available at `/docs` after starting FastAPI. Routes include authentication, user profiles and admin activity, disease metadata, prediction, local explanations, performance/reliability artifacts, PDF reports, and background training/reliability job status.

Prediction records are written only after explicit consent, linked to the authenticated user, and contain no submitted feature values. Keep `.env`, local datasets, database files, and generated artifacts private.

## Development

```powershell
pytest --cov=xai_dashboard
ruff check .
ruff format --check .
mypy src/xai_dashboard
```

See `Makefile` for common commands. Experiment results should be interpreted in the context of dataset provenance, sample size, validation design, and the limitations of post-hoc explanations.
