PYTHON ?= python

.PHONY: install test lint format typecheck api ui migrate profile train
install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest --cov=xai_dashboard

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src/xai_dashboard

api:
	uvicorn xai_dashboard.api.main:app --reload

ui:
	streamlit run src/xai_dashboard/ui/Home.py

migrate:
	alembic upgrade head

profile:
	$(PYTHON) scripts/profile_data.py

train:
	$(PYTHON) scripts/train_all.py
