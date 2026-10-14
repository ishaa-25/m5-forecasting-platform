.PHONY: install test lint format data api dashboard docker-build docker-run clean

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[advanced,llm]"

install-core:
	pip install -e .

# ── Quality ────────────────────────────────────────────────────────────────────
lint:
	ruff check src tests

format:
	ruff format src tests

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

# ── Data pipeline ──────────────────────────────────────────────────────────────
data:
	bash scripts/download_data.sh
	python -m m5forecast.data.ingest

features:
	python -m m5forecast.data.features

# ── Training shortcuts ─────────────────────────────────────────────────────────
train-baselines:
	python -m m5forecast.models.baselines

train-lgbm:
	python -m m5forecast.models.lgbm

train-all:
	python -m m5forecast.models.baselines
	python -m m5forecast.models.lgbm
	python -m m5forecast.models.foundation
	python -m m5forecast.models.deep

# ── Serving ────────────────────────────────────────────────────────────────────
api:
	uvicorn m5forecast.serving.api:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run app/dashboard.py

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t m5forecast:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env m5forecast:latest

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache dist build *.egg-info
