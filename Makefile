# heartrisk: one entry point per pipeline stage. `make help` lists targets.
# Stages are added as the assignment progresses (data, train, serve, deploy, monitor).
SHELL  := /bin/bash
VENV   ?= $(HOME)/.venvs/heartrisk
# The project venv if it exists (WSL / lab VM), otherwise python3 (CI, Docker).
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PKG    := src/heartrisk
# Import the package straight from src/ (no install step writes into the repo).
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))
# MLflow store on the Linux filesystem (SQLite locking fails on /mnt/d); CI overrides it.
MLFLOW_TRACKING_URI ?= sqlite:///$(HOME)/.heartrisk/mlflow/mlflow.db
export MLFLOW_TRACKING_URI

.DEFAULT_GOAL := help
.PHONY: help bootstrap check install lock lint format test data verify-data prepare eda \
        train mlflow-ui experiments-report

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

bootstrap: ## Set up this Linux shell (WSL or lab VM): tools, kubeconfig, venv. Re-runnable
	bash scripts/vm_bootstrap.sh

check: ## Print tool versions, cluster status and connectivity
	bash scripts/vm_bootstrap.sh --check

install: ## Install pinned requirements into the current Python (CI, Docker, any venv)
	$(PYTHON) -m pip install -r requirements.txt

lock: ## Re-resolve requirements.in -> requirements.txt (Python 3.12, Linux x86_64)
	$(PYTHON) -m pip install -q uv
	$(PYTHON) -m uv pip compile requirements.in -o requirements.txt \
	  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --annotation-style line

lint: ## Static checks (fails the pipeline on any finding)
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format: ## Auto-fix lint + formatting
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

test: ## Unit tests with coverage
	$(PYTHON) -m pytest --cov=heartrisk --cov-report=term-missing

# ---- Stage 1: data -------------------------------------------------------------------------
data: ## Download the four UCI files; pin their SHA-256 in data/raw/MANIFEST.json
	$(PYTHON) -m heartrisk.data.download

verify-data: ## Re-hash data/raw against MANIFEST.json (fails on any mismatch)
	$(PYTHON) -m heartrisk.data.download --verify

prepare: ## Validation gate + cleaning -> data/processed (fails if the gate trips)
	$(PYTHON) -m heartrisk.data.clean

# The notebook kernel talks over local unix sockets (IPC) instead of unencrypted TCP.
eda: ## Execute the EDA notebook in place; export reports/eda.html and reports/figures/
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb \
	  --KernelManager.transport=ipc --KernelManager.ip=$$(mktemp -u /tmp/heartrisk-kernel.XXXXXX)
	$(PYTHON) -m jupyter nbconvert --to html notebooks/01_eda.ipynb --output-dir reports --output eda \
	  --template heartrisk --TemplateExporter.extra_template_basedirs=notebooks/nbconvert

# ---- Stages 2-3: model development + experiment tracking --------------------------------
train: ## Tune LR/RF/HGB with CV, evaluate, log to MLflow, register champion + challenger
	$(PYTHON) -m heartrisk.train

mlflow-ui: ## MLflow UI on http://localhost:5000 (Ctrl+C to stop)
	$(PYTHON) -m mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --host 127.0.0.1 --port 5000

experiments-report: ## Family runs from MLflow -> reports/experiments.csv
	$(PYTHON) -m heartrisk.tracking --report reports/experiments.csv
