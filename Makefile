# heartrisk: one entry point per pipeline stage. `make help` lists targets.
# Stages are added as the assignment progresses (data, train, serve, deploy, monitor).
SHELL  := /bin/bash
VENV   ?= $(HOME)/.venvs/heartrisk
# The project venv if it exists (WSL / lab VM), otherwise python3 (CI, Docker).
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PKG    := src/heartrisk
# Import the package straight from src/ (no install step writes into the repo).
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

.DEFAULT_GOAL := help
.PHONY: help bootstrap check install lock lint format test

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

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
