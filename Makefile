# heartrisk: one entry point per pipeline stage. `make help` lists targets.
# Stages are added as the assignment progresses (data, train, serve, deploy, monitor).
SHELL  := /bin/bash
PYTHON ?= python
PKG    := src/heartrisk

.DEFAULT_GOAL := help
.PHONY: help bootstrap check install lock lint format test

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

bootstrap: ## Lab VM: set up tools, git and the venv (run once per fresh VM)
	bash scripts/vm_bootstrap.sh

check: ## Lab VM: print tool versions and connectivity
	bash scripts/vm_bootstrap.sh --check

install: ## Install pinned requirements into the current Python (CI, Docker, any venv)
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

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
