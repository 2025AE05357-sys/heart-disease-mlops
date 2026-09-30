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
        train mlflow-ui experiments-report package all \
        serve docker-build docker-run smoke docker-stop \
        k8s-release k8s-validate k8s-deploy k8s-status k8s-smoke k8s-logs k8s-self-heal k8s-down \
        monitoring-up grafana traffic traffic-shift monitoring-down report

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

bootstrap: ## Set up this Linux shell (WSL or lab VM): tools, kubeconfig, venv. Re-runnable
	bash scripts/vm_bootstrap.sh

check: ## Print tool versions, cluster status and connectivity
	bash scripts/vm_bootstrap.sh --check

install: ## Install pinned requirements into the current Python (CI, Docker, any venv)
	$(PYTHON) -m pip install -r requirements.txt

lock: ## Re-resolve both locks: requirements.txt, then requirements-serve.txt constrained by it
	$(PYTHON) -m pip install -q uv
	$(PYTHON) -m uv pip compile requirements.in -o requirements.txt \
	  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --annotation-style line
	$(PYTHON) -m uv pip compile requirements-serve.in -o requirements-serve.txt -c requirements.txt \
	  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --annotation-style line

lint: ## Static checks (fails the pipeline on any finding)
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

format: ## Auto-fix lint + formatting
	$(PYTHON) -m ruff check --fix src tests scripts
	$(PYTHON) -m ruff format src tests scripts

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

# ---- Stage 4: packaging + reproducibility --------------------------------------------------
package: ## Export the registered champion -> models/champion/ + lineage.json + docs/MODEL_CARD.md
	$(PYTHON) -m heartrisk.package

all: verify-data prepare train package ## Full pipeline from committed raw data to a packaged model

# ---- Stage 6: API + container ----------------------------------------------------------------
IMAGE     ?= heartrisk-api
# The commit, with "-dirty" when the working tree has uncommitted changes.
IMAGE_TAG ?= $(shell git describe --always --dirty --abbrev=7 2>/dev/null || echo dev)
MODEL_META = $(PYTHON) -c "import json,sys; print(json.load(open('models/champion/lineage.json'))['model'][sys.argv[1]])"

serve: ## Run the API locally with auto-reload on http://localhost:8000
	$(PYTHON) -m uvicorn heartrisk.api.app:app --reload --host 127.0.0.1 --port 8000

docker-build: ## Build the API image, tagged with the git commit (and :latest)
	docker build \
	  --build-arg GIT_SHA=$(IMAGE_TAG) \
	  --build-arg BUILD_DATE=$$(date -u +%Y-%m-%dT%H:%M:%SZ) \
	  --build-arg MODEL_VERSION=$$($(MODEL_META) version) \
	  --build-arg MODEL_RUN_ID=$$($(MODEL_META) run_id) \
	  -t $(IMAGE):$(IMAGE_TAG) -t $(IMAGE):latest .
	docker image ls $(IMAGE):$(IMAGE_TAG)

docker-run: ## Start the container detached on http://localhost:8000
	docker run -d --rm --name heartrisk-api -p 8000:8000 $(IMAGE):$(IMAGE_TAG)

smoke: ## Smoke-test a running API (URL=... to override http://localhost:8000)
	PYTHON=$(PYTHON) bash scripts/smoke.sh $(or $(URL),http://localhost:8000)

docker-stop: ## Stop the container
	docker stop heartrisk-api

# ---- Stage 7: Kubernetes (Docker Desktop cluster) ---------------------------------------------
K8S_OVERLAY ?= k8s/overlays/local
K8S_NS      := heartrisk
K8S_APP     := -l app.kubernetes.io/name=heartrisk-api
RELEASE_ENV := k8s/components/release/release.env

k8s-release: ## Write the release values (image tag, commit, model version, run) for kustomize
	@printf 'IMAGE_TAG=%s\nGIT_SHA=%s\nMODEL_VERSION=%s\nMODEL_RUN_ID=%s\n' \
	  "$(IMAGE_TAG)" "$$(git rev-parse HEAD 2>/dev/null || echo unknown)" \
	  "$$($(MODEL_META) version)" "$$($(MODEL_META) run_id)" > $(RELEASE_ENV)
	@cat $(RELEASE_ENV)

k8s-validate: k8s-release ## Server-side dry run of both overlays (schema + admission checks)
	@# A dry run cannot place objects in a namespace that does not exist yet, so create it first.
	kubectl apply -f k8s/base/namespace.yaml
	kubectl apply --dry-run=server -k k8s/overlays/local
	kubectl apply --dry-run=server -k k8s/overlays/ingress

k8s-deploy: docker-build k8s-release ## Build the image, apply the overlay, wait for the rollout
	kubectl apply -k $(K8S_OVERLAY)
	kubectl -n $(K8S_NS) rollout status deploy/heartrisk-api --timeout=180s

k8s-status: ## Deployment, pods, service, PDB and the release record
	kubectl -n $(K8S_NS) get deploy,rs,pods,svc,pdb -o wide
	kubectl -n $(K8S_NS) get cm heartrisk-release -o jsonpath='{.data}{"\n"}'

k8s-smoke: ## Smoke test through the LoadBalancer (URL=... to override http://localhost)
	PYTHON=$(PYTHON) bash scripts/smoke.sh $(or $(URL),http://localhost)

k8s-logs: ## Recent JSON logs from every API pod
	kubectl -n $(K8S_NS) logs $(K8S_APP) --tail=20 --prefix

k8s-self-heal: ## Delete one API pod and show the Deployment replacing it
	kubectl -n $(K8S_NS) get pods $(K8S_APP)
	kubectl -n $(K8S_NS) delete pod "$$(kubectl -n $(K8S_NS) get pods $(K8S_APP) -o jsonpath='{.items[0].metadata.name}')"
	@for i in $$(seq 1 60); do \
	  ready="$$(kubectl -n $(K8S_NS) get deploy heartrisk-api -o jsonpath='{.status.readyReplicas}')"; \
	  [ "$$ready" = "$$(kubectl -n $(K8S_NS) get deploy heartrisk-api -o jsonpath='{.spec.replicas}')" ] && break; \
	  sleep 2; \
	done
	kubectl -n $(K8S_NS) get pods $(K8S_APP)

k8s-down: ## Remove the API from the cluster
	kubectl delete -k $(K8S_OVERLAY) --ignore-not-found

# ---- Stage 8: monitoring (Prometheus + Grafana in namespace monitoring) ------------------------
monitoring-up: ## Deploy Prometheus + Grafana; creates the Grafana admin secret once
	kubectl apply -f monitoring/namespace.yaml
	@kubectl -n monitoring get secret grafana-admin >/dev/null 2>&1 || \
	  kubectl -n monitoring create secret generic grafana-admin \
	    --from-literal=password="$$(openssl rand -hex 12)"
	kubectl apply -k monitoring
	kubectl -n monitoring rollout status deploy/prometheus --timeout=180s
	kubectl -n monitoring rollout status deploy/grafana --timeout=180s
	@$(MAKE) --no-print-directory grafana

grafana: ## Print the Grafana and Prometheus URLs and the Grafana admin password
	@echo "Grafana     http://localhost:3000  user admin  password $$(kubectl -n monitoring get secret grafana-admin -o jsonpath='{.data.password}' | base64 -d)"
	@echo "Prometheus  http://localhost:9090/targets  and  /alerts"

traffic: ## 2 minutes of normal traffic (Cleveland patients) through the LoadBalancer
	$(PYTHON) scripts/traffic.py --url $(or $(URL),http://localhost) --mode normal --duration 120

traffic-shift: ## 2 minutes of shifted traffic (Swiss + VA patients): the drift demonstration
	$(PYTHON) scripts/traffic.py --url $(or $(URL),http://localhost) --mode shift --duration 120

monitoring-down: ## Remove Prometheus + Grafana (and their namespace)
	kubectl delete -k monitoring --ignore-not-found

# ---- Stage 9: report ----------------------------------------------------------------------------
# Built with Chrome through Playwright; on this machine that is the Windows Python (python.exe via
# WSL interop). Override REPORT_PYTHON for any Python with `markdown` and `playwright` installed.
REPORT_PYTHON ?= python.exe
report: ## docs/report/report.md -> docs/report/report.pdf (lists screenshots still missing)
	$(REPORT_PYTHON) scripts/build_report.py
