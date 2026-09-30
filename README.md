# Heart-Disease Risk API: an end-to-end MLOps pipeline

BITS Pilani WILP · Machine Learning Operations (AIMLCZG523) · Assignment 1
Author: Rajesh Patel (BITS ID: _to fill_)

A classifier that predicts the presence of heart disease from routine clinical measurements
(UCI Heart Disease dataset), trained with tracked experiments, tested and built in CI, and served
as a containerised, monitored API on Kubernetes.

**Design principle: traceability.** Every prediction the API returns should be attributable to
a model version, the training run that produced it, and the exact dataset it was trained on, in
the way that GxP-regulated clinical systems demand an audit trail.

## Status

| # | Task | Marks | Status |
|---|---|---|---|
| 0 | Environment (WSL2 + Docker Desktop; lab VM alternative) + repo skeleton | – | ✅ |
| 1 | Data acquisition, cleaning, EDA | 5 | ✅ |
| 2 | Feature engineering + model development | 8 | ✅ |
| 3 | Experiment tracking (MLflow) | 5 | ✅ |
| 4 | Model packaging + reproducibility | 7 | ✅ |
| 5 | CI/CD pipeline + automated tests | 8 | 🟡 workflow written and validated (actionlint); first run on first push |
| 6 | Containerised serving API | 5 | ✅ |
| 7 | Kubernetes deployment | 7 | ✅ LoadBalancer on localhost; Traefik Ingress overlay ready |
| 8 | Monitoring + logging | 3 | ✅ Prometheus + Grafana; input-drift alerts fire on shifted traffic |
| 9 | Report, screenshots, video | 2 | ⏳ |

## Repository layout

```
heart-disease-mlops/
├── Makefile                 one target per pipeline stage (`make help`)
├── requirements.in          direct dependencies (human-edited)
├── requirements.txt         fully pinned lock for Python 3.12 (`make lock`)
├── requirements-serve.*     API-only lock, constrained to the training versions
├── pyproject.toml           package metadata + ruff/pytest config
├── configs/data.yaml        data sources, plausibility ranges, validation-gate threshold
├── configs/train.yaml       seed, CV, model grids, selection rule, threshold (decisions D3-D6)
├── .github/workflows/ci.yml lint -> test -> train (quality gate) + docker build and smoke test
├── Dockerfile               multi-stage, non-root API image with the champion baked in
├── samples/                 example /predict and /predict/batch payloads (real UCI rows)
├── k8s/                     kustomize: base, release component, overlays local (LB) + ingress
├── monitoring/              Prometheus (discovery, rules, alerts) + Grafana (dashboard as code)
├── scripts/vm_bootstrap.sh  sets up WSL Ubuntu or the BITS lab VM
├── scripts/smoke.sh         end-to-end smoke test of a running API
├── scripts/traffic.py       normal / shifted traffic generator for the drift demonstration
├── src/heartrisk/           data/, features, train, evaluate, tracking, package, api/ (FastAPI)
├── tests/                   pytest suite
├── notebooks/01_eda.ipynb   EDA notebook (executed by `make eda`)
├── data/README.md           provenance, licence, cleaning rules
├── data/raw/                UCI source files + MANIFEST.json (SHA-256 per file)
├── data/processed/          heart_clean.csv, rejected.csv, validation report, data card
├── reports/                 EDA + model figures, model comparison, external validation, metrics JSON
├── models/champion/         exported champion (MLflow format, skops) + lineage.json
├── docs/DECISIONS.md        design decisions: options, choice, reason
├── docs/MODEL_CARD.md       generated model card: use, data, performance, limitations
└── docs/screenshots/        evidence for the report
```

## Environment

| | Primary: local PC | Alternative: BITS lab VM |
|---|---|---|
| Shell | WSL2 Ubuntu 24.04 (Python 3.12) | Rocky Linux 9.5 (Python 3.12) |
| Containers | Docker Desktop (WSL integration on) | Docker Engine |
| Kubernetes | Docker Desktop's built-in cluster (kubeadm, single node) | minikube (docker driver) + ingress addon |
| Use | Day-to-day development, deployment, screenshots | Clean-machine reproducibility check |

The repository lives on the Windows drive and is used from Ubuntu through `/mnt/d/...`.
Commits are made from Windows; the Python virtual environment lives inside WSL (`~/.venvs/heartrisk`)
for speed. `scripts/vm_bootstrap.sh` detects which environment it is in.

### Local PC: one-time setup

1. Docker Desktop → Settings → Resources → WSL integration → enable **Ubuntu-24.04** → Apply & restart.
2. Docker Desktop → Kubernetes → **Create cluster** → kubeadm (single node) → wait until it shows as running.
3. In Ubuntu:

```bash
cd "/mnt/d/Mtech_BITS_2025/github-repo/MLOps (S1-26_AIMLZG523)/assignment1/heart-disease-mlops"
bash scripts/vm_bootstrap.sh     # asks once for your Linux password (apt: make, python3-venv)
source ~/.bashrc
make check && make test
```

After that, a new Ubuntu terminal auto-activates the venv and `hr` jumps to the repo.

### BITS lab VM (alternative)

First time:

```bash
cd ~/workspace
git clone https://github.com/<github-user>/heart-disease-mlops.git   # password = your GitHub token
cd heart-disease-mlops
bash scripts/vm_bootstrap.sh && source ~/.bashrc
git config --global user.name  "Your Name"        # stored in ~/workspace, so asked only once
git config --global user.email "you@example.com"
```

Every fresh VM: `cd ~/workspace/heart-disease-mlops && git pull && bash scripts/vm_bootstrap.sh && source ~/.bashrc`

Only `~/workspace` survives VM termination, so the script keeps the pip cache, kubectl download and
git config there, and rebuilds the rest (venv on local disk, `make`, `kubectl`) in 1–2 minutes.
