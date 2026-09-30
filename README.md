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
| 0 | Lab VM environment + repo skeleton | – | ✅ |
| 1 | Data acquisition, cleaning, EDA | 5 | ⏳ |
| 2 | Feature engineering + model development | 8 | ⏳ |
| 3 | Experiment tracking (MLflow) | 5 | ⏳ |
| 4 | Model packaging + reproducibility | 7 | ⏳ |
| 5 | CI/CD pipeline + automated tests | 8 | ⏳ |
| 6 | Containerised serving API | 5 | ⏳ |
| 7 | Kubernetes deployment | 7 | ⏳ |
| 8 | Monitoring + logging | 3 | ⏳ |
| 9 | Report, screenshots, video | 2 | ⏳ |

## Repository layout

```
heart-disease-mlops/
├── Makefile                 one target per pipeline stage (`make help`)
├── requirements.in          direct dependencies (human-edited)
├── requirements.txt         fully pinned lock for Python 3.12 (`make lock`)
├── pyproject.toml           package metadata + ruff/pytest config
├── scripts/vm_bootstrap.sh  sets up a fresh BITS lab VM
├── src/heartrisk/           the Python package (data, features, training, serving)
├── tests/                   pytest suite
├── notebooks/               EDA notebook
├── data/raw/                downloaded source files (with checksums)
├── data/processed/          cleaned dataset
├── reports/                 generated figures and metrics
└── docs/screenshots/        evidence for the report
```

## Environment: BITS Prayogshala lab VM

All pipeline stages run on the course's virtual lab VM (Rocky Linux 9.5, 4 vCPU, 15 GB RAM,
Docker 28, minikube 1.35, Python 3.12). Only `~/workspace` survives VM termination, so the
bootstrap script keeps caches and git settings there and rebuilds everything else in 1–2 minutes.

### First time only

```bash
cd ~/workspace
git clone https://github.com/<github-user>/heart-disease-mlops.git   # password = your GitHub token
cd heart-disease-mlops
bash scripts/vm_bootstrap.sh
source ~/.bashrc
git config --global user.name  "Your Name"        # stored in ~/workspace, so asked only once
git config --global user.email "you@example.com"
```

### Every fresh VM (after Terminate, or on a new day)

```bash
cd ~/workspace/heart-disease-mlops && git pull && bash scripts/vm_bootstrap.sh && source ~/.bashrc
```

### Lab VM rules that shape this repo

| Rule | How the repo copes |
|---|---|
| Only `~/workspace` persists | pip cache, kubectl download and git config live under `~/workspace`; the venv is rebuilt on local disk (faster imports than network storage) |
| 15 min idle → terminated; stopped > 60 min → terminated | Every stage is one `make` target, so a new VM gets back to any stage quickly |
| Software installs are temporary | `vm_bootstrap.sh` is idempotent and re-installs `make` and `kubectl` |
