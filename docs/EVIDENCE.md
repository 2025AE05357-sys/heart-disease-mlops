# Assignment brief → evidence

Where each requirement of the brief is met: files and `make` targets in this repository,
screenshots in [`docs/screenshots/`](screenshots/), and sections of the
[report](report/report.pdf).

## Tasks (50 marks)

| # | Task (marks) | Requirement | Where it is met | Screenshots | Report |
|---|---|---|---|---|---|
| 1 | Data acquisition and EDA (5) | Download script or instructions | `src/heartrisk/data/download.py` (`make data`): four UCI files, SHA-256 pinned in `data/raw/MANIFEST.json`; `make verify-data` | `01_make_data` | §3.1 |
| | | Clean and preprocess: missing values, encoding | `src/heartrisk/data/clean.py` + `schema.py` (`make prepare`): validation gate, `?` and impossible zeros → missing (D2); imputation and one-hot encoding inside the model pipeline (`src/heartrisk/features.py`) | `02_validation_report` | §3.2, §4 |
| | | EDA: histograms, correlation heatmap, class balance | `notebooks/01_eda.ipynb` (`make eda`) → `reports/eda.html`; `reports/figures/eda_03` (histograms), `eda_05` (correlation heatmap), `eda_02` (class balance), plus missingness and site shift | – | §3.3, Figures 1–5 |
| 2 | Features and models (8) | Final features: scaling, encoding | `src/heartrisk/features.py`: median/mode imputation, `StandardScaler`, `OneHotEncoder`, optional heart-rate-reserve feature (D3) | – | §4 |
| | | At least two classifiers | Logistic regression, random forest, histogram gradient boosting (D4) | – | §4 |
| | | Selection and tuning documented | `configs/train.yaml` (grids, seed 42), `GridSearchCV`; selection rule D5; threshold and risk bands D6; `docs/DECISIONS.md` | – | §4 |
| | | Cross-validation; accuracy, precision, recall, ROC-AUC | Stratified 5-fold CV; `reports/model_comparison.csv` (CV and hold-out accuracy, precision, recall, F1, ROC-AUC); `reports/external_validation.csv`; `reports/figures/model_*` | – | §4, §11 |
| 3 | Experiment tracking (5) | MLflow: parameters, metrics, artifacts, plots for all runs | `src/heartrisk/tracking.py`, `train.py`: parent run per `make train` (config, data input, reports), family runs (params, metrics, plots, model), candidate runs (params, CV metrics); registry aliases `champion` / `challenger`; `make mlflow-ui`, `make experiments-report` | `03`–`06` | §5 |
| 4 | Packaging and reproducibility (7) | Model in a reusable format | `models/champion/`: MLflow model (skops, D7) + `lineage.json` (run, commit, data hashes, metrics); `make package` | – | §6 |
| | | Clean requirements file | `requirements.txt` (pinned lock from `requirements.in`, `make lock`); `requirements-serve.txt` for the API image | – | §6 |
| | | Preprocessing pipeline for full reproducibility | The whole preprocessing pipeline is saved inside the model; fixed seeds; committed raw data with checksums. A clean-clone run (Stage 10) retrained a model with predictions identical to the committed one | – | §6, §9 |
| 5 | CI/CD and automated tests (8) | Unit tests for data and model code | `tests/`: data (`test_download`, `test_clean`), model (`test_features`, `test_train`, `test_evaluate`, `test_package`, `test_model_io`, `test_gate`), API (`test_api`), manifests; 105 tests, 95% coverage (`make test`) | – | §7 |
| | | Pipeline with linting, unit tests, model training | `.github/workflows/ci.yml`: jobs `lint` → `test` → `train` (quality gate D9) + `docker` | `07_ci_green` | §7 |
| | | Artifacts and logging per run | Artifacts `test-results`, `training-<sha>`, `container-log`; coverage, metrics and image size in the job summaries | `08_ci_artifacts`, `09_ci_summary` | §7 |
| 6 | Containerisation (5) | Docker image for a FastAPI API | `Dockerfile` (multi-stage, non-root), `src/heartrisk/api/`; `make docker-build` | `11_docker_build` | §8 |
| | | `/predict` takes JSON, returns prediction and confidence | `POST /predict`, `POST /predict/batch`; response has `prediction`, `probability`, `confidence`, `risk_band` and the model's identity | `13_api_docs` | §8 |
| | | Builds and runs locally with sample input | `samples/*.json`; `make docker-run smoke` (`scripts/smoke.sh`); the same smoke test runs in CI | `12_docker_smoke` | §8 |
| 7 | Production deployment (7) | Deploy to local Kubernetes with manifests | `k8s/` (kustomize: base, release component, overlays); Docker Desktop cluster; `make k8s-deploy` | `15_k8s_get_all` | §9 |
| | | Expose via LoadBalancer or Ingress | `k8s/overlays/local` (LoadBalancer on `localhost:80`); `k8s/overlays/ingress` (Traefik, dry-run validated) | `16_k8s_ingress` | §9 |
| | | Verify endpoints | `make k8s-smoke`; self-healing `make k8s-self-heal` | `17_k8s_predict`, `19_k8s_self_heal` | §9 |
| 8 | Monitoring and logging (3) | Log API requests | JSON access log and per-prediction audit log on stdout (`src/heartrisk/api/audit.py`); `make k8s-logs` | `23_api_logs` | §10 |
| | | Prometheus + Grafana | `monitoring/` (scrape config, recording rules, drift alerts, dashboard as code); `make monitoring-up traffic traffic-shift` | `20`, `21`, `21b`, `22` | §10 |
| 9 | Documentation and reporting (2) | Setup, EDA and modelling choices, tracking summary, architecture diagram, CI/CD and deployment screenshots, repository link | [`docs/report/report.pdf`](report/report.pdf) (Markdown source `report.md`, `make report`); [`docs/architecture.md`](architecture.md) | all | §1–§12 |

## Deliverables

| Deliverable | Where |
|---|---|
| Code, Dockerfile, requirements | `src/heartrisk/`, `Dockerfile`, `requirements.txt`, `requirements-serve.txt` |
| Cleaned dataset and download script | `data/processed/heart_clean.csv` (+ `DATA_CARD.md`, `validation_report.json`), `make data` |
| Notebooks and scripts: EDA, training, inference | `notebooks/01_eda.ipynb`; `python -m heartrisk.train`; inference through the API (`make serve`, `make docker-run`, `make k8s-deploy`) with `samples/` and `scripts/smoke.sh` |
| Tests folder | `tests/` |
| Workflow YAML | `.github/workflows/ci.yml` |
| Deployment manifests | `k8s/`, `monitoring/` |
| Screenshot folder | `docs/screenshots/` |
| Final report | `docs/report/report.pdf` |
| Access instructions | [README](../README.md): Quick start and Using the API |
| Video of the pipeline | Link in the README |

## Production-readiness requirements

| Requirement | Evidence |
|---|---|
| Every script runs from a clean setup using the requirements file | CI installs `requirements.txt` on a fresh runner every push; Stage 10 clean-clone run: fresh clone → `scripts/vm_bootstrap.sh` → `make lint test all` → Docker → Kubernetes, all passing |
| The model serves correctly in an isolated environment | CI `docker` job builds the image and runs `scripts/smoke.sh` against it; screenshots `11`, `12` |
| The pipeline fails on code or test errors with clear logs | `set -euo pipefail` in every job; quality gate D9; screenshot `10_ci_failure` (run #1 stopped at the gate with a clear `ModuleNotFoundError`) |
