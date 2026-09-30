# Architecture

```mermaid
flowchart LR
    subgraph DATA["Data (Stage 1)"]
        UCI[("UCI archive<br/>4 hospital files")] -->|"make data<br/>SHA-256 pinned"| RAW["data/raw<br/>+ MANIFEST.json"]
        RAW -->|"make prepare<br/>validation gate"| CLEAN["heart_clean.csv<br/>+ DATA_CARD"]
    end
    subgraph TRAIN["Training and tracking (Stages 2-4)"]
        CLEAN -->|"make train"| CV["Grid search, stratified 5-fold CV<br/>LR · RF · HGB, 256 candidates"]
        CV -->|"runs, metrics, plots"| MLF[("MLflow tracking<br/>SQLite")]
        MLF -->|"alias champion"| REG[("MLflow registry")]
        REG -->|"make package<br/>byte-for-byte copy"| PKG["models/champion<br/>skops + lineage.json"]
    end
    subgraph CI["GitHub Actions (Stage 5)"]
        LINT["lint"] --> TEST["test"]
        TEST --> CITRAIN["train + quality gate"]
        TEST --> CIDOCKER["docker build + smoke"]
    end
    subgraph K8S["Docker Desktop Kubernetes (Stages 7-8)"]
        LB["LoadBalancer :80"] --> POD1["API pod"]
        LB --> POD2["API pod"]
        PROM["Prometheus<br/>rules + alerts"] -->|"scrape /metrics"| POD1
        PROM -->|"scrape /metrics"| POD2
        GRAF["Grafana<br/>dashboard as code"] --> PROM
    end
    PKG -->|"docker build<br/>tag = commit"| IMG["heartrisk-api image<br/>non-root, 162 MB"]
    IMG -->|"make k8s-deploy"| LB
    CLIENT(["Client"]) -->|"POST /predict<br/>X-Request-ID"| LB
```

## How one prediction is traced

| Step | Where it is recorded |
|---|---|
| The response names the model | Body `model.version` / `model.run_id`, header `X-Model-Version` |
| The request is logged without clinical data | One JSON audit line per prediction: request id, model version, SHA-256 of the input, prediction, probability, imputed fields |
| The model version leads to its lineage | `GET /model-info` returns `models/champion/lineage.json` |
| Lineage leads to the training run | MLflow run id and parent run id, git commit, library versions |
| The run leads to the exact data | `data_sha256` of `heart_clean.csv` and the SHA-256 of `data/raw/MANIFEST.json` |
| The manifest leads to the source | URL, size and SHA-256 of every UCI file, retrieved at a recorded time |
| The deployment names its release | Image tag = commit; pod annotations and ConfigMap `heartrisk-release` carry commit, model version and run |

## Components

| Component | Technology | Why |
|---|---|---|
| Data validation gate | pandera schema built from `configs/data.yaml` | One definition of "plausible" shared by the gate, the clean-file check and the API |
| Model | scikit-learn pipeline (impute → scale / one-hot → model) | Preprocessing travels with the model, so there is nothing to re-implement at serving time |
| Experiment tracking and registry | MLflow 3.16 on SQLite | Parent / family / candidate run hierarchy; aliases instead of deprecated stages |
| Serialisation | skops (via MLflow's sklearn flavour) | Loading refuses types outside a pinned trust list; pickle would execute arbitrary code |
| API | FastAPI + uvicorn, one worker per pod | Typed contract with 422s, OpenAPI docs, Prometheus metrics |
| Container | Multi-stage Dockerfile, python 3.12 slim, uid 10001 | 162 MB pull, no compilers or wheels in the final image |
| Orchestration | Kustomize (base + release component + overlays) | No Helm needed; release values injected with `replacements` |
| Monitoring | Prometheus 3.15 + Grafana 13.2, own manifests | Scrapes annotated pods; alerts on input drift as well as outputs |
| CI | GitHub Actions | Lint, tests, reproducible data, training with a quality gate, container smoke test |
