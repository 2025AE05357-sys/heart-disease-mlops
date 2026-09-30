# Video script (6–8 minutes)

Record the screen with Xbox Game Bar (Win+Alt+R) or OBS at 1080p. Use one Ubuntu terminal
(font 16 pt or larger), a browser, and VS Code. Before recording: `make k8s-deploy
monitoring-up`, and run `make traffic` once so Grafana has a baseline. Upload as unlisted and put
the link in the README and the report.

| # | Time | Scene | Show | Say (key points) |
|---|---|---|---|---|
| 1 | 0:00–0:30 | Title | README top: title, disclaimer, architecture diagram | Heart-disease classifier from UCI data, taken to a monitored API; theme: every prediction is traceable |
| 2 | 0:30–1:15 | Data | `make data verify-data prepare` in the terminal; `data/raw/MANIFEST.json` | Four hospital files pinned by SHA-256; the validation gate recodes "not measured" zeros and would reject implausible rows with a reason |
| 3 | 1:15–2:00 | EDA | `reports/eda.html`: missingness (as recorded vs decoded) and site-shift figures | Cleveland is complete; other sites lack tests; Swiss cholesterol is 100% missing once decoded; the sites are different populations |
| 4 | 2:00–3:00 | Training + MLflow | `make train` log (tuning, champion line); `make mlflow-ui`: parent/child runs, compare view, a run's plots, registry alias `champion` | 256 candidates under 5-fold CV; selection rule D5; external validation drop; everything logged, champion and challenger by alias |
| 5 | 3:00–3:40 | Packaging | `models/champion/lineage.json`, `docs/MODEL_CARD.md` | Exact registered file (skops), commit, data hashes, reproducibility check; model card limitations |
| 6 | 3:40–4:30 | CI | GitHub Actions: green run graph, job summary with metrics, artifacts; the red failure-demo run | Lint, tests, reproducible data, training with a quality gate, container smoke test; a failing test stops the pipeline |
| 7 | 4:30–5:15 | Container | `make docker-build docker-run smoke`; Swagger UI at http://localhost:8000/docs with a /predict call | Non-root 162 MB image, labels carry commit and model; the response names the model; imputed fields are reported |
| 8 | 5:15–6:15 | Kubernetes | `make k8s-status`; `make k8s-smoke`; `make k8s-self-heal`; Docker Desktop Kubernetes view | 2 replicas behind a LoadBalancer, restricted Pod Security, probes; a deleted pod is replaced; release stamped on the pods |
| 9 | 6:15–7:30 | Monitoring + drift | Grafana dashboard; run `make traffic-shift`; heart-rate and imputation panels move; Prometheus Alerts page with `HeartriskInputDrift` firing; `make k8s-logs` JSON lines | Input drift is visible and alerts fire, while the positive rate barely moves: output-only monitoring would miss it |
| 10 | 7:30–8:00 | Close | README results table and documentation links | What was built, the key finding (site shift), limitations, next steps |
