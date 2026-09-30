# Runbook

All commands run in Ubuntu (WSL) from the repository root, with the project venv active.

## Deploy the API

```bash
make k8s-deploy        # builds heartrisk-api:<commit>, applies k8s/overlays/local, waits for rollout
make k8s-status        # deployment, pods, service, PDB, and the release record
make k8s-smoke         # health, predictions, 422 on invalid input, lineage, metrics via http://localhost
```

"What exactly is deployed?" is answered in the cluster:

```bash
kubectl -n heartrisk get cm heartrisk-release -o jsonpath='{.data}'
kubectl -n heartrisk get deploy heartrisk-api -o jsonpath='{.spec.template.metadata.annotations}'
curl -s http://localhost/model-info | python -m json.tool
```

## Release a new model

| Step | Command | Check |
|---|---|---|
| 1. Train and register | `make train` | Log line `champion: …`; `make mlflow-ui` shows the new runs |
| 2. Package | `make package` | `models/champion/lineage.json` has the new version; the reproducibility check passed (otherwise the target fails) |
| 3. Review | `git diff models/champion/lineage.json docs/MODEL_CARD.md` | Metrics, external validation and the selection reason make sense |
| 4. Commit | `git commit` | CI re-runs the full pipeline and the quality gate |
| 5. Deploy | `make k8s-deploy` | New pod-template annotations trigger a rolling update with `maxUnavailable: 0` |

## Roll back

```bash
kubectl -n heartrisk rollout history deploy/heartrisk-api
kubectl -n heartrisk rollout undo deploy/heartrisk-api            # previous revision
kubectl -n heartrisk rollout undo deploy/heartrisk-api --to-revision=N
```

The image of each revision carries its own model, so a rollback restores model and code together.
Re-apply the overlay afterwards if the ConfigMap `heartrisk-release` should match again.

## Monitoring

```bash
make monitoring-up     # Prometheus http://localhost:9090, Grafana http://localhost:3000
make grafana           # prints the URLs and the admin password (from Secret grafana-admin)
make traffic           # 2 min of Cleveland patients (the training population)
make traffic-shift     # 2 min of Swiss + VA patients: drift panels move, alerts fire in ~3 min
```

| Alert | Meaning | First action |
|---|---|---|
| `HeartriskInputDrift` | Mean maximum heart rate is > 15 beats/min from the training mean | Identify the sending site; external validation suggests ROC-AUC near 0.70 there |
| `HeartriskHighImputation` | More than one input per prediction arrives empty | Check which fields (`Imputed inputs` panel); predictions rest on estimates |
| `HeartriskPositiveRateHigh` | > 80% of predictions are "disease" | Compare with the case mix; a real shift or a broken input |
| `HeartriskServerErrors` | > 1% of requests return 5xx | `make k8s-logs`; `kubectl -n heartrisk describe pod …` |

To rotate the Grafana password: `kubectl -n monitoring delete secret grafana-admin`, then
`make monitoring-up` and restart Grafana (`kubectl -n monitoring rollout restart deploy/grafana`).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Pod `Running` but `0/1 READY` | The model failed to load; `/ready` returns 503 by design | `make k8s-logs` shows `model_load_failed` with the reason |
| `ErrImageNeverPull` / `ImagePullBackOff` | The tag in `release.env` was never built locally | `make k8s-deploy` (builds first) |
| Pods rejected with a Pod Security error | A manifest change broke the `restricted` standard | Restore `runAsNonRoot`, `drop: ALL`, `allowPrivilegeEscalation: false`, seccomp |
| Grafana panels "No data", data source not found | Grafana 13 must write its bundled plugins at start-up | Keep `readOnlyRootFilesystem: false` for Grafana only (see its manifest) |
| Grafana `CrashLoopBackOff`, last state `OOMKilled` | Memory grew until the limit while the dashboard refreshed (seen at 512 MiB) | Keep `GOMEMLIMIT=1000MiB` and the 1.5 GiB limit; `kubectl -n monitoring describe pod -l app.kubernetes.io/name=grafana` shows the last state |
| `database is locked` in MLflow | Tracking store on the Windows drive | Keep `MLFLOW_TRACKING_URI` on the Linux filesystem (the Makefile default) |
| `make prepare` fails with `GATE FAIL` | Raw data changed or rows broke the plausibility rules | Read `data/processed/validation_report.json` and `rejected.csv` |
| `make package` fails with `LineageError` | Data changed since training, or the artifact does not reproduce its metrics | Retrain (`make train`) before packaging |
| Port 80, 3000 or 9090 busy | Another LoadBalancer or local service | `kubectl get svc -A`; remove the other service or change the port |
