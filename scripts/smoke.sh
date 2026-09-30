#!/usr/bin/env bash
# Smoke-test a running API: readiness, both sample patients, a batch, lineage and metrics.
# Fails (non-zero) on any unexpected status, so CI and `make smoke` can rely on it.
#
#   bash scripts/smoke.sh [BASE_URL]        default http://localhost:8000
set -euo pipefail
BASE="${1:-http://localhost:8000}"
HOST_HEADER=()
if [ -n "${SMOKE_HOST:-}" ]; then HOST_HEADER=(-H "Host: ${SMOKE_HOST}"); fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"

say() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
call() {  # call METHOD PATH [JSON_FILE]
  local args=(-fsS --max-time 10 "${HOST_HEADER[@]}" -X "$1" -H "X-Request-ID: smoke-$(date +%s%N)")
  if [ -n "${3:-}" ]; then args+=(-H "Content-Type: application/json" --data @"$3"); fi
  curl "${args[@]}" "$BASE$2"
}

say "waiting for $BASE/ready"
for i in $(seq 1 30); do
  if curl -fsS --max-time 2 "${HOST_HEADER[@]}" "$BASE/ready" >/dev/null 2>&1; then break; fi
  if [ "$i" = 30 ]; then echo "not ready after 30 s"; exit 1; fi
  sleep 1
done
call GET /ready | "$PY" -m json.tool

for sample in patient_high_risk patient_low_risk patient_swiss_gaps; do
  say "POST /predict  samples/$sample.json"
  call POST /predict "$REPO/samples/$sample.json" | "$PY" -m json.tool
done

say "POST /predict/batch  samples/batch.json"
call POST /predict/batch "$REPO/samples/batch.json" |
  "$PY" -c 'import json,sys; b=json.load(sys.stdin); print(json.dumps({"model": b["model"], "probabilities": [round(p["probability"], 3) for p in b["predictions"]]}, indent=2))'

say "invalid input must be rejected with 422"
status="$(curl -s -o /dev/null -w '%{http_code}' "${HOST_HEADER[@]}" -H 'Content-Type: application/json' \
  --data '{"age": 150, "sex": 1, "cp": 4}' "$BASE/predict")"
echo "status $status"
[ "$status" = 422 ]

say "GET /model-info (summary)"
call GET /model-info | "$PY" -c 'import json,sys; m=json.load(sys.stdin)["model"]; print(json.dumps({k: m[k] for k in ("name","version","family","run_id","file_sha256")}, indent=2))'

say "GET /metrics (heartrisk_* series)"
call GET /metrics | grep -E '^heartrisk_(predictions_total|model_info|validation_errors_total)' || true

say "smoke test passed"
