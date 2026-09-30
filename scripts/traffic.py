"""Send realistic prediction traffic to the API for the monitoring demonstration.

    --mode normal   Cleveland patients: the population the model was trained on
    --mode shift    Swiss and VA patients: older, lower heart rates, many untested values;
                    the drift panels and alerts in Grafana should move within a few minutes

Patients are real rows of data/processed/heart_clean.csv (sampled with replacement); a small
share of deliberately invalid requests exercises the 422 path. Standard library only.

    python scripts/traffic.py --url http://localhost --mode shift --rate 5 --duration 120
"""

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal"]  # fmt: skip
SITES = {"normal": {"cleveland"}, "shift": {"switzerland", "va"}}


def load_patients(path: Path, mode: str) -> list[dict]:
    """Rows of the clean CSV for the mode's sites, as API payloads (empty -> null)."""
    patients = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["site"] in SITES[mode]:
                patients.append(
                    {f: None if row[f] == "" else (float(row[f]) if f == "oldpeak" else int(row[f]))
                     for f in FEATURES}  # fmt: skip
                )
    return patients


def post(url: str, payload: dict, request_id: str) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "X-Request-ID": request_id},
    )  # fmt: skip
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost", help="API base URL")
    parser.add_argument("--mode", choices=sorted(SITES), default="normal")
    parser.add_argument("--rate", type=float, default=5.0, help="requests per second")
    parser.add_argument("--duration", type=float, default=120.0, help="seconds to run")
    parser.add_argument("--invalid", type=float, default=0.02, help="share of invalid requests")
    parser.add_argument("--data", default="data/processed/heart_clean.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    patients = load_patients(Path(args.data), args.mode)
    endpoint = args.url.rstrip("/") + "/predict"
    statuses, labels = Counter(), Counter()
    started = last_report = time.monotonic()
    sent = 0
    print(f"{args.mode} traffic: {len(patients)} patients, {args.rate}/s for {args.duration:.0f} s "
          f"-> {endpoint}")  # fmt: skip
    while time.monotonic() - started < args.duration:
        payload = dict(rng.choice(patients))
        if rng.random() < args.invalid:
            payload["age"] = 150  # implausible: the API must answer 422
        status, body = post(endpoint, payload, f"traffic-{args.mode}-{sent:06d}")
        statuses[status] += 1
        if body:
            labels[body["label"]] += 1
        sent += 1
        now = time.monotonic()
        if now - last_report >= 10:
            positive = labels["disease"] / max(1, sum(labels.values()))
            print(f"  {now - started:5.0f} s  sent {sent:5d}  status {dict(statuses)}  "
                  f"positive rate {positive:.0%}")  # fmt: skip
            last_report = now
        time.sleep(max(0.0, started + sent / args.rate - time.monotonic()))
    positive = labels["disease"] / max(1, sum(labels.values()))
    print(f"done: {sent} requests, status {dict(statuses)}, positive rate {positive:.0%}")
    return 0 if statuses.get(200, 0) else 1


if __name__ == "__main__":
    sys.exit(main())
