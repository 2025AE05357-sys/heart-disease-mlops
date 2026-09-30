"""MLflow configuration, lineage tags and the experiments report.

Default store: `sqlite:///~/.heartrisk/mlflow/mlflow.db` with artifacts beside it, on the Linux
filesystem (SQLite file locking breaks on the Windows drive). Override with MLFLOW_TRACKING_URI.

    python -m heartrisk.tracking --report reports/experiments.csv
"""

import argparse
import logging
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import mlflow
import pandas as pd

from heartrisk.data.download import sha256_bytes

log = logging.getLogger(__name__)

SERVING_PACKAGES = ["scikit-learn", "pandas", "numpy", "scipy", "skops"]


def default_uri() -> str:
    """The local SQLite store under ~/.heartrisk/mlflow."""
    return f"sqlite:///{Path.home() / '.heartrisk' / 'mlflow' / 'mlflow.db'}"


def configure(uri: str | None = None) -> str:
    """Point MLflow tracking and registry at `uri` (or the env var, or the default)."""
    uri = uri or os.environ.get("MLFLOW_TRACKING_URI") or default_uri()
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return uri


def artifact_root(uri: str) -> str | None:
    """For a SQLite store, an `artifacts` directory beside the database file."""
    if not uri.startswith("sqlite:///"):
        return None
    return (Path(uri.removeprefix("sqlite:///")).parent / "artifacts").resolve().as_uri()


def ensure_experiment(name: str, uri: str) -> str:
    """Create the experiment on first use; make it the active one; return its id."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name, artifact_location=artifact_root(uri))
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def lineage_tags(data_sha256: str, manifest: Path | None, train_sites: list[str]) -> dict[str, str]:
    """Tags that tie a run to its code commit, exact data bytes and library versions."""
    sha = os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD") or "unknown"
    status = _git("status", "--porcelain", "--untracked-files=no")
    manifest_sha = (
        sha256_bytes(manifest.read_bytes()) if manifest and manifest.exists() else "unavailable"
    )
    return {
        "git_sha": sha,
        "git_dirty": "unknown" if status is None else str(bool(status)).lower(),
        "data_sha256": data_sha256,
        "raw_manifest_sha256": manifest_sha,
        "train_sites": ",".join(train_sites),
        "python": platform.python_version(),
        "sklearn": version("scikit-learn"),
        "pandas": version("pandas"),
        "mlflow": version("mlflow"),
    }


def pip_requirements() -> list[str]:
    """Pinned requirements for a logged model (avoids MLflow's slow import-tracing inference)."""
    return [f"{package}=={version(package)}" for package in SERVING_PACKAGES]


def experiments_report(experiment: str, out_csv: Path) -> pd.DataFrame:
    """One row per model-family run of the experiment: CV, hold-out and lineage columns."""
    runs = mlflow.search_runs(
        experiment_names=[experiment], filter_string="tags.run_kind = 'family'"
    )
    keep = {
        "start_time": "started",
        "tags.mlflow.parentRunId": "parent_run",
        "run_id": "run_id",
        "tags.family": "family",
        "metrics.cv_roc_auc_mean": "cv_roc_auc",
        "metrics.cv_recall_mean": "cv_recall",
        "metrics.holdout_roc_auc": "holdout_roc_auc",
        "metrics.holdout_recall": "holdout_recall",
        "tags.champion": "champion",
        "tags.git_sha": "git_sha",
        "tags.data_sha256": "data_sha256",
    }
    table = runs[[c for c in keep if c in runs.columns]].rename(columns=keep)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    return table


def main(argv: list[str] | None = None) -> int:
    """CLI: write the experiments report."""
    parser = argparse.ArgumentParser(description="MLflow experiments report")
    parser.add_argument("--experiment", default="heart-disease-risk")
    parser.add_argument("--report", default="reports/experiments.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    configure()
    table = experiments_report(args.experiment, Path(args.report))
    log.info("%d family runs -> %s", len(table), args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
