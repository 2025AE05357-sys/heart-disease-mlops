"""Export the registered champion with its lineage and model card (Stage 4).

`models/champion/` is copied byte for byte from the artifact MLflow registered, so the model the
API serves is provably the one that was selected. `lineage.json` ties it to the code commit, the
exact data bytes and the metrics, and records a reproducibility check: the exported file must
reproduce the hold-out ROC-AUC logged at training time. `docs/MODEL_CARD.md` explains the model
to a human reader.

    python -m heartrisk.package [--config configs/train.yaml] [--alias champion]
"""

import argparse
import json
import logging
import math
import shutil
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from heartrisk import evaluate, model_io, tracking
from heartrisk.data.clean import load_clean
from heartrisk.data.download import sha256_bytes
from heartrisk.data.schema import SITE_LABELS
from heartrisk.features import feature_frame
from heartrisk.train import load_train_config, split_development

log = logging.getLogger(__name__)

LINEAGE_NAME = "lineage.json"
METRIC_KEYS = ["roc_auc", "accuracy", "precision", "recall", "specificity", "f1"]


class LineageError(RuntimeError):
    """The exported model cannot be tied to the data or metrics it was registered with."""


def _iso(millis: int) -> str:
    return datetime.fromtimestamp(millis / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(metrics: dict[str, float]) -> dict[str, float | None]:
    return {
        k: (None if isinstance(v, float) and math.isnan(v) else round(v, 6))
        for k, v in metrics.items()
    }


def export_champion(
    config: dict[str, Any],
    out_dir: Path,
    card_path: Path | None = None,
    tracking_uri: str | None = None,
    alias: str = "champion",
) -> dict[str, Any]:
    """Copy the aliased model version to `out_dir`, write lineage.json and the model card."""
    tracking.configure(tracking_uri)
    client = MlflowClient()
    name = config["tracking"]["registered_model"]
    model_version = client.get_model_version_by_alias(name, alias)
    run = client.get_run(model_version.run_id)
    tags = run.data.tags
    parent_id = tags.get("mlflow.parentRunId")
    lineage_tags = client.get_run(parent_id).data.tags if parent_id else tags

    with tempfile.TemporaryDirectory() as tmp:
        uri = f"models:/{name}/{model_version.version}"
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmp))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(local, out_dir)
    model = model_io.load_model(out_dir)

    clean_path = Path(config["data"]["clean_file"])
    data_sha = sha256_bytes(clean_path.read_bytes())
    if data_sha != lineage_tags.get("data_sha256"):
        raise LineageError(
            f"{clean_path} has changed since training (sha256 {data_sha[:16]}… vs "
            f"{lineage_tags.get('data_sha256', '?')[:16]}…); retrain before packaging"
        )
    df = load_clean(clean_path)
    X_train, X_test, _, y_test, external = split_development(df, config)

    threshold = config["threshold"]
    proba = model.predict_proba(X_test)[:, 1]
    holdout = evaluate.classification_metrics(y_test, proba, threshold)
    logged = run.data.metrics.get("holdout_roc_auc")
    if logged is None or abs(holdout["roc_auc"] - logged) > 1e-9:
        raise LineageError(
            f"exported model gives hold-out ROC-AUC {holdout['roc_auc']:.6f}, "
            f"training logged {logged}; the artifact does not reproduce its own metrics"
        )

    sex = X_test["sex"].to_numpy()
    slices = {
        label: _clean(
            evaluate.classification_metrics(y_test[sex == code], proba[sex == code], threshold)
        )
        for code, label in [(0, "female"), (1, "male")]
    }
    external_metrics = {}
    for site, frame in external.groupby("site", sort=False):
        p_site = model.predict_proba(feature_frame(frame))[:, 1]
        y_site = frame["target"].astype(int).to_numpy()
        external_metrics[site] = _clean(evaluate.classification_metrics(y_site, p_site, threshold))

    model_file = out_dir / "model.skops"
    lineage = {
        "model": {
            "name": name,
            "version": int(model_version.version),
            "alias": alias,
            "family": model_version.tags.get("family"),
            "selection_reason": model_version.tags.get("selection_reason"),
            "registered_at_utc": _iso(model_version.creation_timestamp),
            "run_id": model_version.run_id,
            "parent_run_id": parent_id,
            "file": model_file.name,
            "file_sha256": sha256_bytes(model_file.read_bytes()),
        },
        "code": {
            "git_sha": lineage_tags.get("git_sha"),
            "git_dirty": lineage_tags.get("git_dirty"),
        },
        "data": {
            "clean_file": str(clean_path),
            "clean_file_sha256": data_sha,
            "raw_manifest_sha256": lineage_tags.get("raw_manifest_sha256"),
            "train_sites": config["data"]["train_sites"],
            "rows": {
                "train": len(X_train),
                "holdout": len(X_test),
                "external": {
                    s: int(n) for s, n in external["site"].value_counts(sort=False).items()
                },
            },
        },
        "libraries": {
            "python": lineage_tags.get("python"),
            "scikit-learn": lineage_tags.get("sklearn"),
            "pandas": lineage_tags.get("pandas"),
            "mlflow": lineage_tags.get("mlflow"),
            "skops": version("skops"),
        },
        "decision": {"threshold": threshold, "risk_bands": config["risk_bands"]},
        "metrics": {
            "cv": {k: round(v, 6) for k, v in run.data.metrics.items() if k.startswith("cv_")},
            "holdout": _clean({k: holdout[k] for k in ["n", "prevalence", *METRIC_KEYS]}),
            "holdout_by_sex": slices,
            "external": external_metrics,
        },
        "reproducibility_check": {
            "holdout_roc_auc_logged_at_training": round(logged, 9),
            "holdout_roc_auc_from_exported_file": round(holdout["roc_auc"], 9),
            "identical": True,
        },
    }
    (out_dir / LINEAGE_NAME).write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    if card_path is not None:
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(render_model_card(lineage), encoding="utf-8")
    log.info(
        "exported %s v%s (%s) -> %s",
        name,
        model_version.version,
        lineage["model"]["family"],
        out_dir,
    )
    return lineage


def _row(label: str, m: dict) -> str:
    cells = [
        "–" if m.get(k) is None else f"{m[k]:.3f}"
        for k in ["roc_auc", "recall", "specificity", "precision"]
    ]
    prevalence = "–" if m.get("prevalence") is None else f"{100 * m['prevalence']:.0f}%"
    return f"| {label} | {m['n']} | {prevalence} | " + " | ".join(cells) + " |"


def render_model_card(lineage: dict[str, Any]) -> str:
    """Markdown model card generated from lineage.json (every number comes from the data)."""
    model, data, metrics = lineage["model"], lineage["data"], lineage["metrics"]
    cv = metrics["cv"]
    ext = metrics["external"]
    ext_auc = [m["roc_auc"] for m in ext.values() if m.get("roc_auc") is not None]
    band = lineage["decision"]["risk_bands"]
    header = (
        "| Set | n | Disease | ROC-AUC | Recall | Specificity | Precision |\n"
        "|---|---|---|---|---|---|---|"
    )
    family = evaluate.FAMILY_LABELS.get(model["family"], model["family"])
    code = lineage["code"]
    lines = [
        "# Model card: heart-disease-classifier",
        "",
        "Generated by `python -m heartrisk.package` (`make package`) from "
        "`models/champion/lineage.json`. Do not edit by hand.",
        "",
        "> **Educational project. Not a medical device and not medical advice.** The model has not "
        "been validated prospectively and must not be used to make or support clinical decisions.",
        "",
        "## Model details",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Registered model | `{model['name']}` version {model['version']} "
        f"(alias `{model['alias']}`) |",
        f"| Family | {family} in a scikit-learn pipeline (impute → scale / one-hot → model) |",
        f"| Selected because | {model['selection_reason']} |",
        f"| Registered | {model['registered_at_utc']} |",
        f"| MLflow run | `{model['run_id']}` (parent `{model['parent_run_id']}`) |",
        f"| Code commit | `{code['git_sha']}` (uncommitted changes: {code['git_dirty']}) |",
        f"| Model file | `{model['file']}` (skops), SHA-256 `{model['file_sha256']}` |",
        "",
        "## Intended use",
        "",
        "Demonstrating an MLOps pipeline for a binary classifier of coronary heart disease "
        "(angiographic narrowing > 50% in any major vessel) from 13 routine clinical measurements. "
        "Out of scope: diagnosis, triage, screening or any use on real patients.",
        "",
        "## Training data",
        "",
        f"UCI Heart Disease, Cleveland Clinic Foundation: {data['rows']['train']} patients for "
        f"training and cross-validation, {data['rows']['holdout']} held out (stratified, seed 42). "
        "The Hungarian, Swiss and VA Long Beach databases were never used for training or tuning; "
        f"they are external validation sets. Clean data SHA-256 `{data['clean_file_sha256']}`.",
        "",
        "## Performance",
        "",
        f"Cross-validation (5 folds, training set): ROC-AUC {cv['cv_roc_auc_mean']:.3f} ± "
        f"{cv['cv_roc_auc_std']:.3f}, recall {cv['cv_recall_mean']:.3f} ± "
        f"{cv['cv_recall_std']:.3f}.",
        f"Class threshold {lineage['decision']['threshold']}.",
        "",
        header,
        _row("Cleveland hold-out", metrics["holdout"]),
        *[_row(f"{SITE_LABELS.get(s, s)} (external)", m) for s, m in ext.items()],
        "",
        "## Slice by sex (Cleveland hold-out)",
        "",
        "The slices are small (n in the table); a perfect score on a few dozen patients is noise, "
        "not evidence of better performance. They show whether a gap is large, nothing finer.",
        "",
        header,
        *[_row(label.title(), m) for label, m in metrics["holdout_by_sex"].items()],
        "",
        "## Limitations",
        "",
        "| Limitation | Evidence and consequence |",
        "|---|---|",
        f"| Small, old sample | {data['rows']['train']} training patients from the 1980s; "
        "practice, populations and test protocols have changed since |",
        "| Does not transfer across hospitals | External ROC-AUC "
        f"{min(ext_auc):.2f}–{max(ext_auc):.2f} against {metrics['holdout']['roc_auc']:.2f} on "
        "the hold-out set: case mix, prevalence and recording practice differ by site |",
        "| Missing tests at other sites | `ca` (fluoroscopy) and `thal` (thallium scan) are "
        "missing for most external patients and are imputed with Cleveland's most common value |",
        "| Sex as a predictor | Male sex carries a large positive weight. This reflects the cohort "
        "(two thirds male, higher disease rate in men), not a causal effect; see the slice above |",
        "| Calibration | Probabilities are calibrated to Cleveland's prevalence; at a site with a "
        "different prevalence the probabilities shift even when the ranking holds |",
        "",
        "## Risk bands (API)",
        "",
        f"Besides the class, the API returns a band: low (< {band['low']}), moderate "
        f"({band['low']}–{band['high']}), high (≥ {band['high']}). The bands are a reading aid, "
        "not a clinical risk score.",
        "",
        "## Monitoring",
        "",
        "Production metrics track request volume and latency, the predicted-class mix and the "
        "probability distribution, and the distribution of age, cholesterol and maximum heart "
        "rate. "
        "A shift towards the Swiss or VA profile (older patients, lower heart rates, more "
        "asymptomatic chest pain) is the expected failure mode and the retraining trigger.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Export the champion with lineage and model card")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--out-dir", default="models/champion")
    parser.add_argument("--card", default="docs/MODEL_CARD.md")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    config = load_train_config(args.config)
    try:
        export_champion(config, Path(args.out_dir), Path(args.card), alias=args.alias)
    except LineageError as exc:
        log.error("PACKAGE FAIL: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
