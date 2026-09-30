"""Model development and experiment tracking (Stages 2-3).

One `make train` run:

1. loads heart_clean.csv, keeps the training sites (D1) and splits off a locked, stratified
   hold-out set that no tuning step sees;
2. tunes each model family by grid search under stratified k-fold CV, scoring ROC-AUC,
   accuracy, precision, recall and F1, refitting on ROC-AUC;
3. evaluates every family on the hold-out set and on each external site, then picks the
   champion with the D5 rule;
4. logs to MLflow: one parent run, one child run per family (metrics, plots, model), one nested
   run per hyperparameter candidate; registers the champion and runner-up under aliases.

    python -m heartrisk.train [--config configs/train.yaml] [--tracking-uri URI]
"""

import argparse
import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heartrisk import evaluate, tracking, viz
from heartrisk.config import load_config
from heartrisk.data.clean import load_clean
from heartrisk.data.download import sha256_bytes
from heartrisk.data.schema import SITE_LABELS
from heartrisk.features import build_pipeline, feature_frame

log = logging.getLogger(__name__)

# Precision and F1 are undefined when a heavily regularised candidate predicts no positives in a
# fold; score those as 0 explicitly instead of letting scikit-learn warn thousands of times.
SCORERS = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
}
SCORING = list(SCORERS)
HOLDOUT_METRICS = ["roc_auc", "accuracy", "precision", "recall", "specificity", "f1"]
# Types skops must be told to trust when loading the saved pipeline (checked with
# skops.io.get_untrusted_types); anything else in a model file is refused.
SKOPS_TRUSTED_TYPES = [
    "heartrisk.features.HeartRateReserve",
    "numpy.dtype",
    "sklearn.compose._column_transformer.make_column_selector",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.tree._tree.Tree",
]


def load_train_config(path: str | Path) -> dict[str, Any]:
    """Read configs/train.yaml."""
    return load_config(path)


def make_model(family: str, seed: int):
    """A fresh, seeded estimator for one model family (decision D4)."""
    if family == "logistic_regression":
        return LogisticRegression(solver="saga", max_iter=5000, random_state=seed)
    if family == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"unknown model family: {family}")


def select_champion(summary: pd.DataFrame, rule: dict[str, Any]) -> tuple[str, str]:
    """Decision D5. `summary` is indexed by family with cv_roc_auc_mean and cv_recall_mean.

    1. Families within `auc_tolerance` of the best mean CV ROC-AUC are tied.
    2. Among those, families within `recall_tolerance` of the best mean CV recall are tied.
    3. The first of those in `simplicity_order` wins.
    """
    eps = 1e-9  # 0.90 - 0.89 is 0.010000000000000009 in floating point
    best_auc = summary["cv_roc_auc_mean"].max()
    tied = summary[summary["cv_roc_auc_mean"] >= best_auc - rule["auc_tolerance"] - eps]
    if len(tied) == 1:
        return tied.index[0], (
            f"highest mean CV ROC-AUC ({best_auc:.3f}); no other family within "
            f"{rule['auc_tolerance']}"
        )
    best_recall = tied["cv_recall_mean"].max()
    near = tied[tied["cv_recall_mean"] >= best_recall - rule["recall_tolerance"] - eps]
    order = rule["simplicity_order"]
    champion = min(near.index, key=order.index)
    auc, recall = summary.loc[champion, ["cv_roc_auc_mean", "cv_recall_mean"]]
    others = ", ".join(
        f"{f} {summary.loc[f, 'cv_roc_auc_mean']:.3f}" for f in tied.index if f != champion
    )
    if len(near) == 1:
        return champion, (
            f"mean CV ROC-AUC {auc:.3f} ties with {others} (within {rule['auc_tolerance']} of the "
            f"best, {best_auc:.3f}), so recall decides: highest mean CV recall ({recall:.3f})"
        )
    return champion, (
        f"mean CV ROC-AUC {auc:.3f} and recall {recall:.3f} are within tolerance of the best; "
        f"simplest of {', '.join(sorted(near.index))}"
    )


def _param_grid(spec: dict, config: dict) -> dict[str, list]:
    grid = dict(spec["grid"])
    grid["engineer__enabled"] = list(config["features"]["hr_reserve"])
    return grid


def _cv_summary(search: GridSearchCV) -> dict[str, float]:
    best = search.best_index_
    results = search.cv_results_
    out = {}
    for metric in SCORING:
        out[f"cv_{metric}_mean"] = float(results[f"mean_test_{metric}"][best])
        out[f"cv_{metric}_std"] = float(results[f"std_test_{metric}"][best])
    return out


def _importance(pipeline, family: str, X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple:
    """(values, xlabel, signed): standardised coefficients for LR, permutation otherwise."""
    if family == "logistic_regression":
        names = pipeline[:-1].get_feature_names_out()
        values = pd.Series(pipeline["model"].coef_[0], index=names)
        return values, "Coefficient (log-odds per SD / per category)", True
    result = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=10, random_state=seed, n_jobs=1
    )
    values = pd.Series(result.importances_mean, index=X.columns)
    return values, "Drop in hold-out ROC-AUC when shuffled", False


def _log_figure(fig, name: str, save_dir: Path | None = None) -> None:
    mlflow.log_figure(fig, f"plots/{name}.png")
    if save_dir is not None:
        viz.save_fig(fig, name, save_dir)
    plt.close(fig)


def _log_candidates(family: str, search: GridSearchCV) -> None:
    """One nested run per hyperparameter candidate: its parameters and CV scores."""
    results = search.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"{family}-{i:03d}", nested=True):
            mlflow.set_tags({"run_kind": "candidate", "family": family})
            mlflow.log_params({k: str(v) for k, v in params.items()})
            metrics = {"rank_roc_auc": float(results["rank_test_roc_auc"][i])}
            for metric in SCORING:
                metrics[f"cv_{metric}_mean"] = float(results[f"mean_test_{metric}"][i])
                metrics[f"cv_{metric}_std"] = float(results[f"std_test_{metric}"][i])
            mlflow.log_metrics(metrics)


def _fit_family(family, spec, config, data, cv) -> dict[str, Any]:
    """Tune one family inside its own child run; log metrics, plots and the model."""
    seed, threshold = config["seed"], config["threshold"]
    X_train, y_train, X_test, y_test, external = data
    with mlflow.start_run(run_name=family, nested=True) as run:
        mlflow.set_tags({"run_kind": "family", "family": family})
        search = GridSearchCV(
            build_pipeline(make_model(family, seed)),
            param_grid=_param_grid(spec, config),
            scoring=SCORERS,
            refit=config["primary_metric"],
            cv=cv,
            n_jobs=config.get("n_jobs", -1),
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        _log_candidates(family, search)

        proba = best.predict_proba(X_test)[:, 1]
        holdout = evaluate.classification_metrics(y_test, proba, threshold)
        ext_rows = []
        for site, frame in external.groupby("site", sort=False):
            p_site = best.predict_proba(feature_frame(frame))[:, 1]
            metrics = evaluate.classification_metrics(
                frame["target"].astype(int), p_site, threshold
            )
            ext_rows.append({"family": family, "site": site, **metrics})
        cv_summary = _cv_summary(search)

        mlflow.log_params({f"best__{k}": str(v) for k, v in search.best_params_.items()})
        mlflow.log_params({"n_candidates": len(search.cv_results_["params"])})
        mlflow.log_metrics(cv_summary)
        mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout.items() if k in HOLDOUT_METRICS})
        mlflow.log_metrics(
            {
                f"external_{r['site']}_{k}": r[k]
                for r in ext_rows
                for k in ("roc_auc", "recall", "precision")
                if not np.isnan(r[k])
            }
        )
        label = evaluate.FAMILY_LABELS.get(family, family)
        _log_figure(
            evaluate.plot_roc({family: (y_test, proba)}, family, f"{label}: hold-out ROC"), "roc"
        )
        _log_figure(
            evaluate.plot_confusion(y_test, proba, threshold, f"{label}: hold-out"), "confusion"
        )
        _log_figure(
            evaluate.plot_calibration(y_test, proba, f"{label}: calibration"), "calibration"
        )
        values, xlabel, signed = _importance(best, family, X_test, y_test, seed)
        _log_figure(
            evaluate.plot_importance(values, f"{label}: importance", xlabel, signed), "importance"
        )
        mlflow.log_table(
            pd.DataFrame(search.cv_results_).drop(columns="params").astype(str), "cv_results.json"
        )
        mlflow.log_text(json.dumps(holdout, indent=2), "holdout_metrics.json")

        with warnings.catch_warnings():
            # MLflow hints that integer columns cannot hold missing values; the only integer
            # column is the predicted label. Inputs are float64 precisely so NaN is allowed.
            warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
            signature = infer_signature(X_train, best.predict(X_train))
            model_info = mlflow.sklearn.log_model(
                best,
                name="model",
                serialization_format="skops",
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
                signature=signature,
                input_example=X_train.dropna().head(3),
                pip_requirements=tracking.pip_requirements(),
            )
    return {
        "family": family,
        "run_id": run.info.run_id,
        "model_uri": model_info.model_uri,
        "estimator": best,
        "best_params": search.best_params_,
        "n_candidates": len(search.cv_results_["params"]),
        "cv": cv_summary,
        "holdout": holdout,
        "holdout_proba": proba,
        "importance": (values, xlabel, signed),
        "external": ext_rows,
    }


def _comparison(fits: dict[str, dict], champion: str) -> pd.DataFrame:
    rows = []
    for family, fit in fits.items():
        row = {"family": family, "n_candidates": fit["n_candidates"], **fit["cv"]}
        row.update({f"holdout_{k}": fit["holdout"][k] for k in HOLDOUT_METRICS})
        row["best_params"] = json.dumps(fit["best_params"], sort_keys=True, default=str)
        row["champion"] = family == champion
        rows.append(row)
    return pd.DataFrame(rows).round(6)


def _register(fit: dict, name: str, alias: str, tags: dict[str, str]) -> str:
    version = mlflow.register_model(fit["model_uri"], name).version
    client = MlflowClient()
    client.set_registered_model_alias(name, alias, version)
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    return version


def run(config: dict[str, Any], tracking_uri: str | None = None, register: bool = True) -> dict:
    """Train, evaluate, select and log; writes reports and returns a summary dict."""
    seed = config["seed"]
    clean_path = Path(config["data"]["clean_file"])
    df = load_clean(clean_path)
    train_sites = config["data"]["train_sites"]
    development = df[df["site"].isin(train_sites)]
    external = df[~df["site"].isin(train_sites)]
    X = feature_frame(development)
    y = development["target"].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], stratify=y, random_state=seed
    )
    cv = StratifiedKFold(n_splits=config["cv"]["folds"], shuffle=True, random_state=seed)
    reports = Path(config["outputs"]["reports_dir"])
    figures, metrics_dir = reports / "figures", reports / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    viz.apply_style()

    uri = tracking.configure(tracking_uri)
    tracking.ensure_experiment(config["tracking"]["experiment"], uri)
    manifest = Path(config["data"]["manifest"]) if config["data"].get("manifest") else None
    tags = tracking.lineage_tags(sha256_bytes(clean_path.read_bytes()), manifest, train_sites)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    log.info(
        "tracking %s | train %d / hold-out %d / external %d rows",
        uri,
        len(X_train),
        len(X_test),
        len(external),
    )

    with mlflow.start_run(run_name=f"train-{stamp}", tags={**tags, "run_kind": "parent"}) as parent:
        mlflow.log_params(
            {
                "seed": seed,
                "cv_folds": config["cv"]["folds"],
                "test_size": config["data"]["test_size"],
                "train_rows": len(X_train),
                "holdout_rows": len(X_test),
                "external_rows": len(external),
                "threshold": config["threshold"],
            }
        )
        with warnings.catch_warnings():
            # MLflow registers its local-file dataset source twice and warns that a path is
            # "ambiguous" between the two identical classes; nothing to act on.
            warnings.filterwarnings("ignore", message="The specified dataset source can be")
            # The logged dataset keeps its nullable-integer codes (gaps stay missing), which
            # triggers the same integer-column hint as the signature below.
            warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
            dataset = mlflow.data.from_pandas(
                development.astype({"record_id": object, "site": object}),
                source=str(clean_path),
                name="heart_clean-" + "+".join(train_sites),
                targets="target",
            )
            mlflow.log_input(dataset, context="training")
        if config.get("_path"):
            mlflow.log_artifact(config["_path"], "config")

        data = (X_train, y_train, X_test, y_test, external)
        fits = {}
        for family, spec in config["models"].items():
            log.info("tuning %s", family)
            fits[family] = _fit_family(family, spec, config, data, cv)
            f = fits[family]
            log.info(
                "  CV ROC-AUC %.3f ± %.3f | recall %.3f | hold-out ROC-AUC %.3f",
                f["cv"]["cv_roc_auc_mean"],
                f["cv"]["cv_roc_auc_std"],
                f["cv"]["cv_recall_mean"],
                f["holdout"]["roc_auc"],
            )

        summary = pd.DataFrame({k: v["cv"] for k, v in fits.items()}).T
        champion, reason = select_champion(summary, config["selection"])
        rest = summary.drop(index=champion)
        challenger = select_champion(rest, config["selection"])[0] if len(rest) else None
        log.info("champion: %s (%s)", champion, reason)
        for family, fit in fits.items():
            with mlflow.start_run(run_id=fit["run_id"], nested=True):
                mlflow.set_tag("champion", str(family == champion).lower())

        comparison = _comparison(fits, champion)
        comparison.to_csv(reports / "model_comparison.csv", index=False)
        external_df = pd.DataFrame([r for f in fits.values() for r in f["external"]]).round(6)
        external_df.to_csv(reports / "external_validation.csv", index=False)

        champ = fits[champion]
        recall_t = evaluate.threshold_for_recall(
            y_test, champ["holdout_proba"], config["recall_target"]
        )
        grid = sorted({*np.round(np.arange(0.1, 0.91, 0.05), 2), round(recall_t, 4)})
        thresholds = evaluate.threshold_table(y_test, champ["holdout_proba"], grid)
        thresholds.to_csv(reports / "threshold_analysis.csv", index=False)

        record = {
            "family": champion,
            "selection_reason": reason,
            "challenger": challenger,
            "best_params": {k: str(v) for k, v in champ["best_params"].items()},
            "cv": champ["cv"],
            "holdout": champ["holdout"],
            "external": {r["site"]: r for r in champ["external"]},
            "threshold": config["threshold"],
            "recall_target": config["recall_target"],
            "threshold_for_recall_target": recall_t,
            "parent_run_id": parent.info.run_id,
            "run_id": champ["run_id"],
            "lineage": tags,
        }
        (metrics_dir / "champion.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        for family, fit in fits.items():
            (metrics_dir / f"{family}.json").write_text(
                json.dumps(
                    {"cv": fit["cv"], "holdout": fit["holdout"], "external": fit["external"]},
                    indent=2,
                    default=str,
                )
                + "\n"
            )

        comparison_idx = comparison.set_index("family")
        curves = {family: (y_test, fit["holdout_proba"]) for family, fit in fits.items()}
        n_test = len(y_test)
        sets = [
            {
                "family": f,
                "set_label": f"Cleveland hold-out (n={n_test}, {y_test.mean():.0%})",
                "roc_auc": fits[f]["holdout"]["roc_auc"],
            }
            for f in fits
        ]
        sets += [
            {
                "family": r["family"],
                "roc_auc": r["roc_auc"],
                "set_label": (
                    f"{SITE_LABELS.get(r['site'], r['site'])} (n={r['n']}, {r['prevalence']:.0%})"
                ),
            }
            for r in external_df.to_dict("records")
        ]
        values, xlabel, signed = champ["importance"]
        label = evaluate.FAMILY_LABELS.get(champion, champion)
        _log_figure(
            evaluate.plot_cv_comparison(comparison_idx, champion), "model_01_cv_comparison", figures
        )
        _log_figure(
            evaluate.plot_roc(curves, champion, "Hold-out ROC (Cleveland)"),
            "model_02_roc_holdout",
            figures,
        )
        _log_figure(
            evaluate.plot_confusion(
                y_test,
                champ["holdout_proba"],
                config["threshold"],
                f"Champion ({label}) on the hold-out set",
            ),
            "model_03_confusion_holdout",
            figures,
        )
        _log_figure(
            evaluate.plot_calibration(
                y_test, champ["holdout_proba"], f"Champion calibration ({label})"
            ),
            "model_04_calibration_holdout",
            figures,
        )
        _log_figure(
            evaluate.plot_importance(
                values, f"Champion feature importance ({label})", xlabel, signed
            ),
            "model_05_importance",
            figures,
        )
        _log_figure(
            evaluate.plot_external(pd.DataFrame(sets), champion),
            "model_06_external_validation",
            figures,
        )
        _log_figure(
            evaluate.plot_threshold(thresholds, config["threshold"], recall_t),
            "model_07_threshold",
            figures,
        )
        for path in [
            reports / "model_comparison.csv",
            reports / "external_validation.csv",
            reports / "threshold_analysis.csv",
            metrics_dir / "champion.json",
        ]:
            mlflow.log_artifact(str(path), "reports")
        mlflow.set_tags({"champion": champion, "selection_reason": reason})

    versions = {}
    if register:
        name = config["tracking"]["registered_model"]
        base = {"data_sha256": tags["data_sha256"], "git_sha": tags["git_sha"]}
        versions["champion"] = _register(
            champ,
            name,
            "champion",
            {**base, "family": champion, "selection_reason": reason, "run_id": champ["run_id"]},
        )
        if challenger:
            versions["challenger"] = _register(
                fits[challenger],
                name,
                "challenger",
                {
                    **base,
                    "family": challenger,
                    "selection_reason": "runner-up under the D5 rule",
                    "run_id": fits[challenger]["run_id"],
                },
            )
        log.info("registered %s: %s", name, versions)
    return {
        "champion": champion,
        "challenger": challenger,
        "reason": reason,
        "parent_run_id": parent.info.run_id,
        "comparison": comparison,
        "versions": versions,
    }


def _quiet_third_party_noise() -> None:
    """Silence notices raised inside MLflow itself, so the training log shows our messages."""
    warnings.filterwarnings("ignore", message="Any type hint is inferred as AnyType")
    warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="mlflow")
    warnings.filterwarnings("ignore", module="mlflow", message=".*pandas.*")
    for name in ("mlflow.types.type_hints", "mlflow.store.db.utils", "alembic"):
        logging.getLogger(name).setLevel(logging.WARNING if name == "alembic" else logging.ERROR)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train, evaluate and register the classifier")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--tracking-uri", help="default: $MLFLOW_TRACKING_URI or local SQLite")
    parser.add_argument("--no-register", action="store_true", help="skip the model registry")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    _quiet_third_party_noise()
    config = load_train_config(args.config)
    config["_path"] = args.config
    result = run(config, tracking_uri=args.tracking_uri, register=not args.no_register)
    log.info("done: champion %s, parent run %s", result["champion"], result["parent_run_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
