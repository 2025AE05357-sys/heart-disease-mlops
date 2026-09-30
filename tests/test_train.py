"""Training: the D5 selection rule, and an end-to-end run on a small real sample.

The sample (tests/fixtures/heart_sample.csv) is every 4th Cleveland row plus 15 rows from each
external site; the fast config keeps the search tiny so the whole file runs in seconds.
"""

import hashlib
import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from heartrisk import train

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"

RULE = {
    "auc_tolerance": 0.01,
    "recall_tolerance": 0.01,
    "simplicity_order": ["logistic_regression", "hist_gradient_boosting", "random_forest"],
}


def _summary(rows):
    return pd.DataFrame(rows, columns=["family", "cv_roc_auc_mean", "cv_recall_mean"]).set_index(
        "family"
    )


def test_clear_auc_winner_is_champion():
    summary = _summary(
        [
            ("logistic_regression", 0.80, 0.90),
            ("random_forest", 0.90, 0.70),
            ("hist_gradient_boosting", 0.85, 0.95),
        ]
    )

    champion, reason = train.select_champion(summary, RULE)

    assert champion == "random_forest"
    assert "ROC-AUC" in reason


def test_within_auc_tolerance_higher_recall_wins():
    summary = _summary([("logistic_regression", 0.895, 0.70), ("random_forest", 0.900, 0.80)])

    assert train.select_champion(summary, RULE)[0] == "random_forest"
    summary.loc["logistic_regression", "cv_recall_mean"] = 0.90
    assert train.select_champion(summary, RULE)[0] == "logistic_regression"


def test_auc_exactly_at_the_tolerance_still_counts_as_a_tie():
    # 0.90 - 0.89 is 0.010000000000000009 in floating point; it must still tie
    summary = _summary([("logistic_regression", 0.89, 0.85), ("random_forest", 0.90, 0.80)])

    assert train.select_champion(summary, RULE)[0] == "logistic_regression"


def test_tie_on_auc_and_recall_goes_to_the_simpler_model():
    summary = _summary([("random_forest", 0.900, 0.805), ("logistic_regression", 0.895, 0.800)])

    champion, reason = train.select_champion(summary, RULE)

    assert champion == "logistic_regression"
    assert "simpl" in reason


@pytest.fixture(scope="module")
def fast_config(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("train")
    config = train.load_train_config(REPO / "configs" / "train.yaml")
    config["n_jobs"] = 1  # process start-up costs more than it saves on 2-candidate grids
    config["data"]["clean_file"] = str(FIXTURES / "heart_sample.csv")
    config["data"]["manifest"] = str(tmp_path / "no-manifest.json")
    config["cv"]["folds"] = 3
    config["outputs"]["reports_dir"] = str(tmp_path / "reports")
    config["models"] = {
        "logistic_regression": {"grid": {"model__C": [1.0]}},
        "random_forest": {"grid": {"model__n_estimators": [25], "model__max_depth": [3]}},
        "hist_gradient_boosting": {"grid": {"model__max_leaf_nodes": [7]}},
    }
    config["features"]["hr_reserve"] = [False, True]  # 2 candidates per family
    return config


@pytest.fixture(scope="module")
def trained(fast_config, tmp_path_factory):
    uri = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"
    result = train.run(fast_config, tracking_uri=uri)
    return result, fast_config, uri


def test_run_writes_comparison_champion_and_external_validation(trained):
    result, config, _ = trained
    reports = Path(config["outputs"]["reports_dir"])

    comparison = pd.read_csv(reports / "model_comparison.csv")
    external = pd.read_csv(reports / "external_validation.csv")
    champion = json.loads((reports / "metrics" / "champion.json").read_text())

    assert sorted(comparison["family"]) == sorted(config["models"])
    assert comparison["champion"].sum() == 1
    assert champion["family"] == result["champion"]
    assert champion["selection_reason"]
    assert set(external["site"]) == {"hungary", "switzerland", "va"}
    assert len(external) == 3 * 3  # every family on every external site
    assert (reports / "figures" / "model_02_roc_holdout.png").exists()


def test_mlflow_has_parent_family_and_candidate_runs(trained):
    result, config, uri = trained
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[config["tracking"]["experiment"]])
    parent_id = result["parent_run_id"]
    children = runs[runs["tags.mlflow.parentRunId"] == parent_id]
    candidates = runs[runs["tags.mlflow.parentRunId"].isin(children["run_id"])]

    assert len(children) == 3
    assert len(candidates) == 3 * 2
    assert children["metrics.cv_roc_auc_mean"].notna().all()
    sha = hashlib.sha256((FIXTURES / "heart_sample.csv").read_bytes()).hexdigest()
    parent = runs[runs["run_id"] == parent_id].iloc[0]
    assert parent["tags.data_sha256"] == sha


def test_champion_and_challenger_are_registered_by_alias(trained):
    result, config, uri = trained
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    name = config["tracking"]["registered_model"]

    champion = client.get_model_version_by_alias(name, "champion")
    challenger = client.get_model_version_by_alias(name, "challenger")

    assert champion.tags["family"] == result["champion"]
    assert champion.tags["selection_reason"]
    assert champion.version != challenger.version


def test_retraining_with_the_same_seed_reproduces_the_results(trained, fast_config, tmp_path):
    first, _, _ = trained
    second = train.run(fast_config, tracking_uri=f"sqlite:///{tmp_path / 'again.db'}")

    assert first["champion"] == second["champion"]
    pd.testing.assert_frame_equal(first["comparison"], second["comparison"])
