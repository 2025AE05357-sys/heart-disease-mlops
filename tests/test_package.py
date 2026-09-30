"""Packaging: the exported champion is the registered artifact, traceable and reproducible."""

import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow import MlflowClient

from heartrisk import model_io, package
from heartrisk.data.clean import load_clean
from heartrisk.features import feature_frame

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def packaged(trained, tmp_path_factory):
    result, config, uri = trained
    out = tmp_path_factory.mktemp("package")
    lineage = package.export_champion(
        config, out / "champion", out / "MODEL_CARD.md", tracking_uri=uri
    )
    return lineage, out, config, uri, result


def test_exported_folder_is_the_registered_champion_version(packaged):
    lineage, out, config, uri, result = packaged
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    champion = client.get_model_version_by_alias(config["tracking"]["registered_model"], "champion")

    assert lineage["model"]["version"] == int(champion.version)
    assert lineage["model"]["family"] == result["champion"]
    sha = hashlib.sha256((out / "champion" / "model.skops").read_bytes()).hexdigest()
    assert lineage["model"]["file_sha256"] == sha


def test_exported_model_predicts_exactly_like_the_registry_copy(packaged):
    _, out, config, uri, _ = packaged
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    name = config["tracking"]["registered_model"]
    from_registry = mlflow.sklearn.load_model(f"models:/{name}@champion")
    exported = model_io.load_model(out / "champion")
    X = feature_frame(load_clean(FIXTURES / "heart_sample.csv"))

    np.testing.assert_array_equal(exported.predict_proba(X), from_registry.predict_proba(X))


def test_lineage_ties_the_model_to_code_data_and_metrics(packaged):
    lineage = packaged[0]
    sample_sha = hashlib.sha256((FIXTURES / "heart_sample.csv").read_bytes()).hexdigest()

    assert lineage["data"]["clean_file_sha256"] == sample_sha
    assert lineage["data"]["rows"]["train"] + lineage["data"]["rows"]["holdout"] == 76
    assert lineage["data"]["rows"]["external"] == {"hungary": 15, "switzerland": 15, "va": 15}
    assert lineage["reproducibility_check"]["identical"] is True
    assert lineage["code"]["git_sha"]
    assert set(lineage["metrics"]["holdout_by_sex"]) == {"female", "male"}
    assert lineage["decision"]["risk_bands"] == {"low": 0.3, "high": 0.6}


def test_model_card_covers_use_limits_and_every_evaluation_set(packaged):
    lineage, out = packaged[0], packaged[1]
    card = (out / "MODEL_CARD.md").read_text()

    assert "Not a medical device" in card
    for label in [
        "Cleveland hold-out",
        "Hungary",
        "Switzerland",
        "VA Long Beach",
        "Female",
        "Male",
    ]:
        assert label in card
    assert lineage["model"]["file_sha256"] in card


def test_packaging_refuses_data_that_changed_after_training(trained, tmp_path):
    _, config, uri = trained
    changed = tmp_path / "heart_clean.csv"
    changed.write_bytes((FIXTURES / "heart_sample.csv").read_bytes().replace(b",63,", b",64,", 1))
    edited = {**config, "data": {**config["data"], "clean_file": str(changed)}}

    with pytest.raises(package.LineageError, match="changed since training"):
        package.export_champion(edited, tmp_path / "champion", tracking_uri=uri)
