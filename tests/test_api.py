"""Prediction API: contract, validation, traceability headers, audit log, metrics.

Runs against the committed champion in models/champion, so it also proves that artifact loads.
"""

import json
import logging
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from heartrisk import model_io
from heartrisk.api import schemas
from heartrisk.api.app import create_app
from heartrisk.columns import FEATURES
from heartrisk.config import load_config

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "models" / "champion"
LINEAGE = json.loads((MODEL_DIR / "lineage.json").read_text())

# Rows 3 and 5 of the Cleveland file: num = 1 (disease) and num = 0 (no disease).
HIGH = {"age": 67, "sex": 1, "cp": 4, "trestbps": 120, "chol": 229, "fbs": 0, "restecg": 2,
        "thalach": 129, "exang": 1, "oldpeak": 2.6, "slope": 2, "ca": 2, "thal": 7}  # fmt: skip
LOW = {"age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204, "fbs": 0, "restecg": 2,
       "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 3}  # fmt: skip


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(MODEL_DIR)) as c:
        yield c


def test_health_does_not_depend_on_the_model(tmp_path):
    with TestClient(create_app(tmp_path / "no-model")) as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=HIGH).status_code == 503


def test_ready_names_the_loaded_model_version(client):
    body = client.get("/ready").json()

    assert body == {"status": "ready", "model_version": LINEAGE["model"]["version"]}


def test_predict_returns_class_probability_band_and_model_identity(client):
    body = client.post("/predict", json=HIGH).json()

    p = body["probability"]
    assert 0 <= p <= 1
    assert body["prediction"] == int(p >= 0.5)
    assert body["label"] == ("disease" if body["prediction"] else "no_disease")
    assert body["confidence"] == pytest.approx(max(p, 1 - p))
    assert body["risk_band"] == model_io.risk_band(p, low=0.3, high=0.6)
    assert body["imputed_fields"] == []
    assert body["model"]["version"] == LINEAGE["model"]["version"]
    assert body["model"]["run_id"] == LINEAGE["model"]["run_id"]


def test_probability_equals_the_pipeline_on_the_same_row(client):
    # Catches any mix-up between JSON field names and the model's column order.
    pipeline = model_io.load_model(MODEL_DIR)
    frame = pd.DataFrame([HIGH, LOW])[FEATURES].astype("float64")
    expected = pipeline.predict_proba(frame)[:, 1]

    got = [client.post("/predict", json=row).json()["probability"] for row in (HIGH, LOW)]

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    assert got[0] > got[1]  # the clinically high-risk row scores higher


def test_missing_optional_measurements_are_imputed_and_reported(client):
    partial = {k: v for k, v in HIGH.items() if k not in ("ca", "thal", "chol")}

    body = client.post("/predict", json=partial).json()

    assert body["imputed_fields"] == ["chol", "ca", "thal"]  # in model column order


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"age": 150}, "age"),
        ({"cp": 5}, "cp"),
        ({"chol": 5}, "chol"),
        ({"oldpeak": "high"}, "oldpeak"),
        ({"shoe_size": 42}, "shoe_size"),
    ],
)
def test_invalid_input_is_a_422_that_names_the_field(client, change, field):
    response = client.post("/predict", json={**HIGH, **change})

    assert response.status_code == 422
    assert field in json.dumps(response.json()["detail"])


def test_missing_required_field_is_a_422(client):
    response = client.post("/predict", json={k: v for k, v in HIGH.items() if k != "age"})

    assert response.status_code == 422
    assert "age" in json.dumps(response.json()["detail"])


def test_request_id_is_propagated_and_model_version_is_a_header(client):
    response = client.post("/predict", json=LOW, headers={"X-Request-ID": "req-123"})

    assert response.headers["X-Request-ID"] == "req-123"
    assert response.json()["request_id"] == "req-123"
    assert response.headers["X-Model-Version"] == str(LINEAGE["model"]["version"])


def test_request_id_is_generated_when_absent_or_unsafe(client):
    generated = client.post("/predict", json=LOW).headers["X-Request-ID"]
    unsafe = client.post("/predict", json=LOW, headers={"X-Request-ID": "a b\n<script>"})

    assert len(generated) >= 16
    assert unsafe.headers["X-Request-ID"] != "a b\n<script>"


def test_batch_scores_each_patient_in_order(client):
    single = [client.post("/predict", json=row).json()["probability"] for row in (LOW, HIGH)]

    body = client.post("/predict/batch", json={"patients": [LOW, HIGH]}).json()

    batch = [p["probability"] for p in body["predictions"]]
    np.testing.assert_allclose(batch, single, rtol=0, atol=1e-12)
    assert body["model"]["version"] == LINEAGE["model"]["version"]


def test_batch_rejects_more_than_100_patients(client):
    response = client.post("/predict/batch", json={"patients": [LOW] * 101})

    assert response.status_code == 422


def _audit_payloads(caplog):
    return [r.payload for r in caplog.records if r.name == "heartrisk.audit"]


def test_each_prediction_writes_one_audit_record_without_raw_values(client, caplog):
    caplog.set_level(logging.INFO, logger="heartrisk.audit")
    client.post("/predict", json=HIGH, headers={"X-Request-ID": "audit-1"})
    client.post("/predict", json=HIGH, headers={"X-Request-ID": "audit-2"})
    client.post("/predict", json=LOW, headers={"X-Request-ID": "audit-3"})

    records = _audit_payloads(caplog)
    assert [r["request_id"] for r in records] == ["audit-1", "audit-2", "audit-3"]
    assert records[0]["input_sha256"] == records[1]["input_sha256"]  # same input, same hash
    assert records[0]["input_sha256"] != records[2]["input_sha256"]
    assert records[0]["model_version"] == LINEAGE["model"]["version"]
    assert "input" not in records[0]
    numbers = [v for v in records[0].values() if isinstance(v, int | float)]
    assert 229 not in numbers  # the raw cholesterol value never reaches the log


def test_raw_inputs_are_audited_only_when_enabled(caplog):
    caplog.set_level(logging.INFO, logger="heartrisk.audit")
    with TestClient(create_app(MODEL_DIR, audit_inputs=True)) as c:
        c.post("/predict", json=HIGH)

    assert _audit_payloads(caplog)[-1]["input"]["chol"] == 229


def test_model_info_is_the_lineage_record(client):
    assert client.get("/model-info").json() == LINEAGE


def test_metrics_expose_prediction_and_model_series(client):
    client.post("/predict", json=HIGH)

    text = client.get("/metrics").text

    assert "heartrisk_predictions_total" in text
    assert "heartrisk_prediction_probability_bucket" in text
    assert "heartrisk_http_requests_total" in text
    assert f'version="{LINEAGE["model"]["version"]}"' in text


def _bounds(field_name):
    info = schemas.Patient.model_fields[field_name]
    low = next(m.ge for m in info.metadata if hasattr(m, "ge"))
    high = next(m.le for m in info.metadata if hasattr(m, "le"))
    return [low, high]


def _codes(field_name):
    annotation = schemas.Patient.model_fields[field_name].annotation
    literals = [a for a in typing.get_args(annotation) if typing.get_origin(a) is typing.Literal]
    return sorted(typing.get_args(literals[0] if literals else annotation))


def test_api_validation_matches_the_data_gate_rules():
    # One definition of "plausible": the API must accept exactly what the validation gate does.
    config = load_config(REPO / "configs" / "data.yaml")
    for field, bounds in config["ranges"].items():
        assert _bounds(field) == bounds, field
    for field, codes in config["codes"].items():
        if field != "num":
            assert _codes(field) == codes, field
