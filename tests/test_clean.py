"""Validation gate and cleaning, on the hand-made rows in tests/fixtures/raw.

Expected values are read off the fixture files by hand (see the comments per test).
"""

import hashlib
import json

import pandas as pd
import pytest
import yaml

from heartrisk.data import clean

CLEAN_COLUMNS = [
    "record_id", "site", "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num_raw", "target",
]  # fmt: skip


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "processed"


def _run(config, raw_dir, out_dir):
    report = clean.prepare(config, raw_dir, out_dir)
    return report, clean.load_clean(out_dir / "heart_clean.csv").set_index("record_id")


def test_question_mark_is_parsed_as_missing(data_config, raw_dir, out_dir):
    _, df = _run(data_config, raw_dir, out_dir)

    assert pd.isna(df.loc["cleveland-003", "ca"])  # "...,2.0,?,3.0,1"
    assert pd.isna(df.loc["hungary-002", "chol"])  # "40,1,4,120,?,..."
    assert df.loc["cleveland-002", "ca"] == 3


def test_zero_cholesterol_and_blood_pressure_become_missing(data_config, raw_dir, out_dir):
    report, df = _run(data_config, raw_dir, out_dir)

    assert df.loc[["switzerland-001", "switzerland-002"], "chol"].isna().all()
    assert pd.isna(df.loc["va-002", "trestbps"])
    assert df.loc["va-002", "oldpeak"] == 0  # a zero ST depression is a real value
    assert df.loc["va-002", "fbs"] == 0
    assert report["sites"]["switzerland"]["zeros_recoded"] == {"trestbps": 0, "chol": 2}
    assert report["sites"]["va"]["zeros_recoded"] == {"trestbps": 1, "chol": 0}


def test_target_is_one_exactly_when_num_is_positive(data_config, raw_dir, out_dir):
    _, df = _run(data_config, raw_dir, out_dir)

    pairs = {rid: (df.loc[rid, "num_raw"], df.loc[rid, "target"]) for rid in df.index}
    assert pairs["cleveland-001"] == (0, 0)
    assert pairs["cleveland-002"] == (2, 1)
    assert pairs["cleveland-003"] == (1, 1)
    assert pairs["switzerland-002"] == (3, 1)
    assert pairs["va-002"] == (0, 0)


def test_record_id_traces_each_row_to_its_raw_line(data_config, raw_dir, out_dir):
    _, df = _run(data_config, raw_dir, out_dir)

    assert list(df.index) == [
        "cleveland-001", "cleveland-002", "cleveland-003", "hungary-001", "hungary-002",
        "switzerland-001", "switzerland-002", "va-001", "va-002",
    ]  # fmt: skip
    assert df.loc["cleveland-002", "age"] == 67
    assert df.loc["switzerland-002", "oldpeak"] == -1.5
    assert df.loc["hungary-001", "site"] == "hungary"


def test_clean_file_has_fixed_columns_and_integer_codes(data_config, raw_dir, out_dir):
    clean.prepare(data_config, raw_dir, out_dir)

    df = clean.load_clean(out_dir / "heart_clean.csv")
    assert list(df.columns) == CLEAN_COLUMNS
    assert str(df["ca"].dtype) == "Int64"
    assert str(df["oldpeak"].dtype) == "float64"
    lines = (out_dir / "heart_clean.csv").read_text().splitlines()
    # Codes are written as integers; a missing value is an empty field (ca in row 3).
    assert "cleveland-001,cleveland,63,1,1,145,233,1,2,150,0,2.3,3,0,6,0,0" in lines
    assert "cleveland-003,cleveland,56,0,2,140,294,0,2,153,0,1.3,2,,3,1,1" in lines


@pytest.mark.parametrize(
    ("bad_line", "column"),
    [
        ("70,1,4,150,900,0,1,120,1,1,2,?,?,1", "chol"),  # cholesterol above 700
        ("12,1,4,150,250,0,1,120,1,1,2,?,?,1", "age"),  # age below 18
        ("70,0.5,4,150,250,0,1,120,1,1,2,?,?,1", "sex"),  # not a valid code
        ("70,1,4,150,250,0,1,120,1,1,2,?,5,1", "thal"),  # thal code 5 does not exist
        ("70,1,4,150,250.5,0,1,120,1,1,2,?,?,1", "chol"),  # not integer-valued
        ("70,1,4,150,250,0,1,120,1,1,2,?,?,?", "num"),  # missing diagnosis
    ],
)
def test_invalid_row_is_rejected_with_a_reason_naming_the_column(
    data_config, make_raw_dir, out_dir, bad_line, column
):
    data_config["sites"]["va"]["expected_rows"] = 3
    data_config["gate"]["max_rejection_rate"] = 0.5
    raw_dir = make_raw_dir({"va": [bad_line]})

    report, df = _run(data_config, raw_dir, out_dir)

    rejected = pd.read_csv(out_dir / "rejected.csv")
    assert list(rejected["record_id"]) == ["va-003"]
    assert column in rejected.loc[0, "reason"]
    assert "va-003" not in df.index
    assert report["sites"]["va"]["rows_rejected"] == 1


def test_gate_fails_when_a_site_rejection_rate_exceeds_the_threshold(
    data_config, make_raw_dir, out_dir
):
    data_config["sites"]["va"]["expected_rows"] = 3
    raw_dir = make_raw_dir({"va": ["70,1,4,150,900,0,1,120,1,1,2,?,?,1"]})  # 1 of 3 > 5%

    with pytest.raises(clean.GateError, match="va"):
        clean.prepare(data_config, raw_dir, out_dir)

    report = json.loads((out_dir / "validation_report.json").read_text())
    assert report["status"] == "fail"
    assert (out_dir / "rejected.csv").exists()
    assert not (out_dir / "heart_clean.csv").exists()


def test_gate_fails_when_row_count_differs_from_config(data_config, raw_dir, out_dir):
    data_config["sites"]["cleveland"]["expected_rows"] = 4

    with pytest.raises(clean.GateError, match="cleveland"):
        clean.prepare(data_config, raw_dir, out_dir)


def test_gate_refuses_raw_files_that_do_not_match_the_manifest(data_config, raw_dir, out_dir):
    (raw_dir / "hungary.data").write_text("28,1,2,130,132,0,2,185,0,0,?,?,?,1\n")

    with pytest.raises(clean.GateError, match="hungary.data"):
        clean.prepare(data_config, raw_dir, out_dir)


def test_non_numeric_raw_file_fails_the_gate_naming_the_file(data_config, make_raw_dir, out_dir):
    # e.g. an HTML error page saved (and pinned) instead of the data file
    data_config["sites"]["va"]["expected_rows"] = 3
    raw_dir = make_raw_dir({"va": ["<html><body>Service Unavailable</body></html>"]})

    with pytest.raises(clean.GateError, match="va.data"):
        clean.prepare(data_config, raw_dir, out_dir)


def test_failed_gate_removes_stale_clean_output(data_config, make_raw_dir, out_dir):
    clean.prepare(data_config, make_raw_dir(), out_dir)
    assert (out_dir / "heart_clean.csv").exists()
    data_config["sites"]["va"]["expected_rows"] = 3
    bad_raw = make_raw_dir({"va": ["70,1,4,150,900,0,1,120,1,1,2,?,?,1"]}, name="raw_bad")

    with pytest.raises(clean.GateError):
        clean.prepare(data_config, bad_raw, out_dir)
    assert not (out_dir / "heart_clean.csv").exists()


def test_exact_duplicates_are_reported_but_kept(data_config, make_raw_dir, out_dir):
    data_config["sites"]["hungary"]["expected_rows"] = 3
    raw_dir = make_raw_dir({"hungary": ["28,1,2,130,132,0,2,185,0,0,?,?,?,0"]})  # = line 1

    report, df = _run(data_config, raw_dir, out_dir)

    assert report["sites"]["hungary"]["exact_duplicates"] == 1
    assert {"hungary-001", "hungary-003"} <= set(df.index)


def test_report_counts_rows_missingness_and_checksums(data_config, raw_dir, out_dir):
    report = clean.prepare(data_config, raw_dir, out_dir)

    assert report["status"] == "pass"
    assert report["totals"] == {"rows_read": 9, "rows_accepted": 9, "rows_rejected": 0}
    assert report["sites"]["switzerland"]["missing_pct"]["chol"] == 100.0  # both zeros
    assert report["sites"]["hungary"]["missing_pct"]["chol"] == 50.0  # one "?" of two
    assert report["sites"]["cleveland"]["disease_pct"] == pytest.approx(200 / 3, abs=0.05)
    raw_sha = hashlib.sha256((raw_dir / "va.data").read_bytes()).hexdigest()
    out_sha = hashlib.sha256((out_dir / "heart_clean.csv").read_bytes()).hexdigest()
    assert report["sites"]["va"]["input_sha256"] == raw_sha
    assert report["output"]["sha256"] == out_sha


def test_rerun_produces_byte_identical_outputs(data_config, raw_dir, tmp_path):
    names = ["heart_clean.csv", "validation_report.json", "DATA_CARD.md", "rejected.csv"]
    clean.prepare(data_config, raw_dir, tmp_path / "a")
    clean.prepare(data_config, raw_dir, tmp_path / "b")

    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_data_card_names_every_site_and_the_output_checksum(data_config, raw_dir, out_dir):
    report = clean.prepare(data_config, raw_dir, out_dir)

    card = (out_dir / "DATA_CARD.md").read_text()
    for label in ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]:
        assert label in card
    assert report["output"]["sha256"] in card
    assert "CC BY 4.0" in card


def test_cli_exit_code_reflects_the_gate(data_config, make_raw_dir, tmp_path):
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(data_config))
    good = ["--config", str(config_path), "--raw-dir", str(make_raw_dir()), "--out-dir"]
    assert clean.main([*good, str(tmp_path / "ok")]) == 0

    data_config["sites"]["cleveland"]["expected_rows"] = 99
    config_path.write_text(yaml.safe_dump(data_config))
    assert clean.main([*good, str(tmp_path / "fail")]) == 1
