"""CI quality gate (D9): the pipeline fails when the champion is not good enough."""

import json

import pytest

from heartrisk import gate

CHAMPION = {
    "family": "logistic_regression",
    "selection_reason": "best CV ROC-AUC",
    "cv": {"cv_roc_auc_mean": 0.909, "cv_roc_auc_std": 0.011, "cv_recall_mean": 0.819},
    "holdout": {"n": 61, "prevalence": 0.46, "roc_auc": 0.95, "recall": 0.89, "precision": 0.81},
    "external": {
        "hungary": {"n": 294, "prevalence": 0.36, "roc_auc": 0.89, "recall": 0.62},
        "va": {"n": 200, "prevalence": 0.74, "roc_auc": float("nan"), "recall": 0.48},
    },
}


@pytest.mark.parametrize(("roc_auc", "passes"), [(0.95, True), (0.80, True), (0.7999, False)])
def test_gate_passes_at_or_above_the_minimum_hold_out_roc_auc(roc_auc, passes):
    champion = {**CHAMPION, "holdout": {**CHAMPION["holdout"], "roc_auc": roc_auc}}

    ok, message = gate.check(champion, min_roc_auc=0.80)

    assert ok is passes
    assert f"{roc_auc:.3f}" in message


def test_summary_lists_every_evaluation_set_and_marks_missing_auc():
    text = gate.summary_markdown(CHAMPION, ok=True, min_roc_auc=0.80)

    assert "Cleveland hold-out" in text
    assert "Hungary" in text and "VA Long Beach" in text
    assert "n/a" in text  # the NaN ROC-AUC is shown as not available, not as "nan"


def test_cli_exit_code_and_step_summary(tmp_path):
    champion_file = tmp_path / "champion.json"
    summary_file = tmp_path / "summary.md"
    champion_file.write_text(json.dumps(CHAMPION))
    args = ["--champion", str(champion_file), "--summary", str(summary_file)]

    assert gate.main([*args, "--min-roc-auc", "0.80"]) == 0
    assert gate.main([*args, "--min-roc-auc", "0.99"]) == 1
    assert summary_file.read_text().count("Quality gate") == 2  # appended, not overwritten
