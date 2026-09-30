"""Evaluation metrics and thresholds, on examples small enough to check by hand."""

import math

import numpy as np
import pytest

from heartrisk import evaluate


def test_metrics_match_a_hand_worked_confusion_matrix():
    # threshold 0.5 -> predictions [0, 1, 0, 1]: TN, FP, FN, TP (one of each)
    # ROC-AUC: of the 4 positive/negative pairs, 3 are ranked correctly -> 0.75
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])

    m = evaluate.classification_metrics(y, p, threshold=0.5)

    assert m == {
        "n": 4,
        "prevalence": 0.5,
        "roc_auc": 0.75,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "specificity": 0.5,
        "f1": 0.5,
    }


def test_roc_auc_is_nan_not_an_error_when_only_one_class_is_present():
    m = evaluate.classification_metrics(np.array([1, 1, 1]), np.array([0.2, 0.7, 0.9]))

    assert math.isnan(m["roc_auc"])
    assert m["recall"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(("target", "expected"), [(0.75, 0.5), (0.9, 0.3), (0.5, 0.7)])
def test_threshold_for_recall_is_the_highest_threshold_reaching_the_target(target, expected):
    # positives scored 0.9, 0.7, 0.5, 0.3:
    # recall 0.5 needs t <= 0.7, recall 0.75 needs t <= 0.5, all four need t <= 0.3
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.35, 0.3, 0.5, 0.7, 0.9])

    assert evaluate.threshold_for_recall(y, p, target) == expected


@pytest.mark.parametrize(
    ("probability", "band"),
    [
        (0.0, "low"),
        (0.29, "low"),
        (0.3, "moderate"),
        (0.59, "moderate"),
        (0.6, "high"),
        (1.0, "high"),
    ],
)
def test_risk_band_boundaries(probability, band):
    assert evaluate.risk_band(probability, low=0.3, high=0.6) == band
