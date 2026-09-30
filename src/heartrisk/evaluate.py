"""Evaluation: metrics, thresholds, risk bands and the model figures (house style)."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from heartrisk import viz
from heartrisk.data.schema import CODE_LABELS, DESCRIPTIONS
from heartrisk.model_io import risk_band as risk_band  # re-export: one definition, API included

FAMILY_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting (HGB)",
}
_SHORT = {"cp": "Chest pain", "restecg": "Resting ECG", "slope": "ST slope", "thal": "Thallium"}


def pretty_feature(name: str) -> str:
    """Clinical label for a model feature: "thal_7.0" -> "Thallium: Reversible defect"."""
    column, _, code = name.rpartition("_")
    if column in CODE_LABELS and code.replace(".", "", 1).isdigit():
        return (
            f"{_SHORT.get(column, DESCRIPTIONS[column])}: {CODE_LABELS[column][int(float(code))]}"
        )
    if name == "sex":
        return "Sex (male)"
    if name == "hr_pct_pred":
        return "Share of age-predicted max heart rate"
    return DESCRIPTIONS.get(name, name)


# Non-champion curves are grey; line style tells them apart (the legend names each one).
_OTHER_STYLES = [(0, (6, 3)), (0, (1.5, 2))]


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Threshold metrics plus ROC-AUC (NaN when only one class is present)."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else math.nan,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else math.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def threshold_for_recall(y_true: np.ndarray, proba: np.ndarray, target: float) -> float:
    """The highest probability threshold whose recall is at least `target`."""
    positives = np.sort(np.asarray(proba)[np.asarray(y_true) == 1])[::-1]
    needed = math.ceil(target * len(positives) - 1e-9)
    return float(positives[max(needed, 1) - 1])


def threshold_table(y_true: np.ndarray, proba: np.ndarray, thresholds) -> pd.DataFrame:
    """Precision, recall, specificity and F1 at each threshold."""
    rows = []
    for t in thresholds:
        m = classification_metrics(y_true, proba, threshold=t)
        rows.append({"threshold": round(float(t), 4), **{k: m[k] for k in _THRESHOLD_COLUMNS}})
    return pd.DataFrame(rows)


_THRESHOLD_COLUMNS = ["precision", "recall", "specificity", "f1", "accuracy"]


# ---- figures -------------------------------------------------------------------------------


def _label(family: str, champion: str | None = None) -> str:
    name = FAMILY_LABELS.get(family, family)
    return f"{name} (champion)" if family == champion else name


def plot_cv_comparison(comparison: pd.DataFrame, champion: str) -> Figure:
    """Mean ± SD of CV ROC-AUC and recall per family; the champion in full ink."""
    families = list(comparison.index)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 0.9 + 0.75 * len(families)), sharey=True)
    ypos = np.arange(len(families))[::-1]
    for ax, metric, title in zip(axes, ["roc_auc", "recall"], ["ROC-AUC", "Recall"], strict=True):
        for y, family in zip(ypos, families, strict=True):
            mean = comparison.loc[family, f"cv_{metric}_mean"]
            sd = comparison.loc[family, f"cv_{metric}_std"]
            color = viz.INK if family == champion else viz.INK_MUTED
            ax.errorbar(mean, y, xerr=sd, fmt="o", color=color, markersize=8, elinewidth=2)
            ax.text(mean, y + 0.3, f"{mean:.3f}", ha="center", fontsize=11, color=viz.INK_SECONDARY)
        ax.set_title(title)
        ax.set_ylim(ypos.min() - 0.6, ypos.max() + 0.7)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [_label(f, champion) for f in families])
    fig.suptitle("Cross-validation on the training set (mean ± SD across folds)")
    return fig


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], champion: str, title: str) -> Figure:
    """ROC curves on one axis; the champion emphasised, the others grey with distinct dashes."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0, 1], [0, 1], color=viz.BASELINE, linewidth=1)
    others = iter(_OTHER_STYLES)
    for family, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        is_champion = family == champion
        ax.plot(
            fpr,
            tpr,
            color=viz.INK if is_champion else viz.INK_MUTED,
            linewidth=2.5 if is_champion else 1.8,
            linestyle="-" if is_champion else next(others),
            label=f"{_label(family, champion)}: AUC {auc:.3f}",
            zorder=3 if is_champion else 2,
        )
    ax.set_xlabel("False-positive rate (1 − specificity)")
    ax.set_ylabel("True-positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.set_title(title)
    return fig


def plot_confusion(y_true: np.ndarray, proba: np.ndarray, threshold: float, title: str) -> Figure:
    """2x2 confusion matrix with counts and the share of each actual class."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), pred, labels=[0, 1])
    share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(share, cmap=viz.SEQUENTIAL, vmin=0, vmax=1)
    for i, j in np.ndindex(cm.shape):
        ax.text(
            j,
            i,
            f"{cm[i, j]}\n{share[i, j]:.0%}",
            ha="center",
            va="center",
            fontsize=12,
            color="white" if share[i, j] >= 0.55 else viz.INK,
        )
    labels = [viz.CLASS_LABELS[0], viz.CLASS_LABELS[1]]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel(f"Predicted (threshold {threshold:g})")
    ax.set_ylabel("Actual")
    ax.grid(False)
    ax.set_title(title)
    return fig


def plot_calibration(y_true: np.ndarray, proba: np.ndarray, title: str, bins: int = 5) -> Figure:
    """Reliability curve (quantile bins) against the diagonal, with the Brier score."""
    observed, predicted = calibration_curve(y_true, proba, n_bins=bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot([0, 1], [0, 1], color=viz.BASELINE, linewidth=1, label="Perfect calibration")
    ax.plot(predicted, observed, "o-", color=viz.INK, markersize=8, label="Model")
    ax.text(
        0.03,
        0.95,
        f"Brier score {brier_score_loss(y_true, proba):.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color=viz.INK_SECONDARY,
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed share with disease")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.set_title(title)
    return fig


def plot_importance(values: pd.Series, title: str, xlabel: str, signed: bool) -> Figure:
    """Top features by absolute value; signed values coloured by the class they point to."""
    top = values.reindex(values.abs().sort_values(ascending=False).index)[:12][::-1]
    top.index = [pretty_feature(str(n)) for n in top.index]
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.36 * len(top)))
    colors = [viz.CLASS_COLORS[int(v > 0)] if signed else viz.NEUTRAL for v in top]
    ax.barh(range(len(top)), top.to_numpy(), color=colors, height=0.62)
    ax.set_yticks(range(len(top)), top.index)
    ax.axvline(0, color=viz.BASELINE, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    fig.suptitle(title)
    return fig


def plot_external(summary: pd.DataFrame, champion: str) -> Figure:
    """ROC-AUC per evaluation set (rows) and family (marks); champion filled, others hollow."""
    sets = list(dict.fromkeys(summary["set_label"]))
    families = list(dict.fromkeys(summary["family"]))
    offsets = dict(zip(families, np.linspace(0.18, -0.18, len(families)), strict=True))
    markers = iter(["s", "^", "D"])
    fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.65 * len(sets)))
    ypos = dict(zip(sets, np.arange(len(sets))[::-1], strict=True))
    for family in families:
        rows = summary[(summary["family"] == family) & summary["roc_auc"].notna()]
        is_champion = family == champion
        ax.plot(
            rows["roc_auc"],
            [ypos[s] + offsets[family] for s in rows["set_label"]],
            linestyle="none",
            marker="o" if is_champion else next(markers),
            markersize=10 if is_champion else 8,
            markerfacecolor=viz.INK if is_champion else "none",
            markeredgecolor=viz.INK if is_champion else viz.NEUTRAL,
            markeredgewidth=1.8,
            label=_label(family, champion),
            zorder=3 if is_champion else 2,
        )
    ax.axvline(0.5, color=viz.BASELINE, linewidth=1)
    ax.set_yticks(list(ypos.values()), list(ypos.keys()))
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("ROC-AUC (0.5 = chance)")
    ax.grid(axis="y", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=len(families))
    ax.set_title("ROC-AUC: hold-out set and external hospitals")
    return fig


def plot_threshold(table: pd.DataFrame, threshold: float, recall_threshold: float) -> Figure:
    """Precision and recall as the threshold moves; the class threshold and the recall target."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(table["threshold"], table["recall"], color=viz.INK, linewidth=2.5, label="Recall")
    ax.plot(
        table["threshold"],
        table["precision"],
        color=viz.INK_MUTED,
        linewidth=2,
        linestyle=(0, (6, 3)),
        label="Precision",
    )
    # Label the lower threshold on its left and the higher one on its right, so they never overlap.
    marks = sorted([(threshold, "class threshold"), (recall_threshold, "recall target")])
    for (x, name), side in zip(marks, ["right", "left"], strict=True):
        ax.axvline(x, color=viz.INK_SECONDARY, linewidth=1.2)
        text = f"{name} {x:.2f} " if side == "right" else f" {name} {x:.2f}"
        ax.text(x, 0.04, text, ha=side, fontsize=11, color=viz.INK_SECONDARY)
    ax.set_xlabel("Probability threshold")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center left")
    ax.set_title("Champion on the hold-out set: threshold trade-off")
    return fig


def precision_recall_points(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Every (threshold, precision, recall) point, for logging as a table."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    return pd.DataFrame(
        {"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]}
    )
