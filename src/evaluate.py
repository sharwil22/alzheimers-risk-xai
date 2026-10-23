"""Metrics, cross-validation, bootstrap confidence intervals and plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import CV_FOLDS, MODEL_COLORS, SEED

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def test_metrics(y_true, prob, threshold=0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def cv_auc(model, X, y) -> tuple:
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    s = cross_val_score(clone(model), X, y, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def bootstrap_auc_ci(y_true, prob, n=2000, alpha=0.05) -> tuple:
    """Percentile bootstrap 95% CI for test-set ROC-AUC."""
    rng = np.random.default_rng(SEED)
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], prob[idx]))
    return float(np.percentile(aucs, 100 * alpha / 2)), float(np.percentile(aucs, 100 * (1 - alpha / 2)))


def plot_roc(y_true, probs: dict, metrics: dict, path):
    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, lw=1.8, color=MODEL_COLORS[name], label=f"{name} (AUC {metrics[name]['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="#9aa7bb", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=6.3, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)


def plot_confusion(cm, path, title=None):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(2.5, 2.3), dpi=300)
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#eef0ff", "#2d2a8f"]))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b2149")
    ax.set_xticks([0, 1], ["Normal", "Impaired"]); ax.set_yticks([0, 1], ["Normal", "Impaired"], rotation=90, va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    if title:
        ax.set_title(title, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)


def plot_calibration(y_true, probs: dict, path):
    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)
    for name, p in probs.items():
        frac, mean = calibration_curve(y_true, p, n_bins=5, strategy="quantile")
        ax.plot(mean, frac, "o-", ms=3, lw=1.4, color=MODEL_COLORS[name], label=name)
    ax.plot([0, 1], [0, 1], "--", color="#9aa7bb", lw=1)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction impaired")
    ax.legend(fontsize=6.3, frameon=False)
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)
