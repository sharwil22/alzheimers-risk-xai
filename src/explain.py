"""SHAP explanations for the tree-based XGBoost model."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

PRETTY = {"Educ": "Education"}


def shap_values(pipeline, X):
    """Exact TreeSHAP values (log-odds scale) for an imputer + XGBoost pipeline."""
    Xi = pd.DataFrame(pipeline.named_steps["impute"].transform(X), columns=X.columns, index=X.index)
    explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
    return explainer, explainer.shap_values(Xi), Xi


def global_importance(sv, columns) -> dict:
    """Mean |SHAP| per feature, sorted descending."""
    s = pd.Series(np.abs(sv).mean(axis=0), index=columns).sort_values(ascending=False)
    return {k: round(float(v), 4) for k, v in s.items()}


def plot_summary(sv, Xi, path):
    plt.figure()
    shap.summary_plot(sv, Xi.rename(columns=PRETTY), show=False, plot_size=(4.2, 3.0), color_bar_label="Feature value")
    fig = plt.gcf()
    for a in fig.axes:
        a.tick_params(labelsize=8)
    fig.axes[0].set_xlabel("SHAP value (impact on XGBoost output)", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=300, facecolor="white"); plt.close()


def plot_bar(importance: dict, path):
    s = pd.Series(importance).rename(index=PRETTY)[::-1]
    fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=300)
    ax.barh(s.index, s.values, color="#2d2a8f")
    for i, v in enumerate(s.values):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)


def plot_waterfall(explainer, sv, Xi, row: int, path):
    """Local explanation for one patient (row index into Xi)."""
    exp = shap.Explanation(values=sv[row], base_values=explainer.expected_value,
                           data=Xi.iloc[row].values, feature_names=[PRETTY.get(c, c) for c in Xi.columns])
    plt.figure()
    shap.plots.waterfall(exp, show=False, max_display=9)
    fig = plt.gcf(); fig.set_size_inches(5.6, 3.4)
    plt.tight_layout(); plt.savefig(path, dpi=300, facecolor="white"); plt.close()
