"""Exploratory data analysis figures and tables."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FEATURES

LBL = {0: "Normal (CDR 0)", 1: "Impaired (CDR ≥ 0.5)"}
COL = {0: "#3b82f6", 1: "#c8413f"}


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD of each feature by class (for the report)."""
    g = df.groupby("target")[FEATURES]
    m, s = g.mean().T, g.std().T
    return pd.DataFrame({LBL[c]: [f"{m.loc[f, c]:.2f} ± {s.loc[f, c]:.2f}" for f in FEATURES] for c in (0, 1)},
                        index=FEATURES)


def plot_class_balance(df, path):
    cdr = df["CDR"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(3.2, 2.3), dpi=300)
    colors = ["#3b82f6"] + ["#c8413f"] * (len(cdr) - 1)
    bars = ax.bar([str(c) for c in cdr.index], cdr.values, color=colors)
    for b, v in zip(bars, cdr.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha="center", fontsize=8)
    ax.set_xlabel("CDR"); ax.set_ylabel("Subjects")
    ax.set_title("CDR distribution (blue = normal, red = impaired)", fontsize=7.5)
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)


def plot_distributions(df, path, feats=("MMSE", "nWBV", "Age", "eTIV")):
    fig, axes = plt.subplots(1, len(feats), figsize=(7.0, 2.0), dpi=300)
    for ax, f in zip(axes, feats):
        data = [df.loc[df.target == c, f].dropna() for c in (0, 1)]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(COL[c]); patch.set_alpha(0.75)
        ax.set_xticks([1, 2], ["Normal", "Impaired"], fontsize=7); ax.set_title(f, fontsize=8.5)
        ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)


def plot_correlation(df, path):
    cols = FEATURES + ["CDR"]
    c = df[cols].rename(columns={"Educ": "Education"}).corr()
    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=300)
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=60, fontsize=6.5, ha="right")
    ax.set_yticks(range(len(c)), c.columns, fontsize=6.5)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=4.6,
                    color="white" if abs(c.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)
    return c
