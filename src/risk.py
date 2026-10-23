"""Patient-level risk stratification from predicted probabilities."""
import numpy as np
import pandas as pd

from .config import RISK_BINS, RISK_LABELS


def assign_tiers(prob) -> pd.Series:
    bins = list(RISK_BINS)
    bins[0], bins[-1] = -1e-9, 1 + 1e-9
    return pd.cut(np.asarray(prob), bins=bins, labels=RISK_LABELS)


def tier_table(prob, y_true) -> pd.DataFrame:
    """Number of patients and observed impairment rate in each tier."""
    df = pd.DataFrame({"tier": assign_tiers(prob), "y": np.asarray(y_true)})
    out = df.groupby("tier", observed=False)["y"].agg(n="count", impaired="sum")
    out["impaired_rate"] = (out["impaired"] / out["n"]).round(3)
    return out
