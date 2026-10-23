"""Sanity tests: cohort construction, leakage control and model outputs.
Run with:  pytest -q
"""
import numpy as np

from src import data, models
from src.config import FEATURES
from src.risk import assign_tiers


def _cohort():
    raw = data.load_raw()
    return raw, data.build_cohort(raw)


def test_cohort_counts():
    raw, df = _cohort()
    assert len(raw) == 436                       # 416 subjects + 20 re-scans
    assert raw["ID"].str.endswith("_MR1").sum() == 416
    assert len(df) == 235
    assert df["target"].sum() == 100             # CDR >= 0.5
    assert (df["target"] == 0).sum() == 135      # CDR = 0


def test_no_target_leakage():
    assert "CDR" not in FEATURES
    assert "target" not in FEATURES


def test_split_is_stratified_and_disjoint():
    _, df = _cohort()
    X_tr, X_te, y_tr, y_te = data.split(df)
    assert len(X_tr) == 176 and len(X_te) == 59
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02


def test_models_output_valid_probabilities():
    _, df = _cohort()
    X_tr, X_te, y_tr, _ = data.split(df)
    for name, m in models.build_models(y_tr).items():
        p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        assert p.shape == (len(X_te),), name
        assert np.all((p >= 0) & (p <= 1)), name


def test_risk_tiers():
    tiers = list(assign_tiers([0.0, 0.29, 0.30, 0.45, 0.61, 1.0]))
    assert tiers == ["Low", "Low", "Low", "Medium", "High", "High"]
