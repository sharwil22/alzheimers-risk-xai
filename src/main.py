"""End-to-end pipeline:  python -m src.main

Data -> cohort & target -> stratified split -> train 4 models (+2 benchmarks)
-> test metrics, 5-fold CV, bootstrap CIs -> SHAP -> risk tiers -> ablation.
All outputs go to results/ (metrics.json, CSV tables, figures/)."""
import json
import platform
from pathlib import Path

import joblib
import time

import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.base import clone

from . import data, eda, evaluate, explain, models, risk
from .config import COGNITIVE, DEMOGRAPHIC, FEATURES, FIG_DIR, MRI, RESULTS_DIR, SEED

CORE = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]


def run():
    np.random.seed(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = {"environment": {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__,
        "pandas": pd.__version__, "numpy": np.__version__}}

    # 1. Data inspection & cohort
    raw = data.load_raw()
    df = data.build_cohort(raw)
    out["dataset"] = data.describe(raw, df)
    eda.summary_table(df).to_csv(RESULTS_DIR / "feature_summary_by_class.csv")
    eda.plot_class_balance(df, FIG_DIR / "cdr_distribution.png")
    eda.plot_distributions(df, FIG_DIR / "feature_boxplots.png")
    corr = eda.plot_correlation(df, FIG_DIR / "correlation_heatmap.png")
    out["correlation_with_CDR"] = corr["CDR"].drop("CDR").round(3).to_dict()

    # 2. Split
    X_tr, X_te, y_tr, y_te = data.split(df)
    out["split"] = {"train": len(X_tr), "test": len(X_te),
                    "train_impaired": int(y_tr.sum()), "test_impaired": int(y_te.sum())}

    # 3. Train & evaluate
    zoo = models.build_models(y_tr, include_benchmarks=True)
    metrics, probs, times = {}, {}, {}
    for name, m in zoo.items():
        t0 = time.perf_counter(); m.fit(X_tr, y_tr); times[name] = time.perf_counter() - t0
        p = m.predict_proba(X_te)[:, 1]
        probs[name] = p
        met = evaluate.test_metrics(y_te, p)
        met["cv_auc_mean"], met["cv_auc_std"] = evaluate.cv_auc(m, df[FEATURES], df["target"])
        met["auc_95ci"] = evaluate.bootstrap_auc_ci(y_te, p)
        met["train_time_s"] = round(times[name], 4)
        metrics[name] = met
    out["metrics"] = metrics
    out["hyperparameters"] = models.HYPERPARAMS
    best = max(CORE, key=lambda k: metrics[k]["cv_auc_mean"])
    out["selected_model"] = best

    rows = [{"model": k, **{m: v for m, v in d.items() if m not in ("confusion_matrix", "auc_95ci")},
             "auc_ci_low": d["auc_95ci"][0], "auc_ci_high": d["auc_95ci"][1]} for k, d in metrics.items()]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    core_probs = {k: probs[k] for k in CORE}
    evaluate.plot_roc(y_te, core_probs, metrics, FIG_DIR / "roc_curves.png")
    evaluate.plot_confusion(metrics[best]["confusion_matrix"], FIG_DIR / "confusion_matrix_xgboost.png")
    evaluate.plot_calibration(y_te, core_probs, FIG_DIR / "calibration.png")

    # 4. Explainability (SHAP on XGBoost)
    explainer, sv, Xi = explain.shap_values(zoo["XGBoost"], X_te)
    imp = explain.global_importance(sv, X_te.columns)
    out["shap_mean_abs"] = imp
    explain.plot_summary(sv, Xi, FIG_DIR / "shap_summary.png")
    explain.plot_bar(imp, FIG_DIR / "shap_importance_bar.png")
    # local example: impaired patient with highest predicted probability
    p_x = probs["XGBoost"]
    cand = np.where((y_te.values == 1))[0]
    row = int(cand[np.argmax(p_x[cand])])
    explain.plot_waterfall(explainer, sv, Xi, row, FIG_DIR / "shap_waterfall_patient.png")
    out["waterfall_patient"] = {"probability": round(float(p_x[row]), 3),
                                **{k: (None if pd.isna(v) else float(v)) for k, v in X_te.iloc[row].items()}}

    # save the selected model for src/predict.py
    model_dir = Path(RESULTS_DIR).parent / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(zoo["XGBoost"], model_dir / "xgboost_pipeline.joblib")

    # 5. Risk stratification
    tiers = risk.tier_table(p_x, y_te)
    tiers.to_csv(RESULTS_DIR / "risk_tiers.csv")
    out["risk_tiers"] = {str(k): {"n": int(r.n), "impaired": int(r.impaired), "rate": float(r.impaired_rate)}
                         for k, r in tiers.iterrows()}

    # 6. Ablation: which feature groups carry the signal? (XGBoost, 5-fold CV AUC)
    groups = {"All features": FEATURES,
              "Without MMSE": DEMOGRAPHIC + MRI,
              "MRI only": MRI,
              "Demographic only": DEMOGRAPHIC,
              "MMSE only": COGNITIVE,
              "MMSE + MRI": COGNITIVE + MRI}
    abl = {}
    for gname, feats in groups.items():
        for mname in ["Logistic Regression", "XGBoost"]:
            mdl = clone(models.build_models(y_tr)[mname])
            mu, sd = evaluate.cv_auc(mdl, df[feats], df["target"])
            abl.setdefault(gname, {})[mname] = {"cv_auc_mean": mu, "cv_auc_std": sd}
    out["ablation"] = abl
    pd.DataFrame({g: {m: f"{v['cv_auc_mean']:.3f} ± {v['cv_auc_std']:.3f}" for m, v in d.items()}
                  for g, d in abl.items()}).T.to_csv(RESULTS_DIR / "ablation.csv")

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


if __name__ == "__main__":
    res = run()
    print(f"Selected model: {res['selected_model']}")
    print(pd.read_csv(RESULTS_DIR / "model_comparison.csv").round(3).to_string(index=False))
    print("\nAblation (5-fold CV AUC):")
    print(pd.read_csv(RESULTS_DIR / "ablation.csv", index_col=0).to_string())
    print("\nRisk tiers:", res["risk_tiers"])
    print("SHAP:", res["shap_mean_abs"])
