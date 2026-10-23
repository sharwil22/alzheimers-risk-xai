"""Score a single patient with the trained XGBoost model.

Example:
    python -m src.predict --age 80 --gender F --educ 3 --ses 2 --mmse 20 \\
                          --etiv 1732 --nwbv 0.692 --asf 1.013

Prints the probability of Alzheimer's-related impairment (CDR >= 0.5), the
risk tier and the features that pushed the prediction up or down (SHAP).
Run `python -m src.main` first so that models/xgboost_pipeline.joblib exists.
Research prototype only - not a diagnostic tool.
"""
import argparse

import joblib
import pandas as pd

from .config import FEATURES, ROOT
from .explain import shap_values
from .risk import assign_tiers

MODEL_PATH = ROOT / "models" / "xgboost_pipeline.joblib"


def predict_patient(age, gender, educ, ses, mmse, etiv, nwbv, asf):
    model = joblib.load(MODEL_PATH)
    x = pd.DataFrame([{"Age": age, "Gender": 1 if str(gender).upper().startswith("M") else 0,
                       "Educ": educ, "SES": ses, "MMSE": mmse, "eTIV": etiv, "nWBV": nwbv, "ASF": asf}])[FEATURES]
    prob = float(model.predict_proba(x)[0, 1])
    tier = str(assign_tiers([prob])[0])
    _, sv, _ = shap_values(model, x)
    contrib = pd.Series(sv[0], index=FEATURES).sort_values(key=abs, ascending=False)
    return prob, tier, contrib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--age", type=float, required=True)
    ap.add_argument("--gender", required=True, help="M or F")
    ap.add_argument("--educ", type=float, required=True, help="education level 1-5")
    ap.add_argument("--ses", type=float, default=None, help="socioeconomic status 1-5 (optional)")
    ap.add_argument("--mmse", type=float, required=True)
    ap.add_argument("--etiv", type=float, required=True)
    ap.add_argument("--nwbv", type=float, required=True)
    ap.add_argument("--asf", type=float, required=True)
    a = ap.parse_args()
    prob, tier, contrib = predict_patient(a.age, a.gender, a.educ, a.ses, a.mmse, a.etiv, a.nwbv, a.asf)
    print(f"Probability of impairment : {prob:.3f}")
    print(f"Risk tier                 : {tier}")
    print("Feature contributions (SHAP, log-odds; + raises risk, - lowers risk):")
    for f, v in contrib.items():
        print(f"  {f:<6} {v:+.3f}")


if __name__ == "__main__":
    main()
