"""Central configuration: paths, random seed, features and risk thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "oasis_cross-sectional.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

SEED = 42            # used for the split, CV folds and every model
TEST_SIZE = 0.25     # stratified hold-out fraction (176 train / 59 test)
CV_FOLDS = 5

# Input features. CDR is NOT an input: it defines the target (leakage control).
DEMOGRAPHIC = ["Age", "Gender", "Educ", "SES"]
COGNITIVE = ["MMSE"]
MRI = ["eTIV", "nWBV", "ASF"]
FEATURES = DEMOGRAPHIC + COGNITIVE + MRI

# Provisional risk-tier cut-offs on the predicted probability of impairment
RISK_BINS = [0.0, 0.30, 0.60, 1.0]
RISK_LABELS = ["Low", "Medium", "High"]

MODEL_COLORS = {
    "Logistic Regression": "#3b82f6",
    "Decision Tree": "#f2a93b",
    "Random Forest": "#9b7fe6",
    "XGBoost": "#2d2a8f",
    "SVM (RBF)": "#64748b",
    "k-NN": "#94a3b8",
}
