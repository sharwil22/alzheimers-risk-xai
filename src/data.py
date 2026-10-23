"""Loading, cohort selection and target construction for OASIS-1."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, FEATURES, SEED, TEST_SIZE


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read the OASIS-1 cross-sectional table (436 rows = 416 subjects + 20 re-scans)."""
    return pd.read_csv(path)


def build_cohort(raw: pd.DataFrame) -> pd.DataFrame:
    """Select one scan per subject with a CDR rating and build the binary target.

    * Keep only the first session (``_MR1``); the 20 ``_MR2`` rows are
      reliability re-scans of subjects already present.
    * Keep subjects with a Clinical Dementia Rating (CDR). Younger subjects in
      OASIS-1 were not clinically rated, so they cannot be labelled.
    * Target: CDR = 0 -> 0 (cognitively normal); CDR >= 0.5 -> 1 (impaired).
    """
    df = raw[raw["ID"].str.endswith("_MR1")].copy()
    df = df[df["CDR"].notna()].copy()
    df["target"] = (df["CDR"] >= 0.5).astype(int)
    df["Gender"] = (df["M/F"] == "M").astype(int)     # 1 = male, 0 = female
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, features=FEATURES):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    X, y = df[features], df["target"]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def describe(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Summary numbers used in the report."""
    subjects = raw[raw["ID"].str.endswith("_MR1")]
    return {
        "rows_in_file": int(len(raw)),
        "subjects": int(len(subjects)),
        "reliability_rescans": int((~raw["ID"].str.endswith("_MR1")).sum()),
        "without_cdr": int(subjects["CDR"].isna().sum()),
        "labelled": int(len(df)),
        "normal": int((df["target"] == 0).sum()),
        "impaired": int((df["target"] == 1).sum()),
        "cdr_counts": {str(k): int(v) for k, v in df["CDR"].value_counts().sort_index().items()},
        "age_range_all": [int(subjects["Age"].min()), int(subjects["Age"].max())],
        "age_range_labelled": [int(df["Age"].min()), int(df["Age"].max())],
        "female_labelled": int((df["Gender"] == 0).sum()),
        "missing_in_features": {k: int(v) for k, v in df[FEATURES].isna().sum().items() if v > 0},
        "duplicates": int(df.duplicated(subset=FEATURES + ["target"]).sum()),
    }
