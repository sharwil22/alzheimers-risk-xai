"""Model definitions. Every model is a scikit-learn Pipeline, so imputation and
scaling are fitted on training data only (inside each CV fold as well)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import SEED

# Hyperparameters (reported in the paper, Table "Hyperparameters")
HYPERPARAMS = {
    "Logistic Regression": dict(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced"),
    "Decision Tree": dict(criterion="gini", max_depth=4, min_samples_leaf=5, class_weight="balanced"),
    "Random Forest": dict(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced"),
    "XGBoost": dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9,
                    colsample_bytree=0.9, scale_pos_weight="n_neg / n_pos (train)", eval_metric="logloss"),
    "SVM (RBF)": dict(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True),
    "k-NN": dict(n_neighbors=7, weights="distance"),
}


def _impute():
    return ("impute", SimpleImputer(strategy="median"))


def build_models(y_train, include_benchmarks=False) -> dict:
    """Return the four project models (+ optional SVM / k-NN benchmarks)."""
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    models = {
        "Logistic Regression": Pipeline([_impute(), ("scale", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Decision Tree": Pipeline([_impute(), ("clf", DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=SEED))]),
        "Random Forest": Pipeline([_impute(), ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=SEED))]),
        "XGBoost": Pipeline([_impute(), ("clf", XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=spw, eval_metric="logloss", random_state=SEED))]),
    }
    if include_benchmarks:
        models["SVM (RBF)"] = Pipeline([_impute(), ("scale", StandardScaler()),
                                        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                                                    probability=True, random_state=SEED))])
        models["k-NN"] = Pipeline([_impute(), ("scale", StandardScaler()),
                                   ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))])
    return models
