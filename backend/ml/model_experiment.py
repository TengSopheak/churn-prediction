from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform


# ---------- Constants ----------
RANDOM_STATE = 42
CV_SPLITS = 5
N_ITER = 15  # RandomizedSearchCV iterations
SCORING = "f1"

# ---------- Model catalogue ----------
BASE_MODELS: Dict[str, BaseEstimator] = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
    "SVM": SVC(
        class_weight="balanced",
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
}

# ---------- Hyper-parameter grids ----------
PARAM_GRIDS: Dict[str, dict] = {
    "LightGBM": {
        'num_leaves': randint(20, 50),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5)
    }
}


# ---------- I/O helpers ----------
def load_data(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load pre-processed train & test sets."""
    data_dir = Path(data_dir)
    X_train = pd.read_csv(data_dir / "X_train_balanced.csv")
    y_train = pd.read_csv(data_dir / "y_train_balanced.csv").squeeze()
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def save_artifacts(models: Dict[str, BaseEstimator], results: pd.DataFrame, out_dir: str | Path):
    """Persist trained models and metrics."""
    # out_dir = Path(out_dir)
    # out_dir.mkdir(parents=True, exist_ok=True)

    # # Save each model
    # for name, model in models.items():
    #     joblib.dump(model, out_dir / f"{name.replace(' ', '_')}.pkl")
    pass


# ---------- ML helpers ----------
def cross_validate_model(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, name: str
) -> Dict[str, float]:
    """Return mean ± std of chosen metric across CV folds."""
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {"Model": name, "Mean": scores.mean(), "Std": scores.std()}


def hyperparameter_tune(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, name: str
) -> BaseEstimator:
    """Random-search hyper-parameters for a given model."""
    if name not in PARAM_GRIDS:
        return model  # no tuning defined

    search = RandomizedSearchCV(
        model,
        PARAM_GRIDS[name],
        n_iter=N_ITER,
        cv=CV_SPLITS,
        scoring=SCORING,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_


def evaluate_on_test(
    model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, name: str
) -> Dict[str, Any]:
    """Compute test-set metrics."""
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
    }


# ---------- Public API ----------
def run_experiment(data_dir: str | Path, out_dir: str | Path) -> Tuple[pd.DataFrame, BaseEstimator]:
    """
    End-to-end experiment:
        1. Load data
        2. Cross-validate base models
        3. Hyper-tune best performer
        4. Fit all models on full training set
        5. Evaluate on held-out test set
        6. Persist artefacts
    Returns:
        DataFrame with test-set metrics and the best model object.
    """
    X_train, X_test, y_train, y_test = load_data(data_dir)

    # 1. Cross-validate base models
    cv_results = [cross_validate_model(model, X_train, y_train, name) for name, model in BASE_MODELS.items()]
    cv_df = pd.DataFrame(cv_results).sort_values("Mean", ascending=False)

    best_model_name = cv_df.iloc[0]["Model"]

    # 2. Hyper-parameter tune the winner
    tuned_model = hyperparameter_tune(BASE_MODELS[best_model_name], X_train, y_train, best_model_name)

    # 3. Fit all models on full training set
    trained_models: Dict[str, BaseEstimator] = {}
    for name, model in BASE_MODELS.items():
        if name == best_model_name:
            trained_models[name] = tuned_model
        else:
            trained_models[name] = model
        trained_models[name].fit(X_train, y_train)

    # 4. Evaluate on test set
    test_metrics = [evaluate_on_test(model, X_test, y_test, name) for name, model in trained_models.items()]
    test_df = pd.DataFrame(test_metrics).sort_values("F1-Score", ascending=False)

    # 5. Persist
    save_artifacts(trained_models, test_df, out_dir)

    best_model = trained_models[test_df.iloc[0]["Model"]]
    return test_df, best_model


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telco-Churn model experiment pipeline")
    parser.add_argument("--data-dir", default="./data", help="Directory containing processed CSVs")
    parser.add_argument("--out-dir", default="./models", help="Directory to save models & metrics")
    args = parser.parse_args()

    run_experiment(args.data_dir, args.out_dir)
