from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ---------- Constants ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CORR_THRESHOLD = 0.05  # absolute correlation with target to keep feature
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# ---------- Encoder definitions ----------
# Order matters – must match categories= parameter
ORDINAL_ENCODERS = {
    "gender": OrdinalEncoder(categories=[["Female", "Male"]]),
    "Partner": OrdinalEncoder(categories=[["No", "Yes"]]),
    "Dependents": OrdinalEncoder(categories=[["No", "Yes"]]),
    "PaperlessBilling": OrdinalEncoder(categories=[["No", "Yes"]]),
    "MultipleLines": OrdinalEncoder(categories=[["No phone service", "No", "Yes"]]),
    "InternetService": OrdinalEncoder(categories=[["No", "DSL", "Fiber optic"]]),
    "OnlineSecurity": OrdinalEncoder(categories=[["No internet service", "No", "Yes"]]),
    "OnlineBackup": OrdinalEncoder(categories=[["No internet service", "No", "Yes"]]),
    "DeviceProtection": OrdinalEncoder(categories=[["No internet service", "No", "Yes"]]),
    "TechSupport": OrdinalEncoder(categories=[["No internet service", "No", "Yes"]]),
    "StreamingTV": OrdinalEncoder(categories=[["No internet service", "No", "Yes"]]),
    "StreamingMovies": OrdinalEncoder(categories=[["No internet service", "No", "Yes"]]),
    "Contract": OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]]),
    "PaymentMethod": OrdinalEncoder(
        categories=[["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]]
    ),
}


# ---------- Helper functions ----------
def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load and basic sanity-check."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    return df


def fix_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges object -> float."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train-test split *before* any preprocessing to avoid leakage."""
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})  # encode target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def handle_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing TotalCharges with training median (computed on train only)."""
    X_train, X_test = X_train.copy(), X_test.copy()
    median_tc = X_train["TotalCharges"].median()
    X_train["TotalCharges"] = X_train["TotalCharges"].fillna(median_tc)
    X_test["TotalCharges"] = X_test["TotalCharges"].fillna(median_tc)
    return X_train, X_test


def encode_categorical(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ordinal-encode categorical columns using pre-defined encoders."""
    X_train, X_test = X_train.copy(), X_test.copy()
    for col, enc in ORDINAL_ENCODERS.items():
        X_train[col] = enc.fit_transform(X_train[[col]]).ravel()
        X_test[col] = enc.transform(X_test[[col]]).ravel()
    return X_train, X_test


def scale_numerical(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standard-scale numerical columns; fit on train, transform both."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])
    return X_train, X_test, scaler


def select_features(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep features whose absolute correlation with target exceeds threshold."""
    tmp = pd.concat([X_train, y_train.rename("Churn")], axis=1)
    corr = tmp.corr(numeric_only=True)["Churn"].drop("Churn")
    selected = corr[corr.abs() > CORR_THRESHOLD].index.tolist()
    return X_train[selected], X_test[selected]


def balance_train(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to training data only."""
    smote = SMOTE(random_state=RANDOM_STATE)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    return X_bal, y_bal


def save_outputs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: StandardScaler,
    out_dir: str | Path,
) -> None:
    """Persist artefacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out_dir / "X_train_balanced.csv", index=False)
    y_train.to_csv(out_dir / "y_train_balanced.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)
    joblib.dump(scaler, out_dir / "scaler.pkl")


# ---------- Public API ----------
def build_pipeline() -> list:
    """Return ordered list of (name, func) tuples for pipeline inspection."""
    return [
        ("load", load_data),
        ("fix_types", fix_datatypes),
        ("split", split_data),
        ("handle_na", handle_missing_values),
        ("encode", encode_categorical),
        ("scale", scale_numerical),
        ("select", select_features),
        ("balance", balance_train),
    ]


def run_pipeline(csv_path: str | Path, out_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """End-to-end pipeline; returns balanced train + original test sets."""
    df = load_data(csv_path)
    df = fix_datatypes(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = handle_missing_values(X_train, X_test)
    X_train, X_test = encode_categorical(X_train, X_test)
    X_train, X_test, scaler = scale_numerical(X_train, X_test)
    X_train, X_test = select_features(X_train, y_train, X_test)
    X_train_bal, y_train_bal = balance_train(X_train, y_train)
    save_outputs(X_train_bal, y_train_bal, X_test, y_test, scaler, out_dir)
    return X_train_bal, X_test, y_train_bal, y_test


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telco-Churn feature-engineering pipeline")
    parser.add_argument("--input", required=True, help="Raw dataset.csv path")
    parser.add_argument("--out-dir", default="../dataset/original/dataset.csv", help="Directory to save artefacts")
    args = parser.parse_args()

    run_pipeline(args.input, args.out_dir)
