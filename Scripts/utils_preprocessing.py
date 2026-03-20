# scripts/utils_preprocessing.py

import os
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_data_paths() -> Tuple[str, str]:
    root = get_project_root()
    train_path = os.path.join(root, "data", "industrial_fault_training_dataset.csv")
    test_path = os.path.join(root, "data", "industrial_fault_testing_dataset.csv")
    return train_path, test_path


def get_models_dir() -> str:
    path = os.path.join(get_project_root(), "models")
    os.makedirs(path, exist_ok=True)
    return path


def get_outputs_dir() -> str:
    path = os.path.join(get_project_root(), "outputs")
    os.makedirs(path, exist_ok=True)
    return path


def load_datasets(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Testing file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def inspect_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n=== DATA INSPECTION ===")
    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)

    print("\nTraining preview:")
    print(train_df.head())

    print("\nTesting preview:")
    print(test_df.head())

    print("\nMissing values in training:")
    print(train_df.isnull().sum())

    print("\nMissing values in testing:")
    print(test_df.isnull().sum())

    print("\nDuplicate rows in training:", train_df.duplicated().sum())
    print("Duplicate rows in testing:", test_df.duplicated().sum())


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" not in df.columns:
        raise ValueError("The dataset must contain a 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isnull().sum() > 0:
        raise ValueError("Some timestamp values could not be parsed.")

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    df.drop(columns=["timestamp"], inplace=True)
    return df


def remove_leakage_columns(df: pd.DataFrame, leakage_columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    to_drop = [col for col in leakage_columns if col in df.columns]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
    return df


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return categorical_cols, numeric_cols


def encode_target(y_train: pd.Series, y_test: pd.Series):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return y_train_encoded, y_test_encoded, label_encoder


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    X_train = X_train.copy()
    X_test = X_test.copy()
    feature_encoders: Dict[str, LabelEncoder] = {}

    for col in categorical_cols:
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col].astype(str))

        known_classes = set(encoder.classes_)
        unseen_values = sorted(set(X_test[col].astype(str).unique()) - known_classes)
        if unseen_values:
            encoder.classes_ = list(encoder.classes_) + unseen_values
            encoder.classes_ = pd.Index(encoder.classes_).astype(str).values

        X_test[col] = encoder.transform(X_test[col].astype(str))
        feature_encoders[col] = encoder

    return X_train, X_test, feature_encoders


def scale_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, scaler


def preprocess_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "fault_status",
    leakage_columns: List[str] = None,
):
    if leakage_columns is None:
        leakage_columns = ["error_code"]

    train_df = extract_time_features(train_df)
    test_df = extract_time_features(test_df)

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    X_train = remove_leakage_columns(X_train, leakage_columns)
    X_test = remove_leakage_columns(X_test, leakage_columns)

    selected_feature_columns = X_train.columns.tolist()

    categorical_cols, numeric_cols = get_feature_types(X_train)
    X_train, X_test, feature_encoders = encode_categorical_features(
        X_train, X_test, categorical_cols
    )
    X_train, X_test, scaler = scale_numeric_features(X_train, X_test, numeric_cols)

    y_train_encoded, y_test_encoded, label_encoder = encode_target(y_train, y_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_encoded": y_train_encoded,
        "y_test_encoded": y_test_encoded,
        "label_encoder": label_encoder,
        "feature_encoders": feature_encoders,
        "scaler": scaler,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "selected_feature_columns": selected_feature_columns,
    }


def preprocess_new_data(
    df: pd.DataFrame,
    feature_encoders: Dict[str, LabelEncoder],
    scaler: StandardScaler,
    categorical_cols: List[str],
    numeric_cols: List[str],
    selected_feature_columns: List[str],
    leakage_columns: List[str] = None,
) -> pd.DataFrame:
    if leakage_columns is None:
        leakage_columns = ["error_code"]

    df = extract_time_features(df)
    df = remove_leakage_columns(df, leakage_columns)

    missing_cols = [col for col in selected_feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[selected_feature_columns].copy()

    for col in categorical_cols:
        encoder = feature_encoders[col]
        values = df[col].astype(str)

        unknown = [v for v in values.unique() if v not in set(encoder.classes_)]
        if unknown:
            raise ValueError(f"Unseen category in column '{col}': {unknown}")

        df[col] = encoder.transform(values)

    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


def save_artifact(obj, filename: str) -> None:
    path = os.path.join(get_models_dir(), filename)
    joblib.dump(obj, path)
    print(f"Saved: {path}")


def load_artifact(filename: str):
    path = os.path.join(get_models_dir(), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)