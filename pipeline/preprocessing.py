"""Feature engineering and scaling for the 6D fraud detection pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    BINARY_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_BASE_FEATURES,
    NUMERIC_SCALED_FEATURES,
    RAW_INPUT_COLUMNS,
    SELECTED_TRANSACTION_TYPES,
)


def validate_single_input(raw: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single raw transaction dict. Returns (ok, errors)."""
    errors = []

    for col in RAW_INPUT_COLUMNS:
        if col not in raw or raw[col] is None or raw[col] == "":
            errors.append(f"Missing required field: {col}")

    if "type" in raw and raw["type"] not in SELECTED_TRANSACTION_TYPES:
        errors.append(
            f"Transaction type must be one of {SELECTED_TRANSACTION_TYPES}, "
            f"got '{raw['type']}'"
        )

    numeric_fields = [c for c in RAW_INPUT_COLUMNS if c != "type"]
    for field in numeric_fields:
        if field in raw and raw[field] is not None and raw[field] != "":
            try:
                float(raw[field])
            except (ValueError, TypeError):
                errors.append(f"Field '{field}' must be a number, got '{raw[field]}'")

    return len(errors) == 0, errors


def validate_batch_input(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate a batch DataFrame. Returns (ok, errors)."""
    errors = []
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return False, errors

    if df.empty:
        errors.append("Uploaded CSV is empty.")
        return False, errors

    bad_types = df[~df["type"].isin(SELECTED_TRANSACTION_TYPES)]
    if len(bad_types) > 0:
        errors.append(
            f"{len(bad_types)} rows have unsupported transaction types. "
            f"Only {SELECTED_TRANSACTION_TYPES} are supported."
        )

    return len(errors) == 0, errors


def engineer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns: deltaOrig, deltaDest, is_transfer."""
    df = df.copy()
    df["deltaOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["deltaDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    return df


def scale_features_df(
    df: pd.DataFrame, scaler: StandardScaler
) -> pd.DataFrame:
    """Apply scaler to numeric base features and add scaled columns."""
    df = df.copy()
    scaled_values = scaler.transform(df[NUMERIC_BASE_FEATURES])
    scaled_df = pd.DataFrame(
        scaled_values, columns=NUMERIC_SCALED_FEATURES, index=df.index
    )
    return pd.concat([df, scaled_df], axis=1)


def raw_dict_to_feature_vector(
    raw: dict[str, Any], scaler: StandardScaler
) -> np.ndarray:
    """Convert a single raw transaction dict to a 6D scaled feature vector."""
    row = {
        "type": raw["type"],
        "amount": float(raw["amount"]),
        "oldbalanceOrg": float(raw["oldbalanceOrg"]),
        "newbalanceOrig": float(raw["newbalanceOrig"]),
        "oldbalanceDest": float(raw["oldbalanceDest"]),
        "newbalanceDest": float(raw["newbalanceDest"]),
    }
    df = pd.DataFrame([row])
    df = engineer_features_df(df)
    df = scale_features_df(df, scaler)
    return df[FEATURE_COLUMNS].values[0]


def raw_df_to_feature_matrix(
    df: pd.DataFrame, scaler: StandardScaler
) -> np.ndarray:
    """Convert a batch DataFrame to an (N, 6) scaled feature matrix."""
    df = df.copy()
    numeric_cols = [c for c in RAW_INPUT_COLUMNS if c != "type"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["type"].isin(SELECTED_TRANSACTION_TYPES)].copy()
    df = engineer_features_df(df)
    df = scale_features_df(df, scaler)
    return df[FEATURE_COLUMNS].values
