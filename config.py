"""Application configuration: paths, feature constants, and CKKS parameters."""

from __future__ import annotations

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

# Artifact file paths
SCALER_PATH = ARTIFACT_DIR / "baseline_6d_scaler.pkl"
RIDGE_PATH = ARTIFACT_DIR / "baseline_6d_ridge.pkl"
LOGREG_PATH = ARTIFACT_DIR / "baseline_6d_logreg.pkl"
LINEAR_SVC_PATH = ARTIFACT_DIR / "linear_svc_6d.pkl"
POLY2_LOGREG_PATH = ARTIFACT_DIR / "poly2_logreg_6d.pkl"
POLY2_TRANSFORMER_PATH = ARTIFACT_DIR / "poly2_transformer_6d.pkl"
FEATURE_ORDER_PATH = ARTIFACT_DIR / "feature_order_6d.json"
BASELINE_COMPARISON_PATH = ARTIFACT_DIR / "baseline_model_comparison_summary.json"

# HE evaluation subset
HE_EVAL_FEATURES_PATH = ARTIFACT_DIR / "he_eval_scaled_features_6d.csv"
HE_EVAL_META_PATH = ARTIFACT_DIR / "he_eval_meta_6d.csv"
HE_EVAL_PLAINTEXT_PATH = ARTIFACT_DIR / "he_eval_plaintext_scores_6d.csv"

# ── Feature constants ──────────────────────────────────────────────────────────
NUMERIC_BASE_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "deltaOrig",
    "deltaDest",
]

NUMERIC_SCALED_FEATURES = [
    "amount_scaled",
    "oldbalanceOrg_scaled",
    "oldbalanceDest_scaled",
    "deltaOrig_scaled",
    "deltaDest_scaled",
]

BINARY_FEATURES = ["is_transfer"]

FEATURE_COLUMNS = NUMERIC_SCALED_FEATURES + BINARY_FEATURES

SELECTED_TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT"]

# Raw input columns expected from user / CSV upload
RAW_INPUT_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# ── CKKS parameter configurations ─────────────────────────────────────────────
CKKS_PARAMETER_OPTIONS = [
    {
        "name": "fast_8192",
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale_bits": 40,
    },
    {
        "name": "balanced_16384",
        "poly_modulus_degree": 16384,
        "coeff_mod_bit_sizes": [60, 40, 40, 40, 60],
        "global_scale_bits": 40,
    },
    {
        "name": "precise_16384",
        "poly_modulus_degree": 16384,
        "coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 60],
        "global_scale_bits": 40,
    },
]

PRIMARY_CKKS_CONFIG = CKKS_PARAMETER_OPTIONS[0]

# ── Flask settings ─────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 5
SECRET_KEY = "he-fraud-dashboard-dev-key"
