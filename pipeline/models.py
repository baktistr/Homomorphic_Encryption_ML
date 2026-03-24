"""Load exported model artifacts and expose model configurations."""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from config import (
    BASELINE_COMPARISON_PATH,
    FEATURE_COLUMNS,
    FEATURE_ORDER_PATH,
    HE_EVAL_FEATURES_PATH,
    HE_EVAL_META_PATH,
    HE_EVAL_PLAINTEXT_PATH,
    LINEAR_SVC_PATH,
    LOGREG_PATH,
    POLY2_LOGREG_PATH,
    POLY2_TRANSFORMER_PATH,
    RIDGE_PATH,
    SCALER_PATH,
)


@dataclass
class ModelSpec:
    """Specification for a single model used in inference."""

    name: str
    display_name: str
    score_label: str
    representation: str
    input_dimension: int
    weights: np.ndarray
    bias: float
    threshold: float = 0.0
    use_sigmoid_label: bool = False
    poly2_transformer: PolynomialFeatures | None = None


class ArtifactStore:
    """Loads and caches all exported artifacts from the baseline notebook."""

    def __init__(self) -> None:
        self._loaded = False
        self.scaler = None
        self.models: dict[str, ModelSpec] = {}
        self.feature_order: list[str] = []
        self.baseline_summary: dict = {}
        self.he_eval_features: np.ndarray | None = None
        self.he_eval_meta: pd.DataFrame | None = None
        self.he_eval_plaintext: pd.DataFrame | None = None

    def load(self) -> None:
        """Load all artifacts from disk. Call once at startup."""
        if self._loaded:
            return

        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Artifacts not found at {SCALER_PATH.parent}. "
                "Run fraud_baseline_6d.ipynb first to generate them."
            )

        # Suppress sklearn version mismatch warnings during unpickling
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="sklearn"
        )

        # Feature order
        with FEATURE_ORDER_PATH.open() as f:
            self.feature_order = json.load(f)

        # Baseline comparison summary
        if BASELINE_COMPARISON_PATH.exists():
            with BASELINE_COMPARISON_PATH.open() as f:
                self.baseline_summary = json.load(f)

        # Scaler
        with SCALER_PATH.open("rb") as f:
            self.scaler = pickle.load(f)

        # Ridge
        with RIDGE_PATH.open("rb") as f:
            ridge = pickle.load(f)
        self.models["ridge_score"] = ModelSpec(
            name="ridge_score",
            display_name="Ridge Score",
            score_label="ridge_raw_score",
            representation="base_6d_scaled",
            input_dimension=6,
            weights=ridge["weights"],
            bias=ridge["bias"],
            threshold=0.0,
        )

        # Logistic Regression
        with LOGREG_PATH.open("rb") as f:
            logreg = pickle.load(f)
        self.models["logistic_regression"] = ModelSpec(
            name="logistic_regression",
            display_name="Logistic Regression",
            score_label="logistic_raw_logit",
            representation="base_6d_scaled",
            input_dimension=6,
            weights=logreg["weights"],
            bias=logreg["bias"],
            threshold=0.0,
            use_sigmoid_label=True,
        )

        # LinearSVC
        with LINEAR_SVC_PATH.open("rb") as f:
            svc_model = pickle.load(f)
        self.models["linear_svc"] = ModelSpec(
            name="linear_svc",
            display_name="Linear SVC",
            score_label="linear_svc_decision",
            representation="base_6d_scaled",
            input_dimension=6,
            weights=svc_model.coef_.reshape(-1),
            bias=float(svc_model.intercept_[0]),
            threshold=0.0,
        )

        # Poly2 Logistic Regression
        with POLY2_TRANSFORMER_PATH.open("rb") as f:
            poly2_transformer = pickle.load(f)
        with POLY2_LOGREG_PATH.open("rb") as f:
            poly2_model = pickle.load(f)
        poly2_dim = int(poly2_transformer.n_output_features_)
        self.models["poly2_logistic_regression"] = ModelSpec(
            name="poly2_logistic_regression",
            display_name="Poly2 Logistic Regression",
            score_label="poly2_logreg_raw_score",
            representation="poly2_expanded_from_base_6d",
            input_dimension=poly2_dim,
            weights=poly2_model.coef_.reshape(-1),
            bias=float(poly2_model.intercept_[0]),
            threshold=0.0,
            use_sigmoid_label=True,
            poly2_transformer=poly2_transformer,
        )

        # HE evaluation subset
        if HE_EVAL_FEATURES_PATH.exists():
            self.he_eval_features = pd.read_csv(HE_EVAL_FEATURES_PATH)[
                FEATURE_COLUMNS
            ].values
        if HE_EVAL_META_PATH.exists():
            self.he_eval_meta = pd.read_csv(HE_EVAL_META_PATH)
        if HE_EVAL_PLAINTEXT_PATH.exists():
            self.he_eval_plaintext = pd.read_csv(HE_EVAL_PLAINTEXT_PATH)

        self._loaded = True

    def get_model(self, name: str) -> ModelSpec:
        if not self._loaded:
            self.load()
        return self.models[name]

    def get_all_models(self) -> dict[str, ModelSpec]:
        if not self._loaded:
            self.load()
        return self.models

    def expand_poly2(self, X_base: np.ndarray) -> np.ndarray:
        """Expand 6D base features to degree-2 polynomial features."""
        transformer = self.models["poly2_logistic_regression"].poly2_transformer
        if transformer is None:
            raise RuntimeError("Poly2 transformer not loaded.")
        return transformer.transform(X_base.reshape(1, -1) if X_base.ndim == 1 else X_base)


# Singleton instance
store = ArtifactStore()
