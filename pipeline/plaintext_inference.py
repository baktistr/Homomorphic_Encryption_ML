"""Plaintext (normal) inference pipeline for all four models."""

from __future__ import annotations

import time

import numpy as np
from scipy.special import expit

from pipeline.models import ArtifactStore, ModelSpec


def _compute_score(X: np.ndarray, spec: ModelSpec) -> float:
    """Compute raw score: X @ weights + bias."""
    return float(X @ spec.weights + spec.bias)


def _compute_label(score: float, spec: ModelSpec) -> int:
    """Determine fraud label from score."""
    if spec.use_sigmoid_label:
        return 1 if expit(score) >= 0.5 else 0
    return 1 if score >= spec.threshold else 0


def predict_single(
    X: np.ndarray, spec: ModelSpec, store: ArtifactStore
) -> dict:
    """Run plaintext inference for a single sample.

    For the poly2 model, X should be the base 6D vector — expansion is
    handled internally.
    """
    if spec.representation == "poly2_expanded_from_base_6d":
        X_input = store.expand_poly2(X).reshape(-1)
    else:
        X_input = X

    t0 = time.perf_counter()
    score = _compute_score(X_input, spec)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    label = _compute_label(score, spec)

    return {
        "model_name": spec.name,
        "display_name": spec.display_name,
        "representation": spec.representation,
        "input_dimension": spec.input_dimension,
        "score": score,
        "label": label,
        "label_text": "FRAUD" if label == 1 else "NOT FRAUD",
        "time_ms": elapsed_ms,
    }


def predict_batch(
    X_all: np.ndarray, specs: dict[str, ModelSpec], store: ArtifactStore
) -> dict[str, list[dict]]:
    """Run plaintext inference for all models on a batch of samples.

    Returns {model_name: [result_per_sample, ...]}.
    """
    results: dict[str, list[dict]] = {}

    for name, spec in specs.items():
        if spec.representation == "poly2_expanded_from_base_6d":
            X_input = store.expand_poly2(X_all)
        else:
            X_input = X_all

        model_results = []
        t0 = time.perf_counter()
        scores = X_input @ spec.weights + spec.bias
        total_ms = (time.perf_counter() - t0) * 1000

        for i, score in enumerate(scores):
            score_f = float(score)
            label = _compute_label(score_f, spec)
            model_results.append({
                "sample_index": i,
                "score": score_f,
                "label": label,
                "label_text": "FRAUD" if label == 1 else "NOT FRAUD",
            })

        results[name] = model_results
        # Attach timing to the first record for reference
        if model_results:
            model_results[0]["batch_time_ms"] = total_ms

    return results


def get_batch_scores(
    X_all: np.ndarray, spec: ModelSpec, store: ArtifactStore
) -> np.ndarray:
    """Return raw score array for a batch (used for metric computation)."""
    if spec.representation == "poly2_expanded_from_base_6d":
        X_input = store.expand_poly2(X_all)
    else:
        X_input = X_all
    return X_input @ spec.weights + spec.bias


def get_batch_labels(
    scores: np.ndarray, spec: ModelSpec
) -> np.ndarray:
    """Convert raw scores to binary labels."""
    if spec.use_sigmoid_label:
        return (expit(scores) >= 0.5).astype(int)
    return (scores >= spec.threshold).astype(int)
