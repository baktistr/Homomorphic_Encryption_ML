"""Homomorphic Encryption (CKKS) inference pipeline using TenSEAL."""

from __future__ import annotations

import math
import time

import numpy as np

from pipeline.models import ArtifactStore, ModelSpec

# Graceful TenSEAL import
try:
    import tenseal as ts

    TENSEAL_AVAILABLE = True
except ImportError:
    ts = None
    TENSEAL_AVAILABLE = False


def is_available() -> bool:
    """Check whether TenSEAL is installed."""
    return TENSEAL_AVAILABLE


def build_ckks_context(config: dict):
    """Build a TenSEAL CKKS context from a parameter config dict."""
    if not TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL is not installed.")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=config["poly_modulus_degree"],
        coeff_mod_bit_sizes=config["coeff_mod_bit_sizes"],
    )
    context.global_scale = 2 ** config["global_scale_bits"]
    context.generate_galois_keys()
    return context


def predict_single_he(
    X: np.ndarray,
    spec: ModelSpec,
    store: ArtifactStore,
    ckks_config: dict,
    plaintext_score: float | None = None,
) -> dict:
    """Run CKKS-encrypted inference for a single sample.

    For poly2 model, X should be the base 6D vector — expansion is handled
    internally.
    """
    if not TENSEAL_AVAILABLE:
        return _unavailable_result(spec)

    if spec.representation == "poly2_expanded_from_base_6d":
        X_input = store.expand_poly2(X).reshape(-1)
    else:
        X_input = X

    context = build_ckks_context(ckks_config)

    # Encrypt
    t_enc_start = time.perf_counter()
    enc_vec = ts.ckks_vector(context, X_input.tolist())
    encrypt_ms = (time.perf_counter() - t_enc_start) * 1000

    # Encrypted inference (dot product + bias)
    t_infer_start = time.perf_counter()
    enc_score = enc_vec.dot(spec.weights.tolist())
    enc_score = enc_score + spec.bias
    infer_ms = (time.perf_counter() - t_infer_start) * 1000

    # Decrypt
    t_dec_start = time.perf_counter()
    decrypted_score = float(enc_score.decrypt()[0])
    decrypt_ms = (time.perf_counter() - t_dec_start) * 1000

    # Ciphertext size
    try:
        ct_bytes = len(enc_score.serialize())
    except Exception:
        ct_bytes = 0

    # Absolute error vs plaintext
    abs_error = None
    if plaintext_score is not None:
        abs_error = abs(plaintext_score - decrypted_score)

    # Label from decrypted score
    from scipy.special import expit

    if spec.use_sigmoid_label:
        label = 1 if expit(decrypted_score) >= 0.5 else 0
    else:
        label = 1 if decrypted_score >= spec.threshold else 0

    return {
        "model_name": spec.name,
        "display_name": spec.display_name,
        "representation": spec.representation,
        "input_dimension": spec.input_dimension,
        "ckks_config": ckks_config["name"],
        "decrypted_score": decrypted_score,
        "label": label,
        "label_text": "FRAUD" if label == 1 else "NOT FRAUD",
        "encrypt_ms": encrypt_ms,
        "infer_ms": infer_ms,
        "decrypt_ms": decrypt_ms,
        "total_he_ms": encrypt_ms + infer_ms + decrypt_ms,
        "ciphertext_bytes": ct_bytes,
        "ciphertext_kb": round(ct_bytes / 1024, 2) if ct_bytes else 0,
        "abs_error": abs_error,
    }


def predict_batch_he(
    X_all: np.ndarray,
    plaintext_scores: np.ndarray,
    spec: ModelSpec,
    store: ArtifactStore,
    ckks_config: dict,
) -> dict:
    """Run CKKS-encrypted inference on a batch and collect timing/error stats.

    Returns a dict with 'records' (per-sample) and 'timing' (aggregate).
    """
    if not TENSEAL_AVAILABLE:
        return {"records": [], "timing": _unavailable_timing(), "available": False}

    if spec.representation == "poly2_expanded_from_base_6d":
        X_input = store.expand_poly2(X_all)
    else:
        X_input = X_all

    context = build_ckks_context(ckks_config)

    encrypt_time = 0.0
    inference_time = 0.0
    decrypt_time = 0.0
    ciphertext_sizes = []
    records = []

    for idx in range(len(X_input)):
        features = X_input[idx]
        plain_score = float(plaintext_scores[idx])

        t0 = time.perf_counter()
        enc_vec = ts.ckks_vector(context, features.tolist())
        encrypt_time += time.perf_counter() - t0

        t1 = time.perf_counter()
        enc_score = enc_vec.dot(spec.weights.tolist())
        enc_score = enc_score + spec.bias
        inference_time += time.perf_counter() - t1

        t2 = time.perf_counter()
        decrypted_score = float(enc_score.decrypt()[0])
        decrypt_time += time.perf_counter() - t2

        try:
            ct_size = len(enc_score.serialize())
        except Exception:
            ct_size = 0
        ciphertext_sizes.append(ct_size)

        records.append({
            "sample_index": idx,
            "plaintext_score": plain_score,
            "decrypted_score": decrypted_score,
            "absolute_error": abs(plain_score - decrypted_score),
        })

    n = len(X_input)
    abs_errors = [r["absolute_error"] for r in records]
    avg_ct = float(np.nanmean(ciphertext_sizes)) if ciphertext_sizes else 0

    timing = {
        "n_samples": n,
        "encrypt_time_sec": encrypt_time,
        "inference_time_sec": inference_time,
        "decrypt_time_sec": decrypt_time,
        "total_runtime_sec": encrypt_time + inference_time + decrypt_time,
        "avg_encrypt_ms": (encrypt_time / n * 1000) if n else 0,
        "avg_infer_ms": (inference_time / n * 1000) if n else 0,
        "avg_decrypt_ms": (decrypt_time / n * 1000) if n else 0,
        "avg_total_ms": ((encrypt_time + inference_time + decrypt_time) / n * 1000) if n else 0,
        "mean_abs_error": float(np.mean(abs_errors)) if abs_errors else 0,
        "max_abs_error": float(np.max(abs_errors)) if abs_errors else 0,
        "median_abs_error": float(np.median(abs_errors)) if abs_errors else 0,
        "avg_ciphertext_bytes": avg_ct,
        "avg_ciphertext_kb": round(avg_ct / 1024, 2) if avg_ct else 0,
    }

    return {"records": records, "timing": timing, "available": True}


def run_parameter_sweep(
    X_all: np.ndarray,
    plaintext_scores: np.ndarray,
    spec: ModelSpec,
    store: ArtifactStore,
    configs: list[dict],
    max_samples: int = 16,
) -> list[dict]:
    """Run batch HE inference across multiple CKKS configs for one model.

    Uses a subset of samples (default 16) to keep sweep fast while still
    producing representative timing and error statistics.
    """
    if not TENSEAL_AVAILABLE:
        return []

    # Use a smaller subset for the sweep to reduce total runtime
    n = min(max_samples, len(X_all))
    X_sweep = X_all[:n]
    scores_sweep = plaintext_scores[:n]

    sweep_results = []
    for config in configs:
        result = predict_batch_he(X_sweep, scores_sweep, spec, store, config)
        if result["available"]:
            row = {
                "model_name": spec.name,
                "display_name": spec.display_name,
                "config_name": config["name"],
                "poly_modulus_degree": config["poly_modulus_degree"],
                **result["timing"],
            }
            sweep_results.append(row)

    return sweep_results


def _unavailable_result(spec: ModelSpec) -> dict:
    return {
        "model_name": spec.name,
        "display_name": spec.display_name,
        "available": False,
        "message": "TenSEAL is not installed. Install with: pip install tenseal",
    }


def _unavailable_timing() -> dict:
    return {
        "available": False,
        "message": "TenSEAL is not installed.",
    }
