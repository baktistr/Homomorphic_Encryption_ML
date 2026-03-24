"""Flask application for the HE Fraud Detection Dashboard."""

from __future__ import annotations

import os
import traceback

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    CKKS_PARAMETER_OPTIONS,
    FEATURE_COLUMNS,
    MAX_UPLOAD_SIZE_MB,
    PRIMARY_CKKS_CONFIG,
    SECRET_KEY,
    UPLOAD_DIR,
)
from pipeline.models import store
from pipeline.preprocessing import (
    raw_df_to_feature_matrix,
    raw_dict_to_feature_vector,
    validate_batch_input,
    validate_single_input,
)
from pipeline import plaintext_inference as pt
from pipeline import encrypted_inference as he

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024


# ── Startup ────────────────────────────────────────────────────────────────────

def load_artifacts():
    """Attempt to load model artifacts; return success flag."""
    try:
        store.load()
        return True
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        return False


ARTIFACTS_LOADED = load_artifacts()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        artifacts_loaded=ARTIFACTS_LOADED,
        tenseal_available=he.is_available(),
        models=store.get_all_models() if ARTIFACTS_LOADED else {},
    )


@app.route("/single", methods=["GET", "POST"])
def single():
    if not ARTIFACTS_LOADED:
        flash("Model artifacts not found. Run fraud_baseline_6d.ipynb first.", "error")
        return redirect(url_for("index"))

    results = None
    raw_input = None
    feature_vector = None
    errors = []

    if request.method == "POST":
        raw_input = {
            "type": request.form.get("type", ""),
            "amount": request.form.get("amount", ""),
            "oldbalanceOrg": request.form.get("oldbalanceOrg", ""),
            "newbalanceOrig": request.form.get("newbalanceOrig", ""),
            "oldbalanceDest": request.form.get("oldbalanceDest", ""),
            "newbalanceDest": request.form.get("newbalanceDest", ""),
        }

        ok, errors = validate_single_input(raw_input)
        if ok:
            try:
                X = raw_dict_to_feature_vector(raw_input, store.scaler)
                feature_vector = {
                    name: float(X[i]) for i, name in enumerate(FEATURE_COLUMNS)
                }

                models = store.get_all_models()
                results = []

                for name, spec in models.items():
                    # Plaintext
                    pt_result = pt.predict_single(X, spec, store)

                    # Encrypted
                    he_result = he.predict_single_he(
                        X, spec, store, PRIMARY_CKKS_CONFIG,
                        plaintext_score=pt_result["score"],
                    )

                    results.append({
                        "model": spec,
                        "plaintext": pt_result,
                        "encrypted": he_result,
                    })
            except Exception as e:
                errors.append(f"Inference error: {e}")
                traceback.print_exc()

    return render_template(
        "single.html",
        results=results,
        raw_input=raw_input,
        feature_vector=feature_vector,
        errors=errors,
        tenseal_available=he.is_available(),
        ckks_config=PRIMARY_CKKS_CONFIG,
    )


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if not ARTIFACTS_LOADED:
        flash("Model artifacts not found. Run fraud_baseline_6d.ipynb first.", "error")
        return redirect(url_for("index"))

    results = None
    errors = []
    summary = None

    if request.method == "POST":
        file = request.files.get("csvfile")
        if not file or file.filename == "":
            errors.append("No file selected.")
        elif not file.filename.endswith(".csv"):
            errors.append("Please upload a .csv file.")
        else:
            try:
                df = pd.read_csv(file)
                ok, val_errors = validate_batch_input(df)
                if not ok:
                    errors.extend(val_errors)
                else:
                    X_all = raw_df_to_feature_matrix(df, store.scaler)
                    labels_col = None
                    if "isFraud" in df.columns:
                        # Filter to matching rows
                        from config import SELECTED_TRANSACTION_TYPES
                        mask = df["type"].isin(SELECTED_TRANSACTION_TYPES)
                        labels_col = df.loc[mask, "isFraud"].values

                    n_rows = len(X_all)
                    models = store.get_all_models()
                    results = []

                    for name, spec in models.items():
                        # Plaintext scores & labels
                        pt_scores = pt.get_batch_scores(X_all, spec, store)
                        pt_labels = pt.get_batch_labels(pt_scores, spec)

                        # Metrics (if ground truth available)
                        pt_metrics = {}
                        if labels_col is not None:
                            pt_metrics = _compute_metrics(labels_col, pt_labels, pt_scores)

                        # Encrypted batch
                        he_batch = he.predict_batch_he(
                            X_all, pt_scores, spec, store, PRIMARY_CKKS_CONFIG
                        )

                        he_metrics = {}
                        if he_batch["available"] and labels_col is not None:
                            he_scores = np.array([r["decrypted_score"] for r in he_batch["records"]])
                            he_labels = pt.get_batch_labels(he_scores, spec)
                            he_metrics = _compute_metrics(labels_col, he_labels, he_scores)

                        results.append({
                            "model": spec,
                            "n_rows": n_rows,
                            "pt_metrics": pt_metrics,
                            "he_metrics": he_metrics,
                            "he_timing": he_batch.get("timing", {}),
                            "he_available": he_batch.get("available", False),
                        })

                    summary = {
                        "n_rows": n_rows,
                        "has_labels": labels_col is not None,
                        "n_fraud": int(labels_col.sum()) if labels_col is not None else None,
                        "n_legit": int((labels_col == 0).sum()) if labels_col is not None else None,
                    }

            except Exception as e:
                errors.append(f"Processing error: {e}")
                traceback.print_exc()

    return render_template(
        "batch.html",
        results=results,
        errors=errors,
        summary=summary,
        tenseal_available=he.is_available(),
        ckks_config=PRIMARY_CKKS_CONFIG,
    )


@app.route("/comparison")
def comparison():
    if not ARTIFACTS_LOADED:
        flash("Model artifacts not found. Run fraud_baseline_6d.ipynb first.", "error")
        return redirect(url_for("index"))

    # Use pre-computed HE eval subset
    X_eval = store.he_eval_features
    meta = store.he_eval_meta
    plaintext_ref = store.he_eval_plaintext

    if X_eval is None or meta is None:
        flash("HE evaluation subset not found in artifacts.", "error")
        return redirect(url_for("index"))

    labels = meta["isFraud"].values if "isFraud" in meta.columns else None
    models = store.get_all_models()

    comparison_data = []
    sweep_data = []

    for name, spec in models.items():
        # Plaintext
        pt_scores = pt.get_batch_scores(X_eval, spec, store)
        pt_labels = pt.get_batch_labels(pt_scores, spec)
        pt_metrics = _compute_metrics(labels, pt_labels, pt_scores) if labels is not None else {}

        # HE batch on primary config
        he_batch = he.predict_batch_he(X_eval, pt_scores, spec, store, PRIMARY_CKKS_CONFIG)

        he_metrics = {}
        if he_batch["available"] and labels is not None:
            he_scores_arr = np.array([r["decrypted_score"] for r in he_batch["records"]])
            he_labels_arr = pt.get_batch_labels(he_scores_arr, spec)
            he_metrics = _compute_metrics(labels, he_labels_arr, he_scores_arr)

        comparison_data.append({
            "model": spec,
            "pt_metrics": pt_metrics,
            "he_metrics": he_metrics,
            "he_timing": he_batch.get("timing", {}),
            "he_available": he_batch.get("available", False),
            "he_records": he_batch.get("records", []),
        })

        # Parameter sweep
        model_sweep = he.run_parameter_sweep(
            X_eval, pt_scores, spec, store, CKKS_PARAMETER_OPTIONS
        )
        sweep_data.extend(model_sweep)

    return render_template(
        "comparison.html",
        comparison_data=comparison_data,
        sweep_data=sweep_data,
        n_eval=len(X_eval),
        n_fraud=int(labels.sum()) if labels is not None else None,
        tenseal_available=he.is_available(),
        ckks_configs=CKKS_PARAMETER_OPTIONS,
        primary_config=PRIMARY_CKKS_CONFIG,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict:
    """Compute classification metrics for fraud detection."""
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "predicted_positive": int(y_pred.sum()),
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
