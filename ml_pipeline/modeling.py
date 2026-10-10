from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class FitResult:
    pipeline: Pipeline
    metrics: dict[str, Any]
    schema: dict[str, Any]
    explain_map: dict[str, str]


def _feature_kind(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


def make_pipeline(X: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    numeric, categorical = _feature_kind(X)
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    clf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", pre), ("model", clf)]), numeric, categorical


def _safe_metric(fn, y_true, y_pred, **kwargs):
    try:
        return float(fn(y_true, y_pred, **kwargs))
    except Exception:
        return None


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float | None]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    result: dict[str, float | None] = {
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "precision": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe_metric(f1_score, y_true, y_pred, zero_division=0),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        result["roc_auc"] = None
    return result


def build_schema(X: pd.DataFrame, numeric: list[str]) -> dict[str, Any]:
    fields = []
    for col in X.columns:
        s = X[col]
        if col in numeric:
            clean = pd.to_numeric(s, errors="coerce")
            default = float(clean.median()) if clean.notna().any() else 0.0
            fields.append({
                "name": col,
                "kind": "number",
                "default": default,
                "min": float(clean.min()) if clean.notna().any() else None,
                "max": float(clean.max()) if clean.notna().any() else None,
                "step": 1 if clean.notna().any() and np.isclose(clean.dropna().astype(float) % 1, 0).all() else 0.1,
            })
        else:
            values = [str(v) for v in s.dropna().unique().tolist()]
            mode = str(s.mode(dropna=True).iloc[0]) if not s.mode(dropna=True).empty else (values[0] if values else "Unknown")
            fields.append({
                "name": col,
                "kind": "select",
                "options": values[:30],
                "default": mode,
            })
    return {"fields": fields, "count": len(fields)}


def build_explain_map(pipeline: Pipeline) -> dict[str, str]:
    pre = pipeline.named_steps["preprocess"]
    names = pre.get_feature_names_out()
    cat_cols = list(pre.transformers_[1][2])
    mapping: dict[str, str] = {}
    for out_name in names:
        if out_name.startswith("num__"):
            mapping[out_name] = out_name[len("num__"):]
            continue
        rest = out_name[len("cat__"):] if out_name.startswith("cat__") else out_name
        found = None
        for col in cat_cols:
            if rest == col or rest.startswith(f"{col}_"):
                found = col
                break
        mapping[out_name] = found or rest
    return mapping


def fit_one_target(X: pd.DataFrame, y: pd.Series, target: str, output_dir: Path) -> FitResult:
    pipeline, numeric, _ = make_pipeline(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")

    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    holdout = compute_metrics(y_test, probs)

    metrics = {
        "target": target,
        "n_samples": int(len(X)),
        "positive_rate": float(y.mean()),
        "holdout": holdout,
        "cv5_mean": {k: float(np.mean(v)) for k, v in [(m, cv_result[f"test_{m}"]) for m in scoring]},
        "cv5_std": {k: float(np.std(v)) for k, v in [(m, cv_result[f"test_{m}"]) for m in scoring]},
        "split": {"test_size": 0.20, "random_state": 42},
        "model": "RandomForestClassifier",
        "leakage_excluded": ["LAD", "LCX", "RCA", "Cath"],
        "features_used": list(X.columns),
    }

    out_path = output_dir / f"{target.lower()}_model.joblib"
    joblib.dump(pipeline, out_path)
    schema = build_schema(X, numeric)
    explain_map = build_explain_map(pipeline)
    return FitResult(pipeline=pipeline, metrics=metrics, schema=schema, explain_map=explain_map)


def train_all(X: pd.DataFrame, ys: dict[str, pd.Series], output_dir: Path) -> dict[str, FitResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    imported_lcx_model = output_dir / "lcx_xgb.json"
    preserved_lcx_metrics: dict[str, Any] | None = None
    if imported_lcx_model.is_file():
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"Cannot preserve imported LCX model without its metrics file: {metrics_path}"
            )
        with metrics_path.open("r", encoding="utf-8") as f:
            existing_metrics = json.load(f)
        if "LCX" not in existing_metrics:
            raise ValueError(f"Metrics file does not contain the imported LCX model report: {metrics_path}")
        preserved_lcx_metrics = existing_metrics["LCX"]

    results = {}
    for target in ["CAD", "LAD", "LCX", "RCA"]:
        if target == "LCX" and preserved_lcx_metrics is not None:
            continue
        results[target] = fit_one_target(X, ys[target], target, output_dir)

    # The three CardioMap Random Forest models share the same leakage-safe schema.
    with (output_dir / "schema.json").open("w", encoding="utf-8") as f:
        json.dump(results["CAD"].schema, f, indent=2)
    metrics = {k: v.metrics for k, v in results.items()}
    if preserved_lcx_metrics is not None:
        metrics["LCX"] = preserved_lcx_metrics
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "explain_maps.json").open("w", encoding="utf-8") as f:
        json.dump({k: v.explain_map for k, v in results.items()}, f, indent=2)
    return results
