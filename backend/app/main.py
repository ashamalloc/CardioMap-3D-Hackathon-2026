from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
PROJECT_LCX_MODEL = MODEL_DIR / "lcx_xgb.json"
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
PROJECT_LCX_LOG_FEATURES = {"FBS", "TG", "WBC", "PLT"}
PROJECT_LCX_MISSING_DEFAULTS = {"Thyroid Disease": 0.0}

app = FastAPI(
    title="CardioMap 3D API",
    version="1.0.0",
    description="Leakage-aware CAD/vessel prediction API for a hackathon prototype.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS: dict[str, Any] = {}
SCHEMA: dict[str, Any] = {}
METRICS: dict[str, Any] = {}
EXPLAIN_MAPS: dict[str, dict[str, str]] = {}
EXPLAINERS: dict[str, Any] = {}


def _load_artifacts() -> None:
    global MODELS, SCHEMA, METRICS, EXPLAIN_MAPS, EXPLAINERS
    MODELS = {}
    EXPLAINERS = {}
    for target in TARGETS:
        if target == "LCX":
            if PROJECT_LCX_MODEL.exists():
                model = XGBClassifier()
                model.load_model(PROJECT_LCX_MODEL)
                MODELS[target] = model
            continue

        path = MODEL_DIR / f"{target.lower()}_model.joblib"
        if path.exists():
            MODELS[target] = joblib.load(path)
            try:
                EXPLAINERS[target] = shap.TreeExplainer(MODELS[target].named_steps["model"])
            except Exception:
                EXPLAINERS[target] = None
    for name, default in [("schema.json", {}), ("metrics.json", {}), ("explain_maps.json", {})]:
        p = MODEL_DIR / name
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = default
        if name == "schema.json":
            SCHEMA = payload
        elif name == "metrics.json":
            METRICS = payload
        else:
            EXPLAIN_MAPS = payload


_load_artifacts()


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)


def _require_models() -> None:
    missing = [t for t in TARGETS if t not in MODELS]
    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Models are not trained yet.",
                "missing": missing,
                "command": "python -m ml_pipeline.train",
            },
        )


def _feature_frame(features: dict[str, Any]) -> pd.DataFrame:
    fields = SCHEMA.get("fields", [])
    names = [field["name"] for field in fields]
    unknown_fields = set(features).difference(names)
    if unknown_fields:
        raise ValueError(f"Unknown feature(s): {', '.join(sorted(unknown_fields))}")

    row: dict[str, Any] = {}
    for field in fields:
        name = field["name"]
        value = features.get(name, field.get("default"))
        if value is None or value == "":
            value = field.get("default")

        if field.get("kind") == "number":
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"'{name}' must be a number") from exc
            if not math.isfinite(number):
                raise ValueError(f"'{name}' must be a finite number")
            row[name] = number
        else:
            category = str(value)
            options = field.get("options")
            if options and category not in options:
                raise ValueError(
                    f"'{name}' must be one of: {', '.join(options)}"
                )
            row[name] = category
    # Keep exact training column order.
    return pd.DataFrame([row], columns=names)


def _project_lcx_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    model = MODELS["LCX"]
    feature_names = model.get_booster().feature_names
    if not feature_names:
        raise RuntimeError("The project LCX model has no saved feature names.")

    row: dict[str, float] = {}
    category_codes = {
        "Sex": {"Fmale": 0, "Male": 1},
        "BBB": {"N": 0, "LBBB": 1, "RBBB": 2},
        "VHD": {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3},
    }
    for name in feature_names:
        if name in X.columns:
            value = X.iloc[0][name]
        elif name in PROJECT_LCX_MISSING_DEFAULTS:
            value = PROJECT_LCX_MISSING_DEFAULTS[name]
        else:
            raise RuntimeError(f"CardioMap has no input/default for project LCX feature '{name}'.")

        if name in category_codes:
            try:
                value = category_codes[name][str(value)]
            except KeyError as exc:
                raise ValueError(f"'{name}' has an unsupported value") from exc
        elif isinstance(value, str) and value in {"Y", "N"}:
            value = 1 if value == "Y" else 0

        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{name}' must be a number or supported category") from exc
        if not math.isfinite(value):
            raise ValueError(f"'{name}' must be a finite number")
        if name in PROJECT_LCX_LOG_FEATURES:
            if value <= 0:
                raise ValueError(f"'{name}' must be greater than 0")
            value = math.log(value)
        row[name] = value

    return pd.DataFrame([row], columns=feature_names)


def _risk_band(prob: float) -> str:
    if prob < 0.33:
        return "Lower likelihood"
    if prob < 0.66:
        return "Intermediate likelihood"
    return "Higher likelihood"


def _shap_contributions(target: str, X: pd.DataFrame) -> list[dict[str, Any]]:
    if target == "LCX":
        model = MODELS[target]
        contributions = model.get_booster().predict(
            xgb.DMatrix(X, feature_names=list(X.columns)),
            pred_contribs=True,
        )[0][:-1]
        order = np.argsort(-np.abs(contributions))[:10]
        return [
            {
                "feature": X.columns[index],
                "contribution": round(float(contributions[index]), 5),
                "direction": "increases" if contributions[index] >= 0 else "decreases",
            }
            for index in order
        ]

    explainer = EXPLAINERS.get(target)
    model = MODELS[target]
    if explainer is None:
        return []
    try:
        transformed = model.named_steps["preprocess"].transform(X)
        feature_names = list(model.named_steps["preprocess"].get_feature_names_out())
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        arr = np.asarray(values)
        if arr.ndim == 3:
            vals = arr[0, :, 1]
        elif arr.ndim == 2:
            vals = arr[0]
        else:
            vals = arr.reshape(-1)
        mapping = EXPLAIN_MAPS.get(target, {})
        grouped: dict[str, float] = {}
        for name, value in zip(feature_names, vals):
            raw = mapping.get(name, name)
            grouped[raw] = grouped.get(raw, 0.0) + float(value)
        ordered = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        return [
            {"feature": k, "contribution": round(v, 5), "direction": "increases" if v >= 0 else "decreases"}
            for k, v in ordered
        ]
    except Exception:
        return []


def _global_importance(target: str) -> list[dict[str, Any]]:
    model = MODELS[target]
    if target == "LCX":
        feature_names = model.get_booster().feature_names
        if not feature_names:
            raise RuntimeError("The project LCX model has no saved feature names.")
        scores = model.get_booster().get_score(importance_type="gain")
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:12]
        return [
            {"feature": feature, "importance": round(float(value), 5)}
            for feature, value in ranked
        ]

    clf = model.named_steps["model"]
    pre = model.named_steps["preprocess"]
    names = list(pre.get_feature_names_out())
    importances = getattr(clf, "feature_importances_", np.zeros(len(names)))
    mapping = EXPLAIN_MAPS.get(target, {})
    grouped: dict[str, float] = {}
    for name, value in zip(names, importances):
        raw = mapping.get(name, name)
        grouped[raw] = grouped.get(raw, 0.0) + float(value)
    ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:12]
    return [{"feature": k, "importance": round(v, 5)} for k, v in ranked]


@app.get("/api/status")
def status() -> dict[str, Any]:
    ready = all(t in MODELS for t in TARGETS)
    return {
        "ready": ready,
        "models_ready": [t for t in TARGETS if t in MODELS],
        "missing_models": [t for t in TARGETS if t not in MODELS],
        "feature_count": SCHEMA.get("count", 0),
        "message": "Models ready" if ready else "Train models first with: python -m ml_pipeline.train",
    }


@app.get("/api/schema")
def schema() -> dict[str, Any]:
    return SCHEMA


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    return METRICS


@app.get("/api/importance")
def importance() -> dict[str, Any]:
    _require_models()
    return {target: _global_importance(target) for target in TARGETS}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    _require_models()
    try:
        X = _feature_frame(req.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    predictions: dict[str, Any] = {}
    for target in TARGETS:
        model = MODELS[target]
        try:
            model_X = _project_lcx_feature_frame(X) if target == "LCX" else X
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        probability = float(model.predict_proba(model_X)[0, 1])
        predictions[target] = {
            "probability": round(probability, 4),
            "percent": round(probability * 100, 1),
            "status": "Positive" if probability >= 0.5 else "Negative",
            "band": _risk_band(probability),
            "explanations": _shap_contributions(target, model_X),
        }

    vessel_probs = {v: predictions[v]["probability"] for v in ["LAD", "LCX", "RCA"]}
    return {
        "overall_cad": predictions["CAD"],
        "vessels": {v: predictions[v] for v in vessel_probs},
        "overall_risk_note": "Overall risk score is the model-estimated CAD likelihood; it is not a validated clinical risk score.",
        "highest_vessel": max(vessel_probs, key=vessel_probs.get),
        "patient_features": X.iloc[0].to_dict(),
        "model_version": "CardioMap RF v1 + proj XGBoost LCX",
    }
