from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
TARGETS = ["CAD", "LAD", "LCX", "RCA"]

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
    names = [f["name"] for f in SCHEMA.get("fields", [])]
    row: dict[str, Any] = {}
    for field in SCHEMA.get("fields", []):
        name = field["name"]
        value = features.get(name, field.get("default"))
        if field.get("kind") == "number":
            try:
                row[name] = float(value)
            except (TypeError, ValueError):
                row[name] = field.get("default")
        else:
            row[name] = str(value) if value is not None else field.get("default")
    # Keep exact training column order.
    return pd.DataFrame([row], columns=names)


def _risk_band(prob: float) -> str:
    if prob < 0.33:
        return "Lower likelihood"
    if prob < 0.66:
        return "Intermediate likelihood"
    return "Higher likelihood"


def _shap_contributions(target: str, X: pd.DataFrame) -> list[dict[str, Any]]:
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
    X = _feature_frame(req.features)
    predictions: dict[str, Any] = {}
    for target in TARGETS:
        model = MODELS[target]
        probability = float(model.predict_proba(X)[0, 1])
        predictions[target] = {
            "probability": round(probability, 4),
            "percent": round(probability * 100, 1),
            "status": "Positive" if probability >= 0.5 else "Negative",
            "band": _risk_band(probability),
            "explanations": _shap_contributions(target, X),
        }

    cad = predictions["CAD"]["probability"]
    vessel_probs = {v: predictions[v]["probability"] for v in ["LAD", "LCX", "RCA"]}
    return {
        "overall_cad": predictions["CAD"],
        "vessels": {v: predictions[v] for v in vessel_probs},
        "overall_risk_note": "Overall risk score is the model-estimated CAD likelihood; it is not a validated clinical risk score.",
        "highest_vessel": max(vessel_probs, key=vessel_probs.get),
        "patient_features": X.iloc[0].to_dict(),
        "model_version": "CardioMap RF v1",
    }
