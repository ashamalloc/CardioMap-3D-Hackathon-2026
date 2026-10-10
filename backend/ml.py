"""Loads the XGBoost models produced by train_cad_xgboost.py and runs predictions."""
import json
import math

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from config import DATASET_CSV, LOG_FEATURES, MODEL_DIR, TARGETS


def band(p: float) -> str:
    return "Low" if p < 0.33 else "Intermediate" if p < 0.66 else "Higher"


class CADModels:
    def __init__(self):
        self.models = {}
        self.features = []
        self.fields = []
        self._load_schema()
        self.reload()

    # ---------- schema (built from the training CSV) ----------
    def _load_schema(self):
        df = pd.read_csv(DATASET_CSV)
        self.features = [c for c in df.columns if c not in TARGETS]
        fields = []
        for c in self.features:
            s = df[c]
            if c in LOG_FEATURES:                      # show raw lab units, convert in predict()
                raw = np.exp(s)
                fields.append({"name": c, "kind": "number", "unit_note": "raw units (log applied server-side)",
                               "default": round(float(raw.median()), 1),
                               "min": round(float(raw.min()), 1), "max": round(float(raw.max()), 1), "step": "any"})
            elif s.nunique() <= 5:
                opts = [str(int(v)) for v in sorted(s.unique())]
                fields.append({"name": c, "kind": "select", "options": opts,
                               "default": str(int(s.mode()[0]))})
            else:
                fields.append({"name": c, "kind": "number", "default": round(float(s.median()), 2),
                               "min": round(float(s.min()), 2), "max": round(float(s.max()), 2), "step": "any"})
        self.fields = fields
        self._defaults = {f["name"]: float(f["default"]) for f in fields}

    # ---------- models ----------
    def reload(self):
        self.models = {}
        for target, key in TARGETS.items():
            p = MODEL_DIR / f"xgb_{target}.joblib"
            if p.exists():
                self.models[key] = joblib.load(p)

    @property
    def ready(self) -> bool:
        return len(self.models) == len(TARGETS)

    # ---------- file-based metrics (fallback when MySQL is empty) ----------
    def file_metrics(self):
        p = MODEL_DIR / "metrics.json"
        if not p.exists():
            return {}
        raw = json.loads(p.read_text())
        out = {}
        for t, key in TARGETS.items():
            m = raw.get(t)
            if m:
                out[key] = {"label": m["label"], "cv_auc": m["cv_auc_train"], "holdout": {
                    "accuracy": m["test_accuracy"], "precision": m["test_precision"],
                    "recall": m["test_recall_sensitivity"], "specificity": m["test_specificity"],
                    "f1": m["test_f1"], "roc_auc": m["test_auc"]},
                    "confusion_matrix": m["confusion_matrix"]}
        return out

    def file_importance(self, top=15):
        p = MODEL_DIR / "feature_importances.csv"
        if not p.exists():
            return {}
        df = pd.read_csv(p, index_col=0)
        return {TARGETS[t]: [{"feature": f, "importance": round(float(v), 5)}
                             for f, v in df[t].sort_values(ascending=False).head(top).items()]
                for t in TARGETS if t in df}

    # ---------- prediction ----------
    def build_row(self, values: dict) -> pd.DataFrame:
        row = {}
        for f in self.features:
            v = values.get(f, None)
            if v is None or v == "":
                v = self._defaults[f]
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"'{f}' must be a number")
            if not math.isfinite(v):
                raise ValueError(f"'{f}' must be a finite number")
            if f in LOG_FEATURES:
                if v <= 0:
                    raise ValueError(f"'{f}' must be greater than 0")
                v = math.log(v)
            row[f] = v
        return pd.DataFrame([row], columns=self.features)

    def predict(self, values: dict) -> dict:
        X = self.build_row(values)
        dm = xgb.DMatrix(X)
        res = {}
        for key, model in self.models.items():
            prob = float(model.predict_proba(X)[0, 1])
            contribs = model.get_booster().predict(dm, pred_contribs=True)[0][:-1]  # drop bias
            order = np.argsort(-np.abs(contribs))[:8]
            res[key] = {"probability": prob, "percent": prob * 100, "band": band(prob),
                        "explanations": [{"feature": self.features[i],
                                          "contribution": float(contribs[i])} for i in order]}
        vessels = {v: res[v] for v in ("LAD", "LCX", "RCA")}
        return {"overall_cad": res["CAD"], "vessels": vessels,
                "highest_vessel": max(vessels, key=lambda v: vessels[v]["probability"])}
