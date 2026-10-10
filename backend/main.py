"""CardioMap 3D API.  Run:  uvicorn main:app --reload --port 8000"""
import logging

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import CORS_ORIGINS, TARGETS
from database import (Base, FeatureImportance, ModelMetric, Prediction,
                      SessionLocal, engine)
from ml import CADModels

log = logging.getLogger("cardiomap")
app = FastAPI(title="CardioMap 3D API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

cad = CADModels()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def db_online() -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("Database unavailable: %s", e)
        return False


@app.on_event("startup")
def startup():
    try:
        Base.metadata.create_all(engine)   # no-op if schema.sql was already applied
    except Exception as e:  # noqa: BLE001
        log.warning("Could not create tables: %s", e)


@app.get("/api/status")
def status():
    return {"ready": cad.ready, "models": sorted(cad.models), "database": db_online(),
            "feature_count": len(cad.features)}


@app.get("/api/schema")
def schema():
    return {"count": len(cad.fields), "fields": cad.fields}


@app.get("/api/metrics")
def metrics(db: Session = Depends(get_db)):
    try:
        rows = db.query(ModelMetric).all()
        if rows:
            return {r.target: {"label": r.label, "cv_auc": r.cv_auc_train, "holdout": {
                "accuracy": r.test_accuracy, "precision": r.test_precision, "recall": r.test_recall,
                "specificity": r.test_specificity, "f1": r.test_f1, "roc_auc": r.test_auc},
                "confusion_matrix": {"TN": r.tn, "FP": r.fp, "FN": r.fn, "TP": r.tp}} for r in rows}
    except Exception as e:  # noqa: BLE001
        log.warning("metrics from DB failed, using file: %s", e)
    return cad.file_metrics()


@app.get("/api/importance")
def importance(db: Session = Depends(get_db)):
    try:
        rows = (db.query(FeatureImportance).filter(FeatureImportance.rank_no <= 15)
                .order_by(FeatureImportance.target, FeatureImportance.rank_no).all())
        if rows:
            out = {}
            for r in rows:
                out.setdefault(r.target, []).append({"feature": r.feature, "importance": r.importance})
            return out
    except Exception as e:  # noqa: BLE001
        log.warning("importance from DB failed, using file: %s", e)
    return cad.file_importance()


@app.post("/api/predict")
def predict(request: Request, payload: dict = Body(...), db: Session = Depends(get_db)):
    if not cad.ready:
        raise HTTPException(503, {"message": "Models not trained yet. Run train_cad_xgboost.py first."})
    features = payload.get("features")
    if not isinstance(features, dict):
        raise HTTPException(422, {"message": "Body must be {\"features\": {...}}"})
    try:
        result = cad.predict(features)
    except ValueError as e:
        raise HTTPException(422, {"message": str(e)})

    try:  # log to MySQL, never fail the prediction because of the DB
        db.add(Prediction(
            input_features=features,
            cad_probability=result["overall_cad"]["probability"],
            lad_probability=result["vessels"]["LAD"]["probability"],
            lcx_probability=result["vessels"]["LCX"]["probability"],
            rca_probability=result["vessels"]["RCA"]["probability"],
            highest_vessel=result["highest_vessel"],
            client_ip=request.client.host if request.client else None))
        db.commit()
    except Exception as e:  # noqa: BLE001
        db.rollback()
        log.warning("Could not save prediction: %s", e)
    return result


@app.get("/api/history")
def history(limit: int = 20, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 200))
    try:
        rows = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).all()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, {"message": f"Database unavailable: {e}"})
    return [{"id": r.id, "created_at": r.created_at.isoformat(), "cad": r.cad_probability,
             "LAD": r.lad_probability, "LCX": r.lcx_probability, "RCA": r.rca_probability,
             "highest_vessel": r.highest_vessel} for r in rows]


@app.post("/api/admin/reload")
def reload_models():
    cad.reload()
    return {"ready": cad.ready}
