"""Fill MySQL from the dataset + training outputs.

    python seed_db.py            # after running train_cad_xgboost.py and applying schema.sql
"""
import json

import pandas as pd
from sqlalchemy import text

from config import DATASET_CSV, MODEL_DIR, TARGETS
from database import (Base, FeatureImportance, ModelMetric, SessionLocal,
                      col_name, engine)

Base.metadata.create_all(engine)

# 1) dataset -> patient_records
df = pd.read_csv(DATASET_CSV)
df.columns = [col_name(c) for c in df.columns]
with engine.begin() as conn:
    conn.execute(text("DELETE FROM patient_records"))
df.to_sql("patient_records", engine, if_exists="append", index=False, chunksize=100)
print(f"patient_records: {len(df)} rows")

# 2) metrics + importances -> model_metrics / feature_importance
db = SessionLocal()
db.query(ModelMetric).delete()
db.query(FeatureImportance).delete()
metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
for t, key in TARGETS.items():
    m = metrics[t]; cm = m["confusion_matrix"]
    db.add(ModelMetric(
        target=key, label=m["label"], n_train=m["n_train"], n_test=m["n_test"],
        positive_rate=m["positive_rate"], cv_auc_train=m["cv_auc_train"], test_auc=m["test_auc"],
        test_accuracy=m["test_accuracy"], test_precision=m["test_precision"],
        test_recall=m["test_recall_sensitivity"], test_specificity=m["test_specificity"],
        test_f1=m["test_f1"], tn=cm["TN"], fp=cm["FP"], fn=cm["FN"], tp=cm["TP"],
        best_params=m["best_params"]))
imp = pd.read_csv(MODEL_DIR / "feature_importances.csv", index_col=0)
for t, key in TARGETS.items():
    for rank, (feat, v) in enumerate(imp[t].sort_values(ascending=False).items(), 1):
        db.add(FeatureImportance(target=key, feature=feat, importance=float(v), rank_no=rank))
db.commit()
print("model_metrics + feature_importance loaded")
