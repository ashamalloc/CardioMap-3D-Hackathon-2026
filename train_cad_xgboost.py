"""
Train XGBoost classifiers for:
  - Overall CAD status  (column: Cath)
  - Stenosis per vessel (columns: LAD, LCX, RCA)

Usage:
    python train_cad_xgboost.py final_scaled_data.csv [output_dir]

Features = every column except the four targets.
Evaluation = stratified 5-fold CV on the training split + held-out 20% test set.
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from xgboost import XGBClassifier

TARGETS = {"Cath": "Overall CAD", "LAD": "LAD stenosis",
           "LCX": "LCX stenosis", "RCA": "RCA stenosis"}
SEED = 42

csv_path = sys.argv[1] if len(sys.argv) > 1 else "final_scaled_data.csv"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "cad_models"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)
features = [c for c in df.columns if c not in TARGETS]
X = df[features]
print(f"Rows: {len(df)}  Features: {len(features)}")

param_grid = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [100, 250],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.6, 0.9],
    "min_child_weight": [1, 3],
}

results, importances = {}, {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for target, label in TARGETS.items():
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    spw = (y_tr == 0).sum() / (y_tr == 1).sum()  # class imbalance weight
    base = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                         scale_pos_weight=spw, random_state=SEED, n_jobs=-1)

    # Tune on the training split only (no test leakage)
    gs = GridSearchCV(base, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    model = gs.best_estimator_

    # Honest CV estimate of the tuned model on train split
    cv_prob = cross_val_predict(model, X_tr, y_tr, cv=cv,
                                method="predict_proba")[:, 1]
    cv_auc = roc_auc_score(y_tr, cv_prob)

    # Held-out test
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    results[target] = {
        "label": label,
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "positive_rate": round(float(y.mean()), 3),
        "best_params": gs.best_params_,
        "cv_auc_train": round(float(cv_auc), 3),
        "test_auc": round(float(roc_auc_score(y_te, prob)), 3),
        "test_accuracy": round(float(accuracy_score(y_te, pred)), 3),
        "test_precision": round(float(precision_score(y_te, pred, zero_division=0)), 3),
        "test_recall_sensitivity": round(float(recall_score(y_te, pred)), 3),
        "test_specificity": round(float(tn / (tn + fp)), 3),
        "test_f1": round(float(f1_score(y_te, pred)), 3),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    importances[target] = imp
    joblib.dump(model, os.path.join(out_dir, f"xgb_{target}.joblib"))
    model.save_model(os.path.join(out_dir, f"xgb_{target}.json"))
    print(f"{target}: CV AUC {cv_auc:.3f} | test AUC {results[target]['test_auc']:.3f} "
          f"| acc {results[target]['test_accuracy']:.3f}")

# Save metrics + importances
with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)
pd.DataFrame(importances).to_csv(os.path.join(out_dir, "feature_importances.csv"))

# Plot top-10 features per target
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (target, imp) in zip(axes.ravel(), importances.items()):
    top = imp.head(10)[::-1]
    ax.barh(top.index, top.values, color="#3b6ea8")
    ax.set_title(f"{TARGETS[target]} – top 10 features")
    ax.set_xlabel("XGBoost importance (gain-based)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
print("Saved models, metrics.json, feature_importances.csv, feature_importance.png to", out_dir)
