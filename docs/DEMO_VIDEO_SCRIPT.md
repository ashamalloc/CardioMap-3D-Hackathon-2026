# 5–7 minute demo script

## 0:00–0:40 — Problem

“Raw cardiovascular risk scores are hard to interpret visually. CardioMap 3D links a clinical risk estimate to three coronary vessels while keeping a clear safety boundary: this is decision-support and educational visualization, not diagnosis.”

## 0:40–1:30 — Dataset and leakage prevention

Show the dataset source, the four targets, and the preprocessing diagram. Mention that LAD, LCX, RCA and Cath are removed from the feature matrix for all predictive tasks.

## 1:30–2:30 — Dashboard

Show the dynamic patient form. Edit Age, BP, cholesterol-related fields, ECG findings and echo features. Explain that the form is generated from the saved training schema.

## 2:30–3:30 — Prediction

Click **Predict & Map Risk**. Point to the CAD probability and the three vessel cards. Explain that one independent binary model produces each vessel probability.

## 3:30–4:30 — 3D interaction

Rotate the torso, zoom the heart and click LAD/LCX/RCA. Show that each vessel changes color according to its predicted probability and that selecting a vessel updates the detail panel.

## 4:30–5:30 — Explainability

Show SHAP contributions. Explain that one-hot categorical effects are aggregated back to the original clinical feature. Contrast local explanations with global model importance.

## 5:30–6:20 — Evaluation

Open the metrics view. Show accuracy, precision, recall, F1 and ROC-AUC for CAD and the three vessel targets, emphasizing that values come from the reproducible training script.

## 6:20–7:00 — Architecture and safety

Show: React/Three.js -> FastAPI -> model pipelines -> UCI data. Close on the safety disclaimer and explain that the viewer is a visual abstraction, not a radiological lesion map.
