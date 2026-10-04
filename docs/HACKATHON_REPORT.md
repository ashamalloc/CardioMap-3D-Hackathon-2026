# CardioMap 3D — Track A Technical Report

## 1. Problem and objective

CardioMap 3D converts a clinical tabular risk estimate into an interactive anatomical visualization. The system predicts overall coronary artery disease (CAD) status plus binary stenosis status for the LAD, LCX and RCA, then maps those three probabilities to a lightweight 3D heart/torso viewer.

The UCI Extension of Z-Alizadeh Sani dataset contains 303 patient records and 59 variables spanning demographic, symptom/examination, ECG, laboratory and echocardiographic information. UCI defines CAD using a diameter narrowing threshold of at least 50% and specifically describes the added LAD, LCX and RCA labels. The dataset is CC BY 4.0.

## 2. Data preprocessing

The training script supports either the UCI `ucimlrepo` loader or a local Excel workbook. Column names are normalized only for matching; original names are preserved in the feature schema. Obvious identifiers are removed. Missing values are imputed inside the model pipeline rather than before splitting, preventing train/test contamination.

Categorical features are mode-imputed and one-hot encoded. Numeric features are median-imputed. Random Forests do not require standard scaling, so no unnecessary scaling is applied.

### Leakage control

For **every** target, the model matrix excludes `LAD`, `LCX`, `RCA` and `Cath`. The target column itself is also excluded. This directly follows the UCI dataset warning about angiographic target leakage.

## 3. Predictive modeling

Four independent `RandomForestClassifier` estimators are fitted with stratified train/test splitting, class balancing, 500 trees and a minimum leaf size of 2. The default held-out test split is 20% with `random_state=42`.

To provide a stronger validation picture for a hackathon prototype, the training pipeline also computes stratified 5-fold cross-validation scores on the training partition for accuracy, precision, recall, F1 and ROC-AUC. Both hold-out and CV results are exported to `models/metrics.json`.

No clinical performance numbers are hard-coded into this report. The actual metrics shown by the dashboard are generated only after the real UCI data have been downloaded and the pipeline has been executed.

## 4. Explainability

The application uses SHAP TreeExplainer on the fitted Random Forest. Because categorical variables are one-hot encoded, individual one-hot contributions are aggregated to the originating raw feature. The dashboard therefore presents explanations at the level of clinical variables instead of exposing encoding implementation details.

The top positive and negative local contributions are shown next to the prediction. A separate feature-importance view is provided as global context.

## 5. 3D system

The frontend is a React + Three.js application. To avoid third-party anatomical asset licensing complexity, the prototype uses procedural geometry: torso, heart, aortic root and three simplified coronary curves. The three coronary structures have stable IDs (`LAD`, `LCX`, `RCA`) and are colored by predicted probability.

The visualization supports orbit rotation, zoom and click-to-inspect interactions. It is explicitly presented as a **risk-mapping abstraction** rather than a lesion map or image-derived anatomy.

Risk coloring:

- lower probability: green/teal
- intermediate probability: amber
- higher probability: red

## 6. Integration and safety

FastAPI loads serialized model pipelines at startup. The frontend fetches the schema dynamically, so adding more supported clinical features does not require redesigning the form. Predictions are returned as JSON and immediately update both dashboard cards and 3D vessel objects.

The UI includes a persistent clinical disclaimer: the system is for educational/decision-support prototyping only and is not a substitute for diagnostic imaging or clinical judgment. The dataset is small and historical; this prototype is not clinically validated.

## Reproducibility checklist

- UCI dataset source and DOI recorded.
- Leakage exclusion implemented in one central module.
- Fixed random seed for the benchmark split.
- Cross-validation metrics exported.
- Model artifacts serialized with Joblib.
- Frontend and backend dependencies pinned to compatible major/minor ranges.
- Training, run and smoke-test instructions supplied.
