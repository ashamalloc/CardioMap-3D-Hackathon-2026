# Model card

## Intended use
Hackathon research/education prototype for visualizing estimated CAD and major-vessel stenosis probabilities from structured clinical inputs.

## Out of scope
Diagnosis, treatment selection, emergency triage, patient-specific procedural planning, or replacing diagnostic imaging.

## Training data
UCI Extension of Z-Alizadeh Sani dataset, 303 records, CC BY 4.0.

## Targets
CAD, LAD, LCX, RCA.

## Input exclusions
LAD, LCX, RCA and Cath are excluded from every model input matrix. Obvious identifiers are also excluded.

## Model
Random Forest classifier with median/mode imputation and one-hot encoding inside a single scikit-learn Pipeline.

## Validation
Fixed stratified 20% hold-out plus stratified 5-fold cross-validation on the training partition.

## Limitations
Small sample size; historical dataset; potential dataset-specific selection effects; no external validation; no calibration guarantee; no clinical outcome modeling; no imaging input.

## Explainability
SHAP TreeExplainer for local attribution, aggregated from one-hot categories to original features.
