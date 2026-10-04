# CardioMap 3D — Multimodal AI Hackathon 2026 / Track A

A reproducible, leakage-aware prototype for cardiovascular risk visualization and coronary-vessel stenosis prediction using the **UCI Extension of Z-Alizadeh Sani dataset**.

## What is included

- Leakage-safe training pipeline for four binary classification tasks:
  - Overall CAD status
  - LAD stenosis
  - LCX stenosis
  - RCA stenosis
- Hold-out and stratified 5-fold validation with accuracy, precision, recall, F1 and ROC-AUC.
- Random Forest probability models with class balancing.
- SHAP-based local explanations, aggregated back to clinical features.
- FastAPI backend with dynamic clinical-input schema.
- React + Three.js browser dashboard with interactive torso/heart/coronary visualization.
- Dynamic LAD / LCX / RCA coloring from predicted stenosis probability.
- Clinical feature contribution panel and physiological measurements.
- Safety disclaimer and provenance information.
- Windows PowerShell and cross-platform launch scripts.
- Optional demo-model mode strictly for UI smoke testing; it is marked non-clinical and does not report medical performance.

## Important dataset note

The UCI repository lists 303 instances and states that CAD is defined by >=50% diameter narrowing. It also explicitly notes that the extension adds LAD, LCX and RCA labels and that **LAD, LCX, RCA and Cath should be removed from model inputs to prevent target leakage**. This implementation therefore excludes all four from every predictive feature matrix. Source: UCI dataset 411, DOI 10.24432/C5461K.

The dataset is CC BY 4.0. Keep the attribution in `docs/DATASET_ATTRIBUTION.md` when redistributing it.

## 1. Prerequisites

Recommended:

- Python 3.11–3.13
- Node.js 20+
- npm 10+
- Internet access during the first training run (to fetch UCI data with `ucimlrepo`) or a local `.xlsx` copy of the dataset.

## 2. Windows setup

From PowerShell:

```powershell
cd cardiac_3d_hackathon
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\scripts\setup.ps1
.\scripts\train.ps1
.\scripts\run.ps1
```

The dashboard will normally be available at `http://localhost:5173` and the API at `http://localhost:8000`.

## 3. Cross-platform setup

Backend:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows cmd
.venv\\Scripts\\activate

pip install -r backend/requirements.txt
python -m ml_pipeline.train
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## 4. Local dataset mode

Place the Excel workbook in `data/extention_of_z_alizadeh_sani_dataset.xlsx`, then run:

```bash
python -m ml_pipeline.train --data data/extention_of_z_alizadeh_sani_dataset.xlsx
```

This is useful for an offline environment or when the organizers give you a specific copy of the dataset.

## 5. Model artifacts

After training, the following are created:

```text
models/
  cad_model.joblib
  lad_model.joblib
  lcx_model.joblib
  rca_model.joblib
  schema.json
  metrics.json
```

These files are intentionally not pre-generated in the ZIP because no dataset bytes were available in the conversation and it would be scientifically invalid to invent clinical training results.

## 6. API

- `GET /api/status`
- `GET /api/schema`
- `GET /api/metrics`
- `POST /api/predict`
- `GET /docs` for Swagger/OpenAPI

Example request:

```json
{
  "features": {
    "Age": 58,
    "Sex": "Male",
    "BP": 140,
    "Pulse Rate": 82
  }
}
```

The server fills omitted features using training-set medians/modes. The UI always displays these defaults as editable fields, so reviewers can build a full patient profile.

## 7. Model design

Each target gets an independent pipeline:

`raw clinical table -> identifier/leakage removal -> numeric/categorical preprocessing -> RandomForestClassifier -> probability`

The feature matrix for every task removes `LAD`, `LCX`, `RCA`, `Cath`, the task target itself, and obvious patient identifiers. There is no direct use of angiographic labels as predictors.

### Why independent models?

The competition asks for overall CAD plus three vessel-specific outcomes. Independent binary estimators keep each target explicit and make the 3D mapping deterministic: one probability -> one anatomical vessel.

## 8. Explainability

SHAP TreeExplainer is used for local explanations. One-hot categorical contributions are aggregated back to the original clinical feature so the dashboard can say, for example, that a patient's BP or ST Elevation contributed to a higher predicted likelihood.

If SHAP is unavailable at runtime, the backend falls back to model-native feature importance for global context; it does not silently claim that those are local SHAP values.

## 9. 3D visualization

The viewer uses Three.js and procedural geometry instead of shipping a third-party anatomical mesh. This avoids asset-license ambiguity and keeps the browser lightweight:

- translucent stylized human torso
- heart volume
- aortic root
- LAD / LCX / RCA coronary tubes and node markers
- OrbitControls for rotate / zoom / pan
- click a vessel to inspect its probability and explanation

The coronary geometry is a **visual risk mapping abstraction**, not a radiological reconstruction and not an anatomical localization inferred from the tabular features.

## 10. Safety

This is a hackathon decision-support / educational prototype only. It is not a medical device, does not replace ECG, echocardiography, CT coronary angiography or invasive angiography, and must not be used for clinical diagnosis or treatment decisions.

The model sees a small, historical dataset. Metrics are intended for benchmarking the prototype, not for clinical validation.

## 11. Demonstration flow

1. Open the dashboard.
2. Wait for `Models ready` in the header.
3. Edit the patient fields.
4. Click **Predict & Map Risk**.
5. Observe CAD likelihood and the three vessel probabilities.
6. Rotate / zoom the 3D model.
7. Click LAD, LCX or RCA nodes.
8. Inspect local SHAP contributions and the physiological value table.
9. Open the Metrics tab to show validation results.
10. Point out the visible clinical safety disclaimer.

## 12. Project structure

```text
cardiac_3d_hackathon/
├── backend/
│   ├── app/main.py
│   └── requirements.txt
├── ml_pipeline/
│   ├── __init__.py
│   ├── data.py
│   ├── modeling.py
│   └── train.py
├── models/
├── data/
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
├── docs/
├── scripts/
└── tests/
```

## Sources

- UCI Extension of Z-Alizadeh Sani dataset: https://archive.ics.uci.edu/dataset/411/extention%2Bof%2Bz%2B
- DOI: https://doi.org/10.24432/C5461K
- BodyParts3D licensing reference: https://wiki.lifesciencedb.jp/mw/%E5%88%A9%E7%94%A8%E3%83%A9%E3%82%A4%E3%82%BB%E3%83%B3%E3%82%B9.html
