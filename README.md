# CardioMap 3D – CAD risk prediction (React + FastAPI + MySQL + XGBoost)

```
cardiomap/
├── train_cad_xgboost.py     # your training script (unchanged)
├── final_scaled_data.csv    # dataset (303 rows)
├── cad_models/              # output of training: xgb_*.joblib, metrics.json, ...
├── backend/                 # FastAPI API that serves the models + talks to MySQL
├── database/schema.sql      # MySQL tables
└── frontend/                # React + Three.js app (your zip)
```

## 1. Train the models (skip: cad_models/ is already included)
```bash
pip install -r backend/requirements.txt matplotlib
python train_cad_xgboost.py final_scaled_data.csv cad_models
```

## 2. MySQL
```bash
mysql -u root -p < database/schema.sql
mysql -u root -p -e "CREATE USER 'cardiomap_user'@'localhost' IDENTIFIED BY 'change_me';
                     GRANT ALL ON cardiomap.* TO 'cardiomap_user'@'localhost';"
```
Tables: `patient_records` (dataset), `predictions` (every API prediction),
`model_metrics`, `feature_importance` (from training output).

## 3. Backend
```bash
cd backend
cp .env.example .env          # set DATABASE_URL password
pip install -r requirements.txt
python seed_db.py             # loads dataset + metrics into MySQL
uvicorn main:app --reload --port 8000
```
Endpoints: `GET /api/status`, `/api/schema`, `/api/metrics`, `/api/importance`,
`/api/history`, `POST /api/predict`. Docs at http://localhost:8000/docs.
If MySQL is down, predictions still work (they just aren't logged) and metrics fall back to `cad_models/metrics.json`.

## 4. Frontend
```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173
```
(`VITE_API_URL` defaults to http://localhost:8000.)

## Notes
- FBS, TG, WBC, PLT are stored as natural log in the CSV. The form/API use raw lab units and the backend applies `ln()` before predicting.
- The explanation bars use XGBoost's built-in SHAP-style contributions (log-odds).
- The frontend "Method" tab still says Random Forest/SHAP; the models are XGBoost. Edit the text in `frontend/src/App.jsx` if you want it to match.
- Educational prototype, not a medical device.
