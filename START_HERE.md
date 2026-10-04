# START HERE

This ZIP is the complete CardioMap 3D source project.

### First run on Windows

Open PowerShell in this folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\scripts\setup.ps1
.\scripts\train.ps1
.\scripts\run.ps1
```

Then open `http://localhost:5173`.

### What the first training run does

1. Downloads UCI dataset 411 via `ucimlrepo`.
2. Removes LAD/LCX/RCA/Cath plus identifier columns from the feature matrix.
3. Trains four Random Forest pipelines.
4. Computes hold-out + stratified 5-fold metrics.
5. Writes model artifacts to `models/`.
6. Writes the clinical-input schema and explainability mapping.

### If internet is blocked

Download the UCI Excel workbook yourself, place it at:

`data/extention_of_z_alizadeh_sani_dataset.xlsx`

Then run:

```powershell
.\.venv\Scripts\python.exe -m ml_pipeline.train --data data\extention_of_z_alizadeh_sani_dataset.xlsx
```

### Important

No fabricated clinical model weights or benchmark numbers are shipped. The project is designed to train on the specified UCI dataset so the results remain traceable and reproducible.
