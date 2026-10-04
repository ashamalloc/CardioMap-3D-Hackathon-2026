$ErrorActionPreference = 'Stop'
Write-Host '=== Starting CardioMap 3D ===' -ForegroundColor Cyan

if (-not (Test-Path 'models\cad_model.joblib')) {
  Write-Host 'Warning: model artifacts not found. The UI will open, but prediction will remain disabled.' -ForegroundColor Yellow
}

Start-Process powershell -ArgumentList '-NoExit','-Command',"& '$PWD\.venv\Scripts\python.exe' -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList '-NoExit','-Command',"Set-Location '$PWD\frontend'; npm run dev -- --host 0.0.0.0"

Start-Sleep -Seconds 2
Start-Process 'http://localhost:5173'
