$ErrorActionPreference = 'Stop'
Write-Host '=== CardioMap 3D model training ===' -ForegroundColor Cyan

if (-not (Test-Path '.venv')) { throw 'Virtual environment not found. Run .\scripts\setup.ps1 first.' }

& .\.venv\Scripts\python.exe -m ml_pipeline.train @args
if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }

Write-Host 'Model artifacts created in .\models' -ForegroundColor Green
