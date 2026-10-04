$ErrorActionPreference = 'Stop'
Write-Host '=== CardioMap 3D setup ===' -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python was not found on PATH.' }
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Node.js was not found on PATH.' }

if (-not (Test-Path '.venv')) {
  python -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

Push-Location frontend
npm install
Pop-Location

Write-Host 'Setup complete.' -ForegroundColor Green
Write-Host 'Next: .\scripts\train.ps1' -ForegroundColor Yellow
