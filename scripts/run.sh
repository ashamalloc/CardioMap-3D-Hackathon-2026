#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
cd frontend
npm run dev -- --host 0.0.0.0
