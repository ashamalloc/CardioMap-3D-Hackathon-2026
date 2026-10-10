import os
from pathlib import Path
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL", "mysql+pymysql://cardiomap_user:change_me@localhost:3306/cardiomap")
MODEL_DIR = (BASE / os.getenv("MODEL_DIR", "../cad_models")).resolve()
DATASET_CSV = (BASE / os.getenv("DATASET_CSV", "../final_scaled_data.csv")).resolve()
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

# Target column in the CSV -> name shown in the frontend
TARGETS = {"Cath": "CAD", "LAD": "LAD", "LCX": "LCX", "RCA": "RCA"}
# Columns stored as natural log in the dataset; the API accepts raw lab units
LOG_FEATURES = {"FBS", "TG", "WBC", "PLT"}
