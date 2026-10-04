# Architecture

```text
                   ┌──────────────────────────┐
                   │   React clinical UI      │
                   │ + Three.js visualization │
                   └────────────┬─────────────┘
                                │ JSON
                                ▼
                   ┌──────────────────────────┐
                   │       FastAPI API         │
                   │ schema / predict / stats  │
                   └────────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   CAD Random Forest      LAD Random Forest     LCX/RCA forests
          │                     │                     │
          └────────────── SHAP explanations ──────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │ Leakage-safe preprocessing│
                   │ numeric + categorical    │
                   └────────────┬─────────────┘
                                │
                                ▼
                   UCI Extension of Z-Alizadeh
                         Sani dataset
```
