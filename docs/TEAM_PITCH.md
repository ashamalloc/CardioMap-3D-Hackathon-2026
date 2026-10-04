# 60-second pitch

CardioMap 3D makes cardiovascular risk easier to understand without pretending that a tabular model is an angiogram.

We train four leakage-safe classifiers on the UCI Extension of Z-Alizadeh Sani dataset: one for overall CAD and one each for LAD, LCX and RCA stenosis. We explicitly remove LAD, LCX, RCA and Cath from the inputs, so the model cannot read the answer from angiographic labels.

The prediction service is exposed through FastAPI. A React + Three.js interface turns the three vessel probabilities into a responsive anatomical visualization that users can rotate, zoom and inspect. SHAP converts model behavior into clinical feature contributions.

The key design decision is honesty: the 3D layer is a visualization abstraction, not a hidden anatomical reconstruction. The system clearly labels itself as an educational / decision-support prototype and never claims to replace diagnostic imaging.
