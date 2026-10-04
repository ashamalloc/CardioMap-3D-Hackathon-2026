# 3D anatomy note

The viewer does not infer anatomical coordinates from the tabular dataset. The dataset provides physiological/clinical features and vessel-level labels, not pixel-level or 3D coordinates.

For reproducibility and licensing simplicity, this prototype creates a stylized torso, heart, aorta and coronary curves procedurally with Three.js. Each vessel has a permanent semantic identifier:

- `LAD`
- `LCX`
- `RCA`

The application maps a model probability to vessel color and interaction state. The geometry is an explanatory visual surface only and must not be interpreted as the location, length or severity of an actual lesion.
