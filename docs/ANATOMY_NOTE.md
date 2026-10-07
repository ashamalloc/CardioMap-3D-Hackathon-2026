# 3D Anatomy Note

## Anatomical 3D Meshes & Segmentation
The system has been upgraded from procedural geometric primitives to authentic, high-resolution anatomical 3D surface meshes (`.glb` / glTF 2.0 format) stored in `frontend/public/models/`:

1. **`heart.glb`**: High-fidelity myocardium (cardiac muscle tissue) mesh containing ~246,000 vertices and 362,000 polygons, rendered with anatomical PBR shading (`MeshPhysicalMaterial`) featuring user-controlled translucency (Translucent, X-Ray, Solid, and Arteries-Only modes).
2. **`LAD.glb`**: Left Anterior Descending coronary artery tree (~12,500 polygons) mapped directly along the anterior interventricular sulcus.
3. **`LCX.glb`**: Left Circumflex coronary artery branch (~19,300 polygons) traveling around the left atrioventricular groove.
4. **`RCA.glb`**: Right Coronary Artery branch (~13,800 polygons) traveling down the right coronary sulcus towards the inferior heart surface.

## Coordinate Registration & Alignment
All 4 models were exported in the same unified anatomical scanner coordinate frame (centered at `[22.89, -123.92, 1233.93] mm`). The 3D viewer pipeline:
- Applies an automated center-of-mass translation to position the heart at the world origin `(0, 0, 0)`.
- Reorients the anatomical coordinate frame (`rotation.x = -Math.PI / 2`) so that **Superior/Cranial (Z)** aligns with Three.js **Up (+Y)**, and **Anterior (-Y)** faces the **Front (+Z)** directly toward the clinician.
- Computes vertex normals dynamically on load (`computeVertexNormals()`) for smooth PBR lighting.
- Retains a stylized thoracic context cage around the heart to maintain anatomical spatial grounding within the human chest cavity.

## Model-to-Anatomy Semantic Mapping
Each coronary vessel mesh has a permanent semantic identifier:
- **`LAD`**: Left Anterior Descending
- **`LCX`**: Left Circumflex
- **`RCA`**: Right Coronary Artery

The ML model's predicted stenosis risk probabilities dynamically color-code the individual vessel branches:
- **Low Risk (<33%)**: Emerald Green / Teal
- **Intermediate Risk (33%–66%)**: Amber / Gold
- **High Risk (>66%)**: Crimson Red / Warning Glow
- **Active Focus**: Pulsing emissive illumination and dedicated camera angle presets (Anterior, Lateral, Inferior, Overview).

## Clinical Interpretation Boundary
The tabular clinical dataset (UCI Extension of Z-Alizadeh Sani) contains patient physiological measurements, laboratory findings, and vessel-level stenosis classifications (>50% narrowing), but does *not* contain direct pixel-level or 3D lesion coordinates. The 3D visual mapping represents vessel-level disease probability on the anatomical arterial tree and serves as an interactive clinical decision-support surface, not as an exact patient-specific angiographic reconstruction.
