import json
import math

import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.app import main
from ml_pipeline import modeling
from ml_pipeline.modeling import FitResult, make_pipeline


def test_project_lcx_adapter_matches_model_input_contract():
    assert "LCX" in main.MODELS

    cardio_map_input = main._feature_frame({
        "Sex": "Male",
        "BBB": "RBBB",
        "VHD": "Moderate",
        "FBS": 100,
        "TG": 200,
        "WBC": 8000,
        "PLT": 250,
    })
    project_input = main._project_lcx_feature_frame(cardio_map_input)

    assert list(project_input.columns) == main.MODELS["LCX"].get_booster().feature_names
    assert project_input.loc[0, "Sex"] == 1
    assert project_input.loc[0, "BBB"] == 2
    assert project_input.loc[0, "VHD"] == 2
    assert project_input.loc[0, "FBS"] == math.log(100)
    assert project_input.loc[0, "Thyroid Disease"] == 0
    assert np.isfinite(project_input.to_numpy()).all()


def test_prediction_uses_project_lcx_and_retains_all_targets():
    result = main.predict(main.PredictRequest(features={
        "Age": 61,
        "Sex": "Fmale",
        "BBB": "LBBB",
        "VHD": "Severe",
        "FBS": 100,
        "TG": 200,
        "WBC": 8000,
        "PLT": 250,
    }))

    assert set(result["vessels"]) == {"LAD", "LCX", "RCA"}
    assert 0 <= result["vessels"]["LCX"]["probability"] <= 1
    assert result["model_version"] == "CardioMap RF v1 + proj XGBoost LCX"
    assert result["vessels"]["LCX"]["explanations"]


def test_prediction_rejects_invalid_features_instead_of_using_defaults():
    invalid_inputs = [
        {"FBS": "not a number"},
        {"FBS": float("nan")},
        {"Sex": "unknown"},
        {"Unknown feature": 1},
    ]

    for features in invalid_inputs:
        try:
            main.predict(main.PredictRequest(features=features))
        except HTTPException as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError(f"Expected invalid features to fail: {features}")


def test_training_preserves_imported_lcx_model_and_metrics(tmp_path, monkeypatch):
    (tmp_path / "lcx_xgb.json").write_text("{}", encoding="utf-8")
    imported_metrics = {"model": "XGBClassifier", "holdout": {"f1": 0.583}}
    (tmp_path / "metrics.json").write_text(
        json.dumps({"LCX": imported_metrics}),
        encoding="utf-8",
    )
    fitted_targets = []

    def fake_fit_one_target(*args):
        target = args[2]
        fitted_targets.append(target)
        return FitResult(
            pipeline=make_pipeline(pd.DataFrame({"Age": [1]}))[0],
            metrics={"model": "RandomForestClassifier"},
            schema={"fields": [], "count": 0},
            explain_map={},
        )

    monkeypatch.setattr(modeling, "fit_one_target", fake_fit_one_target)
    targets = ("CAD", "LAD", "LCX", "RCA")
    labels = {target: pd.Series(dtype=int) for target in targets}
    results = modeling.train_all(pd.DataFrame(), labels, tmp_path)
    saved_metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert fitted_targets == ["CAD", "LAD", "RCA"]
    assert set(results) == {"CAD", "LAD", "RCA"}
    assert saved_metrics["LCX"] == imported_metrics
