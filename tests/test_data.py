import pandas as pd
from ml_pipeline.data import prepare_dataset


def test_leakage_columns_removed():
    df = pd.DataFrame({
        'Age': [50, 60, 70, 55],
        'BP': [120, 135, 150, 128],
        'LAD': [0, 1, 0, 1],
        'LCX': [0, 0, 1, 0],
        'RCA': [0, 1, 1, 0],
        'Cath': [0, 1, 1, 1],
        'CAD': ['Normal', 'CAD', 'CAD', 'CAD'],
    })
    X, ys, cols = prepare_dataset(df)
    assert set(['LAD','LCX','RCA','Cath']).isdisjoint(set(X.columns))
    assert 'Age' in X.columns
    assert ys['CAD'].tolist() == [0, 1, 1, 1]
