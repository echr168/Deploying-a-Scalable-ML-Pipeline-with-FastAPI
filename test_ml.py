import os

import numpy as np
import pandas as pd

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model
# TODO: add necessary import
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def _load_balanced_sample_df(n_rows: int = 200) -> pd.DataFrame:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, "data", "census.csv")
    df = pd.read_csv(data_path)

   # Verify both classes exist so training doesn't crash.
    df_pos = df[df[LABEL] == ">50K"]
    df_neg = df[df[LABEL] == "<=50K"]

    n_half = max(1, n_rows // 2)
    df_sample = pd.concat(
        [
            df_pos.sample(n=min(n_half, len(df_pos)), random_state=42),
            df_neg.sample(n=min(n_rows - n_half, len(df_neg)), random_state=42),
        ],
        axis=0,
    ).sample(frac=1.0, random_state=42).reset_index(drop=True)

    return df_sample

# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_fitted_estimator():
    df = _load_balanced_sample_df(200)
    X, y, _, _ = process_data(df, categorical_features=CAT_FEATURES, label=LABEL, training=True)
    model = train_model(X, y)
    assert hasattr(model, "predict")


# TODO: implement the second test. Change the function name and input as needed
def test_inference_outputs_expected_shape():
    df = _load_balanced_sample_df(220)
    X, y, _, _ = process_data(df, categorical_features=CAT_FEATURES, label=LABEL, training=True)

    X_train, y_train = X[:160], y[:160]
    X_test = X[160:200]

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_in_range():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    for metric in (precision, recall, fbeta):
        assert 0.0 <= metric <= 1.0
