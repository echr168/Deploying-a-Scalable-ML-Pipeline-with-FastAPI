from __future__ import annotations

import os
import pickle
from typing import Any, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data

# TODO: add necessary import


# Optional: implement hyperparameter tuning.
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator:
    """Train a fast linear classifier and return the fitted model."""
    model = SGDClassifier(
        loss="log_loss",
        max_iter=2000,
        tol=1e-3,
        random_state=42,
    )
    # TODO: implement the function
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(
    y: np.ndarray, preds: np.ndarray
) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # TODO: implement the function
    return model.predict(X)


def save_model(obj: Any, path: str) -> None:
    """Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # TODO: implement the function
    os_dir = os.path.dirname(path)
    if os_dir:
        os.makedirs(os_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_model(path: str) -> Any:
    """Loads pickle file from `path` and returns it."""
    # TODO: implement the function
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
) -> Tuple[float, float, float]:
    """Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # TODO: implement the function
    data_slice = data[data[column_name] == slice_value].copy()

    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(
        model, X_slice
    )  # your code here to get prediction on X_slice using the inference function
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
