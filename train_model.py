import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
)

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "census.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl")

SLICE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "slice_output.txt")

# Project constants
LABEL = "salary"
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


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[LABEL],
    )

    # Process training data
    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    # Process test data using fitted encoder/lb
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train model
    model = train_model(X_train, y_train)

    # Save artifacts
    save_model(model, MODEL_PATH)
    save_model(encoder, ENCODER_PATH)
    save_model(lb, LB_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Encoder saved to {ENCODER_PATH}")
    print(f"Label binarizer saved to {LB_PATH}")

    # Evaluate on full test set
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # Slice performance
    with open(SLICE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for feature in CAT_FEATURES:
            values = sorted(test_df[feature].dropna().unique())
            for value in values:
                slice_df = test_df[test_df[feature] == value]
                if slice_df.shape[0] == 0:
                    continue

                X_slice, y_slice, _, _ = process_data(
                    slice_df,
                    categorical_features=CAT_FEATURES,
                    label=LABEL,
                    training=False,
                    encoder=encoder,
                    lb=lb,
                )

                slice_preds = inference(model, X_slice)
                p, r, fb = compute_model_metrics(y_slice, slice_preds)

                f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n")
                f.write(f"{feature}: {value}, Count: {slice_df.shape[0]}\n")

    print(f"Slice output saved to {SLICE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
