import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# TODO: load the cencus.csv data
def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, "data", "census.csv")
    model_dir = os.path.join(project_root, "model")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[LABEL],
    )

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train_df, categorical_features=CAT_FEATURES, label=LABEL, training=True
    # your code here
    # use the train dataset 
    # use training=True
    # do not need to pass encoder and lb as input
    )

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)# your code here

# save the model and the encoder
    model_path = os.path.join(model_dir, "model.pkl")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    lb_path = os.path.join(model_dir, "lb.pkl")

    save_model(model, model_path)
    save_model(encoder, encoder_path)
    save_model(lb, lb_path)

    print(f"Model saved to {model_path}")
    print(f"Encoder saved to {encoder_path}")
    print(f"Label binarizer saved to {lb_path}") 

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model_loaded, X_test) # your code here

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
    slice_output_path = os.path.join(project_root, "slice_output.txt")
    with open(slice_output_path, "w", encoding="utf-8") as f:
        for feature in CAT_FEATURES:
            for value in sorted(test_df[feature].dropna().unique()):
                precision, recall, fbeta = performance_on_categorical_slice(
                    data=test_df,
                    column_name=feature,
                    slice_value=value,
                    categorical_features=CAT_FEATURES,
                    label=LABEL,
                    encoder=encoder,
                    lb=lb,
                    model=model_loaded,
                )
                count = int((test_df[feature] == value).sum())
                f.write(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}\n")
                f.write(f"{feature}: {value}, Count: {count}\n")

    print(f"Slice output saved to {slice_output_path}")


if __name__ == "__main__":
    main()