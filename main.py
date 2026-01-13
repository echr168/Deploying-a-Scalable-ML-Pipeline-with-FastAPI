import os
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.data import process_data, apply_label
from ml.model import inference, load_model

app = FastAPI()

# Paths (from repo root)
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_PATH, "model", "model.pkl")
ENCODER_PATH = os.path.join(PROJECT_PATH, "model", "encoder.pkl")
LB_PATH = os.path.join(PROJECT_PATH, "model", "lb.pkl")

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class CensusRow(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        populate_by_name = True


def _ensure_artifacts_exist() -> None:
    """Raise a clear error if model artifacts haven't been created yet."""
    missing = [p for p in (MODEL_PATH, ENCODER_PATH, LB_PATH) if not os.path.exists(p)]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing model artifacts. Run `python train_model.py` to create them. "
                f"Missing: {missing}"
            ),
        )


@app.get("/")
def read_root() -> Dict[str, Any]:
    return {"message": "Hello from the API!"}


@app.post("/data/")
def predict(data: CensusRow) -> Dict[str, Any]:
    _ensure_artifacts_exist()

    model = load_model(MODEL_PATH)
    encoder = load_model(ENCODER_PATH)
    lb = load_model(LB_PATH)

    df = pd.DataFrame([data.model_dump(by_alias=True)])

    X, _, _, _ = process_data(
        df,
        categorical_features=CATEGORICAL_FEATURES,
        label=None,  # no label provided for inference
        training=False,
        encoder=encoder,
        lb=lb,
    )

    pred = inference(model, X)
    return {"prediction": apply_label(pred)}
