import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")
    class Config:
        populate_by_name = True


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

ENCODER_PATH = os.path.join(PROJECT_ROOT, "model", "encoder.pkl") # TODO: enter the path for the saved encoder 
encoder = load_model(ENCODER_PATH)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model.pkl") # TODO: enter the path for the saved model 
model = load_model(MODEL_PATH)

# TODO: create a RESTful API using FastAPI
app = FastAPI() # your code here

# TODO: create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    return {"message": "Hello from the API!"}# your code here


# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict(by_alias=True)
    df = pd.DataFrame.from_dict({k: [v] for k, v in data_dict.items()})

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
   
    X, _, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=encoder,
    )

    pred = inference(model, X)
    return {"result": apply_label(pred)
