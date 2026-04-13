from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="ML API")

artifact = joblib.load("model.joblib")
model = artifact["model"]
target_names = artifact["target_names"]


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(data: IrisFeatures):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]
    label = target_names[prediction]

    return {
        "prediction_index": int(prediction),
        "prediction_label": label
    }