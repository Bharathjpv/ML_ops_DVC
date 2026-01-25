from random import sample
from turtle import title
from fastapi import FastAPI
import pickle
import pandas as pd
import uvicorn
from data_model import WaterFeatures

app = FastAPI(
    title="Water Potability Prediction API",
    description="An API to predict the potability of water based on its chemical properties.",
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def index():
    return {"message": "Welcome to the Water Potability Prediction API!"}

@app.post("/predict")
def predict(features: WaterFeatures):
    sample = pd.DataFrame(
        {'ph' : [features.ph],
        'Hardness': [features.Hardness],
        'Solids': [features.Solids],
        'Chloramines': [features.Chloramines],
        'Sulfate': [features.Sulfate],
        'Conductivity': [features.Conductivity],
        'Organic_carbon': [features.Organic_carbon],
        'Trihalomethanes': [features.Trihalomethanes],
        'Turbidity': [features.Turbidity]}
    )
    prediction = model.predict(sample)
    if prediction == 1:
        return "The water is potable."
    else:
        return "The water is not potable."


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)