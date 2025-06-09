from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

# Load the model and label encoder
clf = joblib.load("flood_risk_model.pkl")
le = joblib.load("label_encoder.pkl")

# Load the augmented rainfall dataset
df = pd.read_csv("augmented_flood_dataset.csv")

app = FastAPI()

class FloodRequest(BaseModel):
    district: str
    current_rainfall: list[float]  # Must be 12 months

@app.post("/predict")
def predict_flood(request: FloodRequest):
    if len(request.current_rainfall) != 12:
        raise HTTPException(status_code=400, detail="Current rainfall must have 12 monthly values.")

    row = df[df['District'].str.lower() == request.district.lower()]
    if row.empty:
        raise HTTPException(status_code=404, detail="District not found in dataset.")

    # Correct past rainfall columns as per your dataset
    rainfall_columns = [
        "JAN_normal", "FEB_normal", "MAR_normal", "APR_normal", "MAY_normal", "JUN_normal",
        "JUL_normal", "AUG_normal", "SEP_normal", "OCT_normal", "NOV_normal", "DEC_normal"
    ]

    past_rainfall = row[rainfall_columns].values.flatten().astype(float)

    # Combine past rainfall and current rainfall features
    input_data = np.concatenate([past_rainfall, request.current_rainfall]).reshape(1, -1)

    # Predict flood risk
    pred = clf.predict(input_data)
    label = le.inverse_transform(pred)[0]

    return {
        "district": request.district.title(),
        "flood_risk": label,
    }
