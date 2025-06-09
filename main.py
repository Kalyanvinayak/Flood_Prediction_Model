from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta
from difflib import get_close_matches
import logging

app = FastAPI()

# ---------- Setup Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flood-predictor")

# ---------- Load ML Assets ----------
model = joblib.load("flood_risk_model.pkl")
encoder = joblib.load("label_encoder.pkl")
rainfall_df = pd.read_csv("augmented_flood_dataset.csv")

# Clean column names to avoid whitespace issues
rainfall_df.columns = rainfall_df.columns.str.strip()

all_districts = rainfall_df['District'].str.upper().unique()

# Define the 12 normal rainfall columns explicitly
normal_cols = [
    "JAN_normal", "FEB_normal", "MAR_normal", "APR_normal", "MAY_normal", "JUN_normal",
    "JUL_normal", "AUG_normal", "SEP_normal", "OCT_normal", "NOV_normal", "DEC_normal"
]

# ---------- Input Schema ----------
class FloodRequest(BaseModel):
    lat: float
    lon: float

# ---------- Reverse Geocode ----------
def reverse_geocode(lat: float, lon: float) -> str:
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
    headers = {
        "User-Agent": "FloodPredictorApp/1.0 (kalyanleomessi@gmail.com)"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Reverse geocoding failed: {response.text}")
        raise HTTPException(status_code=500, detail="Reverse geocoding failed")
    data = response.json()
    address = data.get("address", {})
    
    # Log all the fields of reverse geocoding
    logger.info("📍 Reverse Geocoding Details:")
    logger.info(f"  - County: {address.get('county')}")
    logger.info(f"  - State District: {address.get('state_district')}")
    logger.info(f"  - City District: {address.get('city_district')}")
    logger.info(f"  - District: {address.get('district')}")
    logger.info(f"  - City: {address.get('city')}")
    logger.info(f"  - State: {address.get('state')}")
    
    district = (
        address.get("state_district") 
        or address.get("county")  
        or address.get("city_district") 
        or address.get("district") 
        or address.get("city") 
        or address.get("state")
    )
    if not district:
        raise HTTPException(status_code=500, detail="Unable to determine district")
    logger.info(f"🗺 Reverse geocoded district: {district}")
    return district.upper()

# ---------- Fuzzy Match ----------
def match_district_name(district: str, all_districts) -> str | None:
    matches = get_close_matches(district.upper(), all_districts, n=3, cutoff=0.6)
    logger.info(f"🔍 Fuzzy matches for '{district}': {matches}")
    return matches[0] if matches else None

# ---------- Fetch Past Rainfall ----------
def get_past_rainfall(lat: float, lon: float) -> list:
    today = datetime.today()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=precipitation_sum&timezone=auto"
    )
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Rainfall API failed: {response.text}")
        raise HTTPException(status_code=500, detail="Rainfall API failed")

    data = response.json()
    dates = data.get("daily", {}).get("time", [])
    rainfall = data.get("daily", {}).get("precipitation_sum", [])

    if not dates or not rainfall or len(dates) != len(rainfall):
        raise HTTPException(status_code=500, detail="Rainfall data is incomplete")

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "rainfall": rainfall
    })
    df["month"] = df["date"].dt.to_period("M")
    monthly_totals = df.groupby("month")["rainfall"].sum().tolist()

    if len(monthly_totals) < 12:
        raise HTTPException(status_code=400, detail="Insufficient rainfall data")

    return monthly_totals[-12:]

# ---------- Prediction Endpoint ----------
@app.post("/predict")
def predict_flood(req: FloodRequest):
    original_district = reverse_geocode(req.lat, req.lon)
    matched_district = match_district_name(original_district, all_districts)

    if not matched_district:
        raise HTTPException(status_code=404, detail=f"District '{original_district}' not in training data")

    logger.info(f"✅ Matched district: {matched_district}")

    rainfall_12_months = get_past_rainfall(req.lat, req.lon)
    if len(rainfall_12_months) != 12:
        raise HTTPException(status_code=400, detail="Insufficient rainfall data")

    # Fetch the district row
    row = rainfall_df[rainfall_df['District'].str.upper() == matched_district]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No rainfall data for district {matched_district}")

    try:
        past_rainfall = row.iloc[0][normal_cols].astype(float).values
    except Exception as e:
        logger.error(f"Error converting past rainfall data: {e}")
        raise HTTPException(status_code=500, detail="Invalid past rainfall data format")

    # Combine past normal rainfall + current monthly rainfall from API as input features
    input_features = list(past_rainfall) + list(map(float, rainfall_12_months))
    input_df = pd.DataFrame([input_features])

    try:
        prediction = model.predict(input_df)[0]
        label = encoder.inverse_transform([prediction])[0]
    except Exception as e:
        logger.error(f"Model prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    logger.info(f"📊 Predicted risk for {matched_district}: {label}")

    return {
        "original_district": original_district,
        "matched_district": matched_district,
        "flood_risk": label
    }
