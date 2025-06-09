
# 🌧️ Flood Risk Prediction API

This FastAPI application predicts **flood risk** for a specific district based on current and historical monthly rainfall data using a pre-trained machine learning model.

---

## 🚀 Features

- Accepts district name and 12 months of current rainfall data
- Looks up historical rainfall data from a dataset
- Predicts **flood risk level** using a trained classifier
- Returns a human-readable risk label to the client app (e.g., "Low", "Moderate", "High")

---

## 📦 Requirements

- Python 3.8+
- FastAPI
- scikit-learn
- pandas
- numpy
- joblib
- uvicorn

Install dependencies:

```bash
pip install fastapi uvicorn scikit-learn pandas numpy joblib
```

---

## 🧠 ML Model Structure

- `flood_risk_model.pkl`: Trained classifier model (e.g., RandomForest, DecisionTree)
- `label_encoder.pkl`: LabelEncoder used to encode/decode risk levels
- `augmented_flood_dataset.csv`: Dataset with districts and their average monthly rainfall (JAN_normal to DEC_normal)

---

## 📂 Project Structure

```
.
├── main.py                   # FastAPI app
├── flood_risk_model.pkl      # Trained ML model
├── label_encoder.pkl         # Label encoder for risk levels
├── augmented_flood_dataset.csv  # District-level historical rainfall data
└── README.md                 # Project documentation
```

---

## 🔁 Data Flow (Flutter App Integration)

```plaintext
User (in app)
   ⬇️
Get Location (lat/lng)
   ⬇️
Get District/City (via reverse geocoding)
   ⬇️
Fetch Rainfall Data (via API like IMD/OpenWeather)
   ⬇️
POST to FastAPI /predict
   ⬇️
Model predicts Flood Risk
   ⬇️
Return Risk Level to Flutter app
```

---

## 🛠️ API Endpoint

### `POST /predict`

#### Request Body:

```json
{
  "district": "Pune",
  "current_rainfall": [32.4, 28.1, 24.7, 12.0, 14.2, 90.3, 210.5, 190.7, 130.2, 45.1, 22.0, 19.8]
}
```

- `district`: Name of the district (case-insensitive)
- `current_rainfall`: List of 12 floats, each representing monthly rainfall (Jan–Dec)

#### Response:

```json
{
  "district": "Pune",
  "flood_risk": "High"
}
```

---

## ❗Error Handling

- `400 Bad Request`: If rainfall list is not 12 items long
- `404 Not Found`: If the district is not found in the dataset

---

## ▶️ Running the Server

```bash
uvicorn main:app --reload
```

Visit the interactive Swagger docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📌 Future Improvements

- Replace static CSV with live district-wise historical data API
- Add support for multi-language district names
- Integrate model versioning and feedback loop for model improvement

---

## 🧑‍💻 Maintainers

This API is part of a larger disaster management project for region-specific flood and cyclone prediction in India.

Feel free to reach out or contribute!
