
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "churn_threshold.txt"

model = joblib.load(MODEL_PATH)
threshold = float(THRESHOLD_PATH.read_text(encoding="utf-8"))

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn risk using a trained ML pipeline.",
    version="1.0.0",
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: CustomerData):
    data = pd.DataFrame([customer.model_dump()])

    probability = float(model.predict_proba(data)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "churn_prediction": prediction,
        "churn_label": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(probability, 4),
        "threshold": threshold,
    }
