from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "champion_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Breast Cancer Classification API",
    description="API for the trained classification model",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="The 30 numerical features in training order",
    )


@app.get("/")
def home():
    return {"message": "ML Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        columns = list(model.feature_names_in_)
        if len(columns) != len(request.features):
            raise HTTPException(
                status_code=422,
                detail="Incorrect number of features",
            )

        input_df = pd.DataFrame(
            [request.features],
            columns=columns,
        )

        prediction = int(model.predict(input_df)[0])

        response = {
            "prediction": prediction,
            "class_name": (
                "malignant" if prediction == 0 else "benign"
            ),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            response["probabilities"] = {
                str(int(label)): float(probability)
                for label, probability in zip(
                    model.classes_, probabilities
                )
            }

        return response

    except HTTPException:
        raise