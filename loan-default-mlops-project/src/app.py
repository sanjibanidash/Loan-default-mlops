from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.pipeline.prediction_pipeline import (
    PredictionPipeline
)


# ==================================================
# CREATE FASTAPI APP
# ==================================================

app = FastAPI()


# ==================================================
# CREATE PIPELINE OBJECT ONCE
# ==================================================

pipeline = PredictionPipeline()


# ==================================================
# PYDANTIC VALIDATION SCHEMA
# ==================================================

class LoanData(BaseModel):

    AMT_INCOME_TOTAL: float = Field(
        gt=0,
        description="Income must be positive"
    )

    AMT_CREDIT: float = Field(
        gt=0,
        description="Credit amount must be positive"
    )

    AMT_ANNUITY: float = Field(
        gt=0,
        description="Annuity amount must be positive"
    )

    DAYS_BIRTH: int = Field(
        lt=0,
        description="Birth days should be negative"
    )

    DAYS_EMPLOYED: int = Field(
        lt=0,
        description="Employment days should be negative"
    )


# ==================================================
# HOME ROUTE
# ==================================================

@app.get("/")
def home():

    return {

        "message": "Loan Default Prediction API Running"

    }


# ==================================================
# HEALTH CHECK ROUTE
# ==================================================

@app.get("/health")
def health_check():

    return {

        "status": "API Running Successfully"

    }


# ==================================================
# PREDICTION ROUTE
# ==================================================

@app.post("/predict")
def predict(data: LoanData):

    try:

        # Convert input to dictionary
        input_data = data.dict()

        # Get prediction
        prediction, probability = pipeline.predict(
            input_data
        )

        # Return response
        return {

            "prediction": prediction,

            "default_probability": round(
                float(probability),
                4
            )

        }

    except Exception as e:

        return {

            "error": str(e)

        }