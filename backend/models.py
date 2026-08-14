from pydantic import BaseModel, Field
from typing import Dict, Optional


class ComplaintRequest(BaseModel):
    narrative: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Consumer complaint narrative text to classify",
        examples=["I received a call from a debt collector about a debt I do not owe. They refused to provide validation."]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "narrative": "I received a call from a debt collector about a debt I do not owe. They refused to provide validation."
                }
            ]
        }
    }


class ModelPrediction(BaseModel):
    model_name: str = Field(description="Name of the ML model")
    prediction: str = Field(description="Predicted category: Credit card or Debt collection")
    confidence: float = Field(description="Prediction confidence score between 0 and 1")


class PredictionResponse(BaseModel):
    narrative: str = Field(description="Original complaint narrative")
    best_model: str = Field(description="Name of the best performing model")
    best_prediction: str = Field(description="Prediction from the best model")
    best_confidence: float = Field(description="Confidence of the best model")
    all_models: Dict[str, ModelPrediction] = Field(description="Predictions from all available models")


class ModelInfo(BaseModel):
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


class ModelsResponse(BaseModel):
    available_models: list[str]
    best_model: str
    metrics: Dict[str, ModelInfo]


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    tfidf_ready: bool
