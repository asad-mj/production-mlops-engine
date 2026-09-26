from typing import List
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    features: List[float] = Field(
        ..., 
        description="Array of 4 numerical features (e.g. sepal/petal dimensions)",
        min_length=4,
        max_length=4
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"features": [5.1, 3.5, 1.4, 0.2]}
            ]
        }
    }

class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="Target class index")
    probabilities: List[float] = Field(..., description="Normalized class distribution")
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
