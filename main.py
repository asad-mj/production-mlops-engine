from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.schemas import PredictionInput, PredictionOutput, HealthResponse
from src.model import ModelService

model_service: ModelService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service
    model_service = ModelService()
    yield

app = FastAPI(
    title="Production MLOps Prediction Engine",
    description="Containerized microservice with input validation, automated inference, and health probes.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    is_loaded = model_service is not None and model_service.model is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "version": model_service.version if model_service else "unknown"
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(payload: PredictionInput):
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model runtime uninitialized.")
    pred, probs = model_service.predict(payload.features)
    return {
        "prediction": pred,
        "probabilities": probs,
        "model_version": model_service.version
    }
