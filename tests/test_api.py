from fastapi.testclient import TestClient
from main import app

def test_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        payload = res.json()
        assert payload["status"] == "healthy"
        assert payload["model_loaded"] is True

def test_predict_endpoint_valid_input():
    with TestClient(app) as client:
        res = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        assert res.status_code == 200
        payload = res.json()
        assert "prediction" in payload
        assert isinstance(payload["prediction"], int)
        assert len(payload["probabilities"]) == 3

def test_predict_endpoint_invalid_dimension():
    with TestClient(app) as client:
        res = client.post("/predict", json={"features": [1.0, 2.0]})
        assert res.status_code == 422
