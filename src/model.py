import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
MODEL_VERSION = "1.0.0"

def train_and_persist_model() -> str:
    """Trains a baseline model on load and serializes weights to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    data = load_iris()
    X_train, _, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump({"model": clf, "version": MODEL_VERSION}, MODEL_PATH)
    return MODEL_PATH

class ModelService:
    def __init__(self):
        self.model = None
        self.version = MODEL_VERSION
        self.load_model()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            train_and_persist_model()
        artifact = joblib.load(MODEL_PATH)
        self.model = artifact["model"]
        self.version = artifact.get("version", MODEL_VERSION)

    def predict(self, features: list[float]):
        if self.model is None:
            raise RuntimeError("Model artifact failed to load into memory.")
        data_arr = np.array([features])
        pred = int(self.model.predict(data_arr)[0])
        probs = self.model.predict_proba(data_arr)[0].tolist()
        return pred, probs
