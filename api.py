"""
Predictive Maintenance — FastAPI REST API (Deployment Opsi A)
ABK4ABB3 Pembelajaran Mesin dan Aplikasi

Run:   uvicorn api:app --host 0.0.0.0 --port 8000
Docs:  http://localhost:8000/docs
Endpoint /predict accepts POST JSON, returns prediction in < 3s.
"""
import json, joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Predictive Maintenance API", version="1.0")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
with open("metrics.json") as f:
    META = json.load(f)
FEATURES = META["feature_names"]
TYPE_MAP = META["type_mapping"]

class MachineInput(BaseModel):
    Type: str = Field("L", description="Product quality: L, M, or H")
    air_temperature: float = Field(300.0, alias="Air temperature [K]")
    process_temperature: float = Field(310.0, alias="Process temperature [K]")
    rotational_speed: float = Field(1500, alias="Rotational speed [rpm]")
    torque: float = Field(40.0, alias="Torque [Nm]")
    tool_wear: float = Field(100, alias="Tool wear [min]")

    class Config:
        populate_by_name = True

@app.get("/")
def root():
    return {"service": "Predictive Maintenance API", "model": META["winner"],
            "docs": "/docs", "predict": "POST /predict"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(inp: MachineInput):
    temp_diff = inp.process_temperature - inp.air_temperature
    power = inp.torque * inp.rotational_speed * 2 * np.pi / 60.0
    wear_torque = inp.tool_wear * inp.torque
    row = {
        "Type": TYPE_MAP.get(inp.Type, 0),
        "Air temperature [K]": inp.air_temperature,
        "Process temperature [K]": inp.process_temperature,
        "Rotational speed [rpm]": inp.rotational_speed,
        "Torque [Nm]": inp.torque,
        "Tool wear [min]": inp.tool_wear,
        "Temp diff [K]": temp_diff,
        "Power [W]": power,
        "Wear*Torque": wear_torque,
    }
    X = np.array([[row[f] for f in FEATURES]])
    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    proba = float(model.predict_proba(Xs)[0][1])
    return {
        "machine_failure": pred,
        "failure_probability": round(proba, 4),
        "label": "Failure" if pred == 1 else "OK",
    }
