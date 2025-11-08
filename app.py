from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="API de Predicción de Consumo Energético - Equipo 49",
    description="Servicio de inferencia del modelo entrenado con DVC/MLflow",
    version="1.0.0"
)

MODEL_PATH = "models/best_model.pkl"
MODEL_VERSION = "v1.0"

# Cargar artefacto
artifacts = joblib.load(MODEL_PATH)
best_model_name = artifacts["best_model"]
models_dict = artifacts["models"]
model = models_dict[best_model_name]

class SteelEnergyInput(BaseModel):
    lagging_current_reactive_power_kvarh: float
    leading_current_reactive_power_kvarh: float
    co2_tco2: float
    lagging_current_power_factor: float
    leading_current_power_factor: float
    nsm: float
    mixed_type_col: float
    week_status: str
    day_of_week: str
    load_type: str

@app.get("/")
def home():
    return {
        "message": "API funcionando correctamente con corrección de codificación",
        "model_path": MODEL_PATH,
        "model_version": MODEL_VERSION,
        "best_model_name": best_model_name
    }

@app.post("/predict")
def predict_energy(data: SteelEnergyInput):
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([data.dict()])

        # --- Codificar manualmente las columnas categóricas ---
        df_encoded = pd.get_dummies(df, columns=["week_status", "day_of_week", "load_type"])

        # Alinear columnas con las que espera el modelo
        expected_features = model.feature_names_in_
        for col in expected_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # añadir columna faltante
        df_encoded = df_encoded[expected_features]  # mismo orden

        # Predicción
        prediction = model.predict(df_encoded)[0]

        return {
            "inputs": data.dict(),
            "prediction_usage_kwh": float(prediction),
            "model_version": MODEL_VERSION,
            "best_model_name": best_model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {e}")


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (for Render/Cloud platforms) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(app, host="0.0.0.0", port=port)
