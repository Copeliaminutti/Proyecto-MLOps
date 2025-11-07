from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features import build
from src.config import load_config

app = FastAPI(
    title="Energy Consumption Prediction API",
    description="API para predecir el consumo de energía (usage_kwh) basado en features de entrada",
    version="1.0.0"
)

# Cargar el modelo al iniciar la aplicación
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "best_model.pkl"
model = None
best_model_name = None
expected_features = None

@app.on_event("startup")
async def load_model():
    """Carga el modelo al iniciar la aplicación"""
    global model, best_model_name, expected_features
    try:
        # Cargar el payload completo
        payload = joblib.load(MODEL_PATH)
        
        # Extraer el nombre del mejor modelo
        best_model_name = payload["best_model"]
        
        # Extraer el modelo entrenado del diccionario
        model = payload["models"][best_model_name]
        
        # Obtener las features esperadas por el modelo
        expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        
        print(f"✓ Modelo cargado exitosamente desde {MODEL_PATH}")
        print(f"✓ Mejor modelo: {best_model_name}")
        if expected_features is not None:
            print(f"✓ Features esperadas: {len(expected_features)}")
    except Exception as e:
        print(f"✗ Error al cargar el modelo: {e}")
        raise


class PredictionInput(BaseModel):
    """Schema para los datos de entrada de predicción"""
    lagging_current_reactive_power_kvarh: float = Field(..., description="Potencia reactiva de corriente retrasada (kVArh)")
    leading_current_reactive_power_kvarh: float = Field(..., description="Potencia reactiva de corriente adelantada (kVArh)")
    co2_tco2: float = Field(..., description="Emisiones de CO2 (tCO2)")
    lagging_current_power_factor: float = Field(..., description="Factor de potencia de corriente retrasada")
    leading_current_power_factor: float = Field(..., description="Factor de potencia de corriente adelantada")
    nsm: float = Field(..., description="NSM (Number of Seconds from Midnight)")
    mixed_type_col: float = Field(..., description="Columna de tipo mixto")
    week_status: str = Field(..., description="Estado de la semana (e.g., 'Weekday', 'Weekend')")
    day_of_week: str = Field(..., description="Día de la semana (e.g., 'Monday', 'Tuesday')")
    load_type: str = Field(..., description="Tipo de carga (e.g., 'Light_Load', 'Medium_Load', 'Maximum_Load')")

    class Config:
        schema_extra = {
            "example": {
                "lagging_current_reactive_power_kvarh": 15.5,
                "leading_current_reactive_power_kvarh": 8.2,
                "co2_tco2": 0.05,
                "lagging_current_power_factor": 25.3,
                "leading_current_power_factor": 18.7,
                "nsm": 43200.0,
                "mixed_type_col": 1.0,
                "week_status": "Weekday",
                "day_of_week": "Monday",
                "load_type": "Medium_Load"
            }
        }


class PredictionOutput(BaseModel):
    """Schema para la respuesta de predicción"""
    prediction: float = Field(..., description="Predicción de consumo de energía (usage_kwh)")
    input_features: dict = Field(..., description="Features de entrada procesados")


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Energy Consumption Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Verifica el estado de salud de la API"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "best_model": best_model_name
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Realiza una predicción de consumo de energía basado en los features de entrada.
    
    Los features categóricos (week_status, day_of_week, load_type) serán procesados
    automáticamente mediante one-hot encoding igual que en el pipeline de entrenamiento.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Convertir input a DataFrame
        input_dict = input_data.dict()
        df_input = pd.DataFrame([input_dict])
        
        # Cargar configuración y procesar features usando el mismo pipeline
        cfg = load_config("features.yaml")
        df_processed = build(df_input, cfg)
        
        # Asegurar que todas las columnas esperadas estén presentes
        if expected_features is not None:
            # Crear DataFrame con todas las columnas esperadas, inicializadas en 0
            df_aligned = pd.DataFrame(0, index=df_processed.index, columns=expected_features)
            
            # Copiar los valores de las columnas que existen en df_processed
            for col in df_processed.columns:
                if col in df_aligned.columns:
                    df_aligned[col] = df_processed[col].values
            
            df_processed = df_aligned
        
        # Hacer predicción
        prediction = model.predict(df_processed)[0]
        
        # Preparar respuesta con features procesados (solo los no cero para legibilidad)
        processed_features = {k: v for k, v in df_processed.iloc[0].to_dict().items() if v != 0}
        
        return PredictionOutput(
            prediction=float(prediction),
            input_features=processed_features
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

