# -*- coding: utf-8 -*-
"""
Entrenamiento de modelo (clasificación) leyendo configs/train.yaml.
Guarda el modelo en artifacts/model.pkl.
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from proyecto_mlops.config import load_config

# Mapeo nombre->clase sklearn
MODEL_REGISTRY = {
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
}

def build_model(model_name: str, params: dict):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo no soportado: {model_name}. Opciones: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[model_name]
    return cls(**(params or {}))

def train(cfg_name: str = "train.yaml"):
    cfg = load_config(cfg_name)
    paths = cfg.get("paths", {})
    target = cfg.get("target", "target")

    train_path = Path(paths["train"])
    model_out = Path(paths["model_out"])

    if not train_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrenamiento: {train_path}")

    # Carga datos
    df = pd.read_csv(train_path)
    if target not in df.columns:
        raise KeyError(f"Columna target '{target}' no está en el dataset. Columnas: {list(df.columns)}")

    X = df.drop(columns=[target])
    y = df[target]

    # Construye y entrena
    model = build_model(cfg.get("model", "RandomForestClassifier"), cfg.get("params", {}))
    model.fit(X, y)

    # Guarda artefacto
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f" Modelo entrenado y guardado en: {model_out}")

if __name__ == "__main__":
    train()
