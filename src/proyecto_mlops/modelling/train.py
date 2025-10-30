# -*- coding: utf-8 -*-
"""
Entrenamiento de modelo (clasificación) leyendo configs/train.yaml.
Guarda el modelo en artifacts/model.pkl.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from proyecto_mlops.config import load_config, CONFIGS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

MODEL_REGISTRY: Dict[str, Any] = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "knn": KNeighborsRegressor,
    "decision_tree": DecisionTreeRegressor,
    "svm": SVR,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}

def ensure_numeric_features(df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    missing = [col for col in numeric_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas numéricas en el dataset: {missing}")
    return df[numeric_cols].apply(pd.to_numeric, errors="coerce")

def normalise_model_type(raw_type: Optional[str]) -> str:
    if not raw_type:
        raise ValueError("Cada modelo necesita un campo 'type'.")
    key = raw_type.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Tipo de modelo '{raw_type}' no soportado. Tipos válidos: {sorted(MODEL_REGISTRY)}"
        )
    return key

def load_model_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "models" not in cfg or not cfg["models"]:
        raise ValueError("Debes definir al menos un modelo en la configuración (config.py) bajo el campo 'models'.")

    raw_models = cfg["models"]
    specs: List[Dict[str, Any]] = []

    if isinstance(raw_models, dict):
        items = raw_models.items()
    elif isinstance(raw_models, list):
        items = []
        for idx, entry in enumerate(raw_models):
            if not isinstance(entry, dict):
                raise ValueError("Cada elemento dentro de 'models' debe ser un mapeo/diccionario.")
            name = entry.get("name") or f"model_{idx}"
            spec = dict(entry)
            spec.pop("name", None)
            items.append((name, spec))
    else:
        raise ValueError("El campo 'models' debe ser un diccionario o una lista de diccionarios.")

    for name, data in items:
        model_type = normalise_model_type(data.get("type"))
        params = dict(data.get("params", {}))
        specs.append({"name": name, "type": model_type, "params": params})

    return specs

def build_pipeline(model_type: str, params: Dict[str, Any]) -> Pipeline:
    estimator_cls = MODEL_REGISTRY[model_type]
    # No imputation here; features.py must handle all missing values
    return Pipeline([("model", estimator_cls(**params))])

def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def run_train(df: pd.DataFrame, cfg_path: str = "configs/train.yaml", out_path: str = "artifacts/model.pkl"):
    cfg = load_config(cfg_path)
    # Si no hay modelos en el YAML, usa los de config.py legacy
    if "models" not in cfg or not cfg["models"]:
        print("[INFO] No se encontraron modelos en el YAML, usando modelos de config.py (CONFIGS['train'])")
        cfg["models"] = CONFIGS["train"]["models"]
    target = cfg.get("target", "usage_kwh").lower()
    # Asume que df ya está limpio y procesado por features.py
    X = df.drop(columns=[target])
    y = df[target]

    test_size = cfg.get("test_size", 0.12)
    val_size = cfg.get("val_size", 0.2)
    random_state = cfg.get("random_state", 42)
    val_random_state = cfg.get("val_random_state", 43)

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, shuffle=True, random_state=val_random_state
    )

    # Guardar splits a disco para uso en plots.py
    split_dir = Path("artifacts/eval/splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(val_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    cv_splits = cfg.get("cv_n_splits", 5)
    cv_repeats = cfg.get("cv_n_repeats", 3)
    cv_random_state = cfg.get("cv_random_state", 7)
    cv_strategy = RepeatedKFold(
        n_splits=cv_splits, n_repeats=cv_repeats, random_state=cv_random_state
    )
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    model_specs = load_model_specs(cfg)
    print("Resultados de Validación y Test:")
    best_model_name: Optional[str] = None
    best_avg_r2 = -np.inf
    avg_r2_dict = {}
    trained_models: Dict[str, Pipeline] = {}
    validation_results: List[Dict[str, Any]] = []
    test_metrics: Dict[str, Dict[str, float]] = {}
    train_metrics: Dict[str, Dict[str, float]] = {}
    for spec in model_specs:
        name = spec["name"]
        pipeline = build_pipeline(spec["type"], spec["params"])
        scores = cross_validate(
            pipeline, X_train, y_train, scoring=scoring, cv=cv_strategy, n_jobs=1
        )
        rmse_scores = -scores["test_rmse"]
        mae_scores = -scores["test_mae"]
        r2_scores = scores["test_r2"]
        mean_rmse = float(np.mean(rmse_scores))
        std_rmse = float(np.std(rmse_scores, ddof=1))
        mean_r2 = float(np.mean(r2_scores))
        std_r2 = float(np.std(r2_scores, ddof=1))
        fitted_pipeline = build_pipeline(spec["type"], spec["params"])
        fitted_pipeline.fit(X_train, y_train)
        trained_models[name] = fitted_pipeline
        # Validation metrics
        y_val_pred = fitted_pipeline.predict(X_val)
        val_rmse = rmse(y_val, y_val_pred)
        val_r2 = float(r2_score(y_val, y_val_pred))
        # Test metrics
        y_test_pred = fitted_pipeline.predict(X_test)
        test_rmse = rmse(y_test, y_test_pred)
        test_r2 = float(r2_score(y_test, y_test_pred))
        train_metrics[name] = {
            "rmse": rmse(y_train, fitted_pipeline.predict(X_train)),
            "mae": float(mean_absolute_error(y_train, fitted_pipeline.predict(X_train))),
            "r2": float(r2_score(y_train, fitted_pipeline.predict(X_train))),
        }
        test_metrics[name] = {
            "rmse": test_rmse,
            "mae": float(mean_absolute_error(y_test, y_test_pred)),
            "r2": test_r2,
        }
        validation_results.append(
            {
                "name": name,
                "cv_rmse_scores": rmse_scores.tolist(),
                "cv_mae_scores": mae_scores.tolist(),
                "cv_r2_scores": r2_scores.tolist(),
                "cv_rmse_mean": mean_rmse,
                "cv_rmse_std": std_rmse,
                "cv_r2_mean": mean_r2,
                "cv_r2_std": std_r2,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
            }
        )
        avg_r2 = (val_r2 + test_r2) / 2
        avg_r2_dict[name] = avg_r2
        print(f"{name}")
        print(f"  Val:   RMSE={val_rmse:.3f}, R2={val_r2:.3f}")
        print(f"  Test:  RMSE={test_rmse:.3f}, R2={test_r2:.3f}")
        print(f"  Promedio R2 (val+test)/2: {avg_r2:.3f}")
        print()
        if avg_r2 > best_avg_r2:
            best_avg_r2 = avg_r2
            best_model_name = name
    payload = {
        "best_model": best_model_name,
        "models": trained_models,
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "metrics": {name: {"train": train_metrics[name], "test": test_metrics[name]} for name in trained_models},
        "config": {
            "target": target,
            "test_size": test_size,
            "random_state": random_state,
            "cv_n_splits": cv_splits,
            "cv_n_repeats": cv_repeats,
            "cv_random_state": cv_random_state,
        },
        "avg_r2": avg_r2_dict,
        "split_paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(payload, out_path)
    # Guardar el mejor modelo como archivo individual
    best_model_obj = trained_models[best_model_name] if best_model_name in trained_models else None
    if best_model_obj is not None:
        best_model_path = Path("src/proyecto_mlops/models/best_model.pkl")
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model_obj, best_model_path)
        print(f"Mejor modelo exportado a: {best_model_path}")
    print(f"Mejor modelo seleccionado (según promedio R2 val+test): {best_model_name}")
    # Guardar métricas básicas en reports/metrics.json para DVC
    import json
    metrics_report = {
        "best_model": best_model_name,
        "train_metrics": train_metrics.get(best_model_name, {}),
        "test_metrics": test_metrics.get(best_model_name, {}),
        "validation_results": [vr for vr in validation_results if vr["name"] == best_model_name],
        "avg_r2": avg_r2_dict.get(best_model_name, None),
    }
    metrics_path = Path("reports/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)
    return payload

if __name__ == "__main__":
    cfg_path = "train.yaml"
    cfg = load_config(cfg_path)
    features_path = Path(cfg["paths"]["train"])
    if not features_path.exists():
        raise FileNotFoundError(f"No existe el archivo de features: {features_path}")
    df = pd.read_csv(features_path)
    run_train(df, cfg_path=cfg_path)


