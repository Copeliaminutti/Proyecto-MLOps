# -*- coding: utf-8 -*-
"""
Entrenamiento de modelo (clasificación) leyendo configs/train.yaml.
Guarda el modelo en artifacts/model.pkl.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
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


class ModelSpec:
    """Representa una especificación de modelo con su tipo y parámetros."""
    
    def __init__(self, name: str, model_type: str, params: Dict[str, Any]):
        self.name = name
        self.model_type = model_type
        self.params = params
    
    def build_pipeline(self) -> Pipeline:
        """Construye un pipeline con el modelo especificado."""
        estimator_cls = MODEL_REGISTRY[self.model_type]
        return Pipeline([("model", estimator_cls(**self.params))])


class ModelSpecLoader:
    """Carga y valida especificaciones de modelos desde configuración."""
    
    @staticmethod
    def normalise_model_type(raw_type: Optional[str]) -> str:
        """Normaliza y valida el tipo de modelo."""
        if not raw_type:
            raise ValueError("Cada modelo necesita un campo 'type'.")
        key = raw_type.strip().lower()
        if key not in MODEL_REGISTRY:
            raise ValueError(
                f"Tipo de modelo '{raw_type}' no soportado. Tipos válidos: {sorted(MODEL_REGISTRY)}"
            )
        return key
    
    @staticmethod
    def load_from_config(cfg: Dict[str, Any]) -> List[ModelSpec]:
        """Carga especificaciones de modelos desde configuración."""
        if "models" not in cfg or not cfg["models"]:
            raise ValueError("Debes definir al menos un modelo en la configuración bajo el campo 'models'.")

        raw_models = cfg["models"]
        specs: List[ModelSpec] = []

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
            model_type = ModelSpecLoader.normalise_model_type(data.get("type"))
            params = dict(data.get("params", {}))
            specs.append(ModelSpec(name, model_type, params))

        return specs


class DataSplitter:
    """Gestiona la división de datos en conjuntos de train, validation y test."""
    
    def __init__(self, test_size: float = 0.12, val_size: float = 0.2, 
                 random_state: int = 42, val_random_state: int = 43):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.val_random_state = val_random_state
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                                   pd.Series, pd.Series, pd.Series]:
        """Divide los datos en train, validation y test."""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=True, random_state=self.random_state
        )
        val_relative_size = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, shuffle=True, random_state=self.val_random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_splits(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   output_dir: Path = Path("artifacts/eval/splits")) -> Dict[str, str]:
        """Guarda los splits a disco y retorna las rutas."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / "train.csv"
        val_path = output_dir / "val.csv"
        test_path = output_dir / "test.csv"
        
        pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
        pd.concat([X_val, y_val], axis=1).to_csv(val_path, index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)
        
        return {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        }


class ModelEvaluator:
    """Evalúa y compara múltiples modelos."""
    
    def __init__(self, cv_splits: int = 5, cv_repeats: int = 3, cv_random_state: int = 7):
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.cv_random_state = cv_random_state
        self.cv_strategy = RepeatedKFold(
            n_splits=cv_splits, n_repeats=cv_repeats, random_state=cv_random_state
        )
        self.scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }
    
    @staticmethod
    def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
        """Calcula RMSE."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    def evaluate_model(self, spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evalúa un modelo específico."""
        pipeline = spec.build_pipeline()
        
        # Cross-validation
        scores = cross_validate(
            pipeline, X_train, y_train, scoring=self.scoring, cv=self.cv_strategy, n_jobs=1
        )
        
        rmse_scores = -scores["test_rmse"]
        mae_scores = -scores["test_mae"]
        r2_scores = scores["test_r2"]
        
        # Entrenar modelo final
        fitted_pipeline = spec.build_pipeline()
        fitted_pipeline.fit(X_train, y_train)
        
        # Métricas de validación
        y_val_pred = fitted_pipeline.predict(X_val)
        val_rmse = self.rmse(y_val, y_val_pred)
        val_r2 = float(r2_score(y_val, y_val_pred))
        
        # Métricas de test
        y_test_pred = fitted_pipeline.predict(X_test)
        test_rmse = self.rmse(y_test, y_test_pred)
        test_r2 = float(r2_score(y_test, y_test_pred))
        
        # Métricas de train
        y_train_pred = fitted_pipeline.predict(X_train)
        train_rmse = self.rmse(y_train, y_train_pred)
        train_r2 = float(r2_score(y_train, y_train_pred))
        
        avg_r2 = (val_r2 + test_r2) / 2
        
        return {
            "name": spec.name,
            "pipeline": fitted_pipeline,
            "cv_results": {
                "rmse_scores": rmse_scores.tolist(),
                "mae_scores": mae_scores.tolist(),
                "r2_scores": r2_scores.tolist(),
                "rmse_mean": float(np.mean(rmse_scores)),
                "rmse_std": float(np.std(rmse_scores, ddof=1)),
                "r2_mean": float(np.mean(r2_scores)),
                "r2_std": float(np.std(r2_scores, ddof=1)),
            },
            "train_metrics": {
                "rmse": train_rmse,
                "mae": float(mean_absolute_error(y_train, y_train_pred)),
                "r2": train_r2,
            },
            "val_metrics": {
                "rmse": val_rmse,
                "r2": val_r2,
            },
            "test_metrics": {
                "rmse": test_rmse,
                "mae": float(mean_absolute_error(y_test, y_test_pred)),
                "r2": test_r2,
            },
            "avg_r2": avg_r2,
        }

class ModelTrainer:
    """Orquesta el entrenamiento completo de modelos."""
    
    def __init__(self, cfg_path: str = "configs/train.yaml", experiment_name: str = "energy-consumption-prediction"):
        self.cfg_path = cfg_path
        self.cfg = None
        self.model_specs = None
        self.data_splitter = None
        self.evaluator = None
        self.experiment_name = experiment_name
        
        # Configurar MLflow
        mlflow.set_experiment(self.experiment_name)
        
    def load_configuration(self):
        """Carga la configuración desde el archivo YAML."""
        self.cfg = load_config(self.cfg_path)
        if "models" not in self.cfg or not self.cfg["models"]:
            print("[INFO] No se encontraron modelos en el YAML, usando modelos de config.py (CONFIGS['train'])")
            self.cfg["models"] = CONFIGS["train"]["models"]
        
        # Inicializar componentes
        self.data_splitter = DataSplitter(
            test_size=self.cfg.get("test_size", 0.12),
            val_size=self.cfg.get("val_size", 0.2),
            random_state=self.cfg.get("random_state", 42),
            val_random_state=self.cfg.get("val_random_state", 43)
        )
        
        self.evaluator = ModelEvaluator(
            cv_splits=self.cfg.get("cv_n_splits", 5),
            cv_repeats=self.cfg.get("cv_n_repeats", 3),
            cv_random_state=self.cfg.get("cv_random_state", 7)
        )
        
        self.model_specs = ModelSpecLoader.load_from_config(self.cfg)
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Entrena todos los modelos y selecciona el mejor."""
        import joblib
        
        # Iniciar run padre de MLflow
        with mlflow.start_run(run_name="full_training_pipeline") as parent_run:
            target = self.cfg.get("target", "usage_kwh").lower()
            X = df.drop(columns=[target])
            y = df[target]
            
            # Loguear configuración general en MLflow
            mlflow.log_params({
                "target": target,
                "test_size": self.data_splitter.test_size,
                "val_size": self.data_splitter.val_size,
                "random_state": self.data_splitter.random_state,
                "cv_n_splits": self.evaluator.cv_splits,
                "cv_n_repeats": self.evaluator.cv_repeats,
                "n_models": len(self.model_specs),
            })
            
            # Dividir datos
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_splitter.split_data(X, y)
            split_paths = self.data_splitter.save_splits(X_train, y_train, X_val, y_val, X_test, y_test)
            
            # Loguear tamaños de datasets
            mlflow.log_params({
                "n_samples_train": len(X_train),
                "n_samples_val": len(X_val),
                "n_samples_test": len(X_test),
                "n_features": X_train.shape[1],
            })
            
            # Evaluar todos los modelos
            print("Resultados de Validación y Test:")
            results = []
            best_model_name = None
            best_avg_r2 = -np.inf
            
            for spec in self.model_specs:
                # Nested run para cada modelo
                with mlflow.start_run(run_name=f"model_{spec.name}", nested=True):
                    print(f"\n{'='*60}")
                    print(f"Entrenando modelo: {spec.name} ({spec.model_type})")
                    print(f"{'='*60}")
                    
                    # Loguear parámetros del modelo
                    mlflow.log_params({
                        "model_name": spec.name,
                        "model_type": spec.model_type,
                        **{f"param_{k}": v for k, v in spec.params.items()}
                    })
                    
                    result = self.evaluator.evaluate_model(spec, X_train, y_train, X_val, y_val, X_test, y_test)
                    results.append(result)
                    
                    # Loguear métricas en MLflow
                    mlflow.log_metrics({
                        "train_rmse": result['train_metrics']['rmse'],
                        "train_mae": result['train_metrics']['mae'],
                        "train_r2": result['train_metrics']['r2'],
                        "val_rmse": result['val_metrics']['rmse'],
                        "val_r2": result['val_metrics']['r2'],
                        "test_rmse": result['test_metrics']['rmse'],
                        "test_mae": result['test_metrics']['mae'],
                        "test_r2": result['test_metrics']['r2'],
                        "avg_r2_val_test": result['avg_r2'],
                        "cv_rmse_mean": result['cv_results']['rmse_mean'],
                        "cv_rmse_std": result['cv_results']['rmse_std'],
                        "cv_r2_mean": result['cv_results']['r2_mean'],
                        "cv_r2_std": result['cv_results']['r2_std'],
                    })
                    
                    # Loguear el modelo en MLflow
                    signature = infer_signature(X_train, result['pipeline'].predict(X_train))
                    mlflow.sklearn.log_model(
                        result['pipeline'],
                        artifact_path="model",
                        signature=signature
                    )
                    
                    # Imprimir resultados
                    print(f"{result['name']}")
                    print(f"  Val:   RMSE={result['val_metrics']['rmse']:.3f}, R2={result['val_metrics']['r2']:.3f}")
                    print(f"  Test:  RMSE={result['test_metrics']['rmse']:.3f}, R2={result['test_metrics']['r2']:.3f}")
                    print(f"  Promedio R2 (val+test)/2: {result['avg_r2']:.3f}")
                    print(f"✅ Modelo {spec.name} logueado en MLflow")
                    
                    # Actualizar mejor modelo
                    if result['avg_r2'] > best_avg_r2:
                        best_avg_r2 = result['avg_r2']
                        best_model_name = result['name']
        
            print(f"\n{'='*60}")
            print(f"🏆 Mejor modelo seleccionado: {best_model_name}")
            print(f"{'='*60}")
            
            # Loguear mejor modelo en el run padre
            best_result = next(r for r in results if r['name'] == best_model_name)
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_metrics({
                "best_val_r2": best_result['val_metrics']['r2'],
                "best_test_r2": best_result['test_metrics']['r2'],
                "best_val_rmse": best_result['val_metrics']['rmse'],
                "best_test_rmse": best_result['test_metrics']['rmse'],
                "best_avg_r2": best_avg_r2,
            })
            
            # Registrar el mejor modelo en MLflow Model Registry
            best_model = best_result['pipeline']
            signature = infer_signature(X_train, best_model.predict(X_train))
            
            model_info = mlflow.sklearn.log_model(
                best_model,
                artifact_path="best_model",
                signature=signature,
                registered_model_name=f"{self.experiment_name}_best"
            )
            
            print(f"\n✅ Mejor modelo registrado en MLflow Model Registry")
            print(f"   Model URI: {model_info.model_uri}")
            
            # Construir payload
            trained_models = {r['name']: r['pipeline'] for r in results}
            train_metrics = {r['name']: r['train_metrics'] for r in results}
            test_metrics = {r['name']: r['test_metrics'] for r in results}
            avg_r2_dict = {r['name']: r['avg_r2'] for r in results}
            
            validation_results = [
                {
                    "name": r['name'],
                    "cv_rmse_scores": r['cv_results']['rmse_scores'],
                    "cv_mae_scores": r['cv_results']['mae_scores'],
                    "cv_r2_scores": r['cv_results']['r2_scores'],
                    "cv_rmse_mean": r['cv_results']['rmse_mean'],
                    "cv_rmse_std": r['cv_results']['rmse_std'],
                    "cv_r2_mean": r['cv_results']['r2_mean'],
                    "cv_r2_std": r['cv_results']['r2_std'],
                    "val_rmse": r['val_metrics']['rmse'],
                    "val_r2": r['val_metrics']['r2'],
                    "test_rmse": r['test_metrics']['rmse'],
                    "test_r2": r['test_metrics']['r2'],
                }
                for r in results
            ]
            
            payload = {
                "best_model": best_model_name,
                "models": trained_models,
                "validation_results": validation_results,
                "test_metrics": test_metrics,
                "metrics": {name: {"train": train_metrics[name], "test": test_metrics[name]} for name in trained_models},
                "config": {
                    "target": target,
                    "test_size": self.data_splitter.test_size,
                    "random_state": self.data_splitter.random_state,
                    "cv_n_splits": self.evaluator.cv_splits,
                    "cv_n_repeats": self.evaluator.cv_repeats,
                    "cv_random_state": self.evaluator.cv_random_state,
                },
                "avg_r2": avg_r2_dict,
                "split_paths": split_paths,
                "mlflow_run_id": parent_run.info.run_id,
                "mlflow_model_uri": model_info.model_uri,
            }
            
            # Guardar el payload completo en el nuevo path
            best_model_path = Path("src/proyecto_mlops/models/best_model.pkl")
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(payload, best_model_path)
            print(f"\n✅ Payload completo exportado a: {best_model_path}")
            
            # Guardar métricas básicas en reports/metrics.json para DVC
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


def run_train(df: pd.DataFrame, cfg_path: str = "configs/train.yaml"):
    """Función wrapper para mantener compatibilidad con código existente."""
    trainer = ModelTrainer(cfg_path)
    trainer.load_configuration()
    return trainer.train(df)

if __name__ == "__main__":
    cfg_path = "train.yaml"
    cfg = load_config(cfg_path)
    features_path = Path(cfg["paths"]["train"])
    if not features_path.exists():
        raise FileNotFoundError(f"No existe el archivo de features: {features_path}")
    df = pd.read_csv(features_path)
    run_train(df, cfg_path=cfg_path)

