import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.linear_model import Ridge


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from proyecto_mlops.config import CONFIGS
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DEFAULT_TARGET = "usage_kwh"

MODEL_TYPE_ALIASES = {
    "linear_regression": "linear_regression",
    "linear": "linear_regression",
    "lr": "linear_regression",
    "ridge": "ridge",
    "regression_ridge": "ridge",
    "regresion_ridge": "ridge",
    "regression_lineal": "linear_regression",
    "regresion_lineal": "linear_regression",
    "k_neighbors": "knn",
    "k-nearest": "knn",
    "k-nearest-neighbors": "knn",
    "knearest": "knn",
    "knn": "knn",
    "regression_tree": "decision_tree",
    "decision_tree": "decision_tree",
    "tree": "decision_tree",
    "svm": "svm",
    "svr": "svm",
    "support_vector_machine": "svm",
    "support_vector_regression": "svm",
    "random_forest": "random_forest",
    "rf": "random_forest",
    "gradient_boosting": "gradient_boosting",
    "gbr": "gradient_boosting",

}

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
    if key not in MODEL_TYPE_ALIASES:
        raise ValueError(
            f"Tipo de modelo '{raw_type}' no soportado. Tipos válidos: {sorted(MODEL_TYPE_ALIASES)}"
        )
    return MODEL_TYPE_ALIASES[key]


def build_default_models() -> List[Dict[str, Any]]:
    return [
        {"name": "LinearRegression", "type": "linear_regression", "params": {}},
        {"name": "Ridge", "type": "ridge", "params": {"alpha": 1.0, "fit_intercept": True, "random_state": 42}},
        {"name": "KNN", "type": "knn", "params": {"n_neighbors": 50, "weights": "distance"}},
        {
            "name": "RegressionTree",
            "type": "decision_tree",
            "params": {"max_depth": 30, "min_samples_leaf": 1000, "random_state": 2},
        },
        {
            "name": "RandomForest",
            "type": "decision_tree",
            "params": {
                "max_depth": 20,
                "min_samples_leaf": 10,
                "random_state": 42
            }
        },
    ]


def load_model_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "models" not in cfg or not cfg["models"]:
        return build_default_models()

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


def plot_rmse_boxplot(data: List[np.ndarray], labels: List[str], output_path: Path) -> None:

    def run_train(df: pd.DataFrame, out_path: str = "artifacts/model.pkl"):
        cfg = CONFIGS["train"]
        target = cfg.get("target", "usage_kwh").lower()
        df.columns = df.columns.str.strip().str.lower()
        target = target.lower()
        # Use all columns except the target for X
        X = df.drop(columns=[target])
        y = pd.to_numeric(df[target], errors="coerce")
        mask = y.notna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

        test_size = cfg.get("test_size", 0.12)
        val_size = cfg.get("val_size", 0.2)
        random_state = cfg.get("random_state", 42)
        val_random_state = cfg.get("val_random_state", 43)

        # First split off test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=True,
            random_state=random_state,
        )
        # Then split temp into train and val
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_relative_size,
            shuffle=True,
            random_state=val_random_state,
        )

        cv_splits = cfg.get("cv_n_splits", 5)
        cv_repeats = cfg.get("cv_n_repeats", 3)
        cv_random_state = cfg.get("cv_random_state", 7)

        cv_strategy = RepeatedKFold(
            n_splits=cv_splits,
            n_repeats=cv_repeats,
            random_state=cv_random_state,
        )

        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }

        model_specs = load_model_specs(cfg)
        boxplot_data: List[np.ndarray] = []
        validation_results: List[Dict[str, Any]] = []
        trained_models: Dict[str, Pipeline] = {}
        test_metrics: Dict[str, Dict[str, float]] = {}
        train_metrics: Dict[str, Dict[str, float]] = {}

        print("Resultados de Validación y Test:")
        best_model_name: Optional[str] = None
        best_avg_r2 = -np.inf
        avg_r2_dict = {}

        for spec in model_specs:
            name = spec["name"]
            pipeline = build_pipeline(spec["type"], spec["params"])

            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                scoring=scoring,
                cv=cv_strategy,
                n_jobs=1,
            )

            rmse_scores = -scores["test_rmse"]
            mae_scores = -scores["test_mae"]
            r2_scores = scores["test_r2"]

            boxplot_data.append(rmse_scores)

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
        }

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(payload, out_path)


def run_train(df: pd.DataFrame, out_path: str = "artifacts/model.pkl"):
    cfg = CONFIGS["train"]
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
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(payload, out_path)
    print(f"Mejor modelo seleccionado (según promedio R2 val+test): {best_model_name}")


