import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
from sklearn.linear_model import Ridge
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
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

DEFAULT_TARGET = "Usage_kWh"
DEFAULT_NUMERIC_FEATURES = [
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "mixed_type_col",
]

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
    
def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


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


def build_pipeline(model_type: str, params: Dict[str, Any], numeric_cols: Sequence[str], cat_cols: Sequence[str] = None) -> Pipeline:
    estimator_cls = MODEL_REGISTRY[model_type]
    # Only numeric features, as one-hot encoding is done in features
    ct = ColumnTransformer(transformers=[("num", "passthrough", list(numeric_cols))], remainder="drop")
    steps = [("ct", ct), ("model", estimator_cls(**params))]
    return Pipeline(steps)


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def plot_rmse_boxplot(data: List[np.ndarray], labels: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel("RMSE")
    plt.title("RMSE - Modelos")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/train.yaml")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--out", type=str, default="artifacts/model.pkl")
    parser.add_argument("--plots-dir", type=str, default="artifacts/plots")
    parser.add_argument("--show-plots", action="store_true")
    ns = parser.parse_args(args)

    cfg = load_config(ns.cfg)
    target = cfg.get("target", DEFAULT_TARGET)
    numeric_cols = cfg.get("numeric_features", DEFAULT_NUMERIC_FEATURES)
    input_path = cfg.get("input", ns.input or "data/interim/features.csv")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip().str.lower()
    numeric_cols = [c.lower() for c in numeric_cols]
    target = target.lower()
    # Only use numeric columns (including dummies from features)
    X_all = df[numeric_cols].copy()
    X_all[numeric_cols] = X_all[numeric_cols].apply(pd.to_numeric, errors="coerce")
    y_all = pd.to_numeric(df[target], errors="coerce")

    mask = X_all[numeric_cols].notna().all(axis=1) & y_all.notna()
    X = X_all.loc[mask].reset_index(drop=True)
    y = y_all.loc[mask].reset_index(drop=True)

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

    print("Resultados de Validación (CV):")
    best_model_name: Optional[str] = None
    best_r2 = -np.inf

    for spec in model_specs:
        name = spec["name"]
        pipeline = build_pipeline(spec["type"], spec["params"], numeric_cols)

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

        print(name)
        print("RMSE: >> %.3f (%.3f)" % (mean_rmse, std_rmse))
        print("R2:   >> %.3f (%.3f)" % (mean_r2, std_r2))
        print()

        fitted_pipeline = build_pipeline(spec["type"], spec["params"], numeric_cols)
        fitted_pipeline.fit(X_train, y_train)
        trained_models[name] = fitted_pipeline

        # Train metrics
        y_train_pred = fitted_pipeline.predict(X_train)
        train_metrics[name] = {
            "rmse": rmse(y_train, y_train_pred),
            "mae": float(mean_absolute_error(y_train, y_train_pred)),
            "r2": float(r2_score(y_train, y_train_pred)),
        }

        # Test metrics
        y_test_pred = fitted_pipeline.predict(X_test)
        test_metrics[name] = {
            "rmse": rmse(y_test, y_test_pred),
            "mae": float(mean_absolute_error(y_test, y_test_pred)),
            "r2": float(r2_score(y_test, y_test_pred)),
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
            }
        )

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_model_name = name

    if best_model_name is None:
        raise RuntimeError("No se entrenó ningún modelo correctamente.")

    plots_dir = Path(ns.plots_dir)
    plot_rmse_boxplot(boxplot_data, [spec["name"] for spec in model_specs], plots_dir / "rmse_boxplot.png")

    # Save splits to disk for evaluation
    split_dir = Path("artifacts/splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    train_split_path = split_dir / "train.csv"
    test_split_path = split_dir / "test.csv"
    val_split_path = split_dir / "val.csv"

    train_df = X_train.copy()
    train_df[target] = y_train
    train_df.to_csv(train_split_path, index=False)

    test_df = X_test.copy()
    test_df[target] = y_test
    test_df.to_csv(test_split_path, index=False)

    val_df = X_val.copy()
    val_df[target] = y_val
    val_df.to_csv(val_split_path, index=False)

    split_paths = {
        "train": str(train_split_path),
        "test": str(test_split_path),
        "val": str(val_split_path),
    }

    payload = {
        "best_model": best_model_name,
        "models": trained_models,
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "metrics": {name: {"train": train_metrics[name], "test": test_metrics[name]} for name in trained_models},
        "split_paths": split_paths,
        "config": {
            "target": target,
            "numeric_features": numeric_cols,
            "test_size": test_size,
            "random_state": random_state,
            "cv_n_splits": cv_splits,
            "cv_n_repeats": cv_repeats,
            "cv_random_state": cv_random_state,
        },
    }

    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, ns.out)

    print(f"Mejor modelo seleccionado (según R2 en validación): {best_model_name}")
    print("Métricas en el conjunto de prueba:")
    for name, metrics in test_metrics.items():
        print(f"- {name}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
