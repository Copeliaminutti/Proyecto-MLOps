import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

## Defining the numerical features that are expected in the dataset. 
DEFAULT_NUMERIC_FEATURES = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "mixed_type_col",
]

DEFAULT_METRIC_NAMES: Dict[str, Iterable[str]] = {
    "classification": ("accuracy", "f1_weighted"),
    "regression": ("r2", "rmse"),
}

## Defining the models that are going to be used in the training process.
## Important that we are using regression models since the target variable is continuous.
BUILTIN_MODEL_REGISTRY = {
    "LinearRegression": ("sklearn.linear_model", "LinearRegression", "regression"),
    "LogisticRegression": ("sklearn.linear_model", "LogisticRegression", "classification"),
    "DecisionTreeRegressor": ("sklearn.tree", "DecisionTreeRegressor", "regression"),
    "DecisionTreeClassifier": ("sklearn.tree", "DecisionTreeClassifier", "classification"),
    "KNeighborsRegressor": ("sklearn.neighbors", "KNeighborsRegressor", "regression"),
    "KNeighborsClassifier": ("sklearn.neighbors", "KNeighborsClassifier", "classification"),
    "RandomForestRegressor": ("sklearn.ensemble", "RandomForestRegressor", "regression"),
    "RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier", "classification"),
}


def metric_f1_weighted(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def metric_f1_macro(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def metric_precision_weighted(y_true, y_pred) -> float:
    return precision_score(y_true, y_pred, average="weighted", zero_division=0)


def metric_precision_macro(y_true, y_pred) -> float:
    return precision_score(y_true, y_pred, average="macro", zero_division=0)


def metric_recall_weighted(y_true, y_pred) -> float:
    return recall_score(y_true, y_pred, average="weighted", zero_division=0)


def metric_recall_macro(y_true, y_pred) -> float:
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


def metric_rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)


METRIC_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "classification": {
        "accuracy": {"func": accuracy_score, "greater_is_better": True},
        "f1": {"func": metric_f1_macro, "greater_is_better": True},
        "f1_weighted": {"func": metric_f1_weighted, "greater_is_better": True},
        "precision": {"func": metric_precision_macro, "greater_is_better": True},
        "precision_weighted": {"func": metric_precision_weighted, "greater_is_better": True},
        "recall": {"func": metric_recall_macro, "greater_is_better": True},
        "recall_weighted": {"func": metric_recall_weighted, "greater_is_better": True},
    },
    "regression": {
        "r2": {"func": r2_score, "greater_is_better": True},
        "rmse": {"func": metric_rmse, "greater_is_better": False},
        "mse": {"func": mean_squared_error, "greater_is_better": False},
        "mae": {"func": mean_absolute_error, "greater_is_better": False},
    },
}


def load_config(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_estimator(estimator_name: str) -> Tuple[Any, Optional[str]]:
    module_name: Optional[str]
    class_name = estimator_name

    if "." in estimator_name:
        module_name, class_name = estimator_name.rsplit(".", 1)
        default_task = BUILTIN_MODEL_REGISTRY.get(class_name, (None, None, None))[2]
    else:
        if estimator_name not in BUILTIN_MODEL_REGISTRY:
            raise ValueError(f"Unsupported estimator '{estimator_name}'. Provide a full dotted path.")
        module_name, class_name, default_task = BUILTIN_MODEL_REGISTRY[estimator_name]
    estimator_cls = getattr(importlib.import_module(module_name), class_name)
    return estimator_cls, default_task


def normalize_models_config(raw_models: Any, cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if raw_models:
        if isinstance(raw_models, dict):
            models = {}
            for key, value in raw_models.items():
                if isinstance(value, dict):
                    model_cfg = dict(value)
                else:
                    model_cfg = {"estimator": value}
                model_cfg.setdefault("name", key)
                models[model_cfg["name"]] = model_cfg
            return models
        if isinstance(raw_models, list):
            models = {}
            for idx, item in enumerate(raw_models):
                if not isinstance(item, dict):
                    raise ValueError("Each item in models list must be a mapping.")
                name = item.get("name") or f"model_{idx}"
                models[name] = dict(item)
                models[name]["name"] = name
            return models
        raise ValueError("Unsupported models configuration format.")

    estimator_name = cfg.get("model", "RandomForestClassifier")
    legacy_model_cfg = {
        "name": cfg.get("model_name", estimator_name),
        "estimator": estimator_name,
        "params": cfg.get("params", {}),
    }
    if "task" in cfg:
        legacy_model_cfg["task"] = cfg["task"]
    if "metrics" in cfg and isinstance(cfg["metrics"], (list, tuple)):
        legacy_model_cfg["metrics"] = list(cfg["metrics"])
    return {legacy_model_cfg["name"]: legacy_model_cfg}


def maybe_stratify_labels(y: pd.Series, enabled: bool) -> Optional[pd.Series]:
    if not enabled:
        return None
    value_counts = y.value_counts()
    if value_counts.empty or value_counts.min() < 2:
        return None
    return y


def main(args=None):
    parser = argparse.ArgumentParser()

    ## Defining the command line arguments for configuration, input data, and output model artefacts
    parser.add_argument("--cfg", type=str, default="configs/train.yaml")
    parser.add_argument("--input", type=str, default="data/interim/features.csv")
    parser.add_argument("--out", type=str, default="artifacts/model.pkl")
    ns = parser.parse_args(args)

    cfg = load_config(ns.cfg)
    target = cfg.get("target", "Load_Type")

    df = pd.read_csv(ns.input)
    numeric_cols = cfg.get("numeric_features", DEFAULT_NUMERIC_FEATURES)
    missing_columns = [col for col in numeric_cols if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing numeric feature columns: {missing_columns}")
    X = df[numeric_cols]
    y = df[target]

    ## Apply the errors coerce to convert non-numeric entries to NaN, this helps with data 
    ## and software engineering issues where data might not be clean
    X = X.apply(pd.to_numeric, errors='coerce')
    feature_mask = X.notna().all(axis=1)
    target_mask = y.notna()
    base_mask = feature_mask & target_mask
    X = X.loc[base_mask]
    y = y.loc[base_mask]

    ## Normalize the models configuration.
    models_cfg = normalize_models_config(cfg.get("models"), cfg)

    ## Normalize the metrics configuration.
    metrics_cfg = cfg.get("metrics", {})

    ## If metrics_cfg is a list or tuple, assume it's for classification task. This is for backward compatibility. 
    if isinstance(metrics_cfg, (list, tuple)):
        metrics_cfg = {"classification": list(metrics_cfg)}
    selection_metrics_cfg = cfg.get("selection_metrics", cfg.get("selection_metric", {}))
    if isinstance(selection_metrics_cfg, str):
        selection_metrics_cfg = {
            "classification": selection_metrics_cfg,
            "regression": selection_metrics_cfg,
        }

    ## Define the training parameters.
    test_size = cfg.get("test_size", 0.2)
    random_state = cfg.get("random_state", 42)
    stratify_enabled = cfg.get("stratify", True)

    ## Containers to hold trained models, their metrics, and metadata.
    trained_models: Dict[str, Any] = {}
    model_metrics: Dict[str, Dict[str, float]] = {}
    model_metadata: Dict[str, Dict[str, Any]] = {}
    best_models: Dict[str, Dict[str, Any]] = {}

    ## Iterate over each model configuration, train the model, evaluate it, and store the results.
    for model_name, model_cfg in models_cfg.items():

        estimator_name = model_cfg.get("estimator")
        if not estimator_name:
            raise ValueError(f"Model '{model_name}' missing 'estimator' configuration.")

        ## Using the resolve_estimator function to get the model classes instantiated.
        estimator_cls, default_task = resolve_estimator(estimator_name)
        task = model_cfg.get("task") or default_task
        if task not in METRIC_REGISTRY:
            raise ValueError(f"Task '{task}' for model '{model_name}' is not supported.")

        params = model_cfg.get("params", {})
        estimator = estimator_cls(**params)

        if task == "regression":
            y_prepared = pd.to_numeric(y, errors="coerce")
            valid_mask = y_prepared.notna()
            X_model = X.loc[valid_mask]
            y_model = y_prepared.loc[valid_mask]
            stratify = None
        else:
            y_model = y
            X_model = X
            stratify = maybe_stratify_labels(y_model, stratify_enabled)

        X_train, X_test, y_train, y_test = train_test_split(
            X_model,
            y_model,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)

        default_metrics = metrics_cfg.get(task, DEFAULT_METRIC_NAMES[task])
        metric_names = list(model_cfg.get("metrics", default_metrics))

        selection_metric_name = model_cfg.get("selection_metric") or selection_metrics_cfg.get(task)
        if selection_metric_name:
            if selection_metric_name not in METRIC_REGISTRY[task]:
                raise ValueError(
                    f"Selection metric '{selection_metric_name}' is not supported for task '{task}'."
                )
            if selection_metric_name not in metric_names:
                metric_names.append(selection_metric_name)

        metrics_result: Dict[str, float] = {}
        for metric_name in metric_names:
            if metric_name not in METRIC_REGISTRY[task]:
                raise ValueError(f"Metric '{metric_name}' is not defined for task '{task}'.")
            metric_func = METRIC_REGISTRY[task][metric_name]["func"]
            metrics_result[metric_name] = metric_func(y_test, predictions)

        trained_models[model_name] = estimator
        model_metrics[model_name] = metrics_result
        model_metadata[model_name] = {"task": task, "estimator": estimator_name}

        if selection_metric_name:
            metric_value = metrics_result[selection_metric_name]
            spec = METRIC_REGISTRY[task][selection_metric_name]
            current_best = best_models.get(task)
            if not current_best:
                best_models[task] = {
                    "name": model_name,
                    "value": metric_value,
                    "greater_is_better": spec["greater_is_better"],
                }
            else:
                is_better = (
                    metric_value > current_best["value"]
                    if spec["greater_is_better"]
                    else metric_value < current_best["value"]
                )
                if is_better:
                    best_models[task].update({"name": model_name, "value": metric_value})

        print(f"Model '{model_name}' ({task}) metrics:")
        for metric_name, metric_value in metrics_result.items():
            print(f"  - {metric_name}: {metric_value:.4f}")

    best_overall = best_models.get("classification") or best_models.get("regression")
    best_model_name = best_overall["name"] if best_overall else None
    if best_model_name:
        print(f"Selected best model: {best_model_name}")

    output_payload = {
        "models": trained_models,
        "metrics": model_metrics,
        "metadata": model_metadata,
        "best_model": best_model_name,
    }

    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output_payload, ns.out)
    print(f"Model artefacts saved to {ns.out}")

if __name__ == "__main__":
    main()
