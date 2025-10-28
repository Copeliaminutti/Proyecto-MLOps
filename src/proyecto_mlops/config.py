# -*- coding: utf-8 -*-
"""Configuración del proyecto.

- Modo recomendado: `load_config("train.yaml")` lee YAML desde /configs
- Modo legacy: `CONFIGS` (dict estático) se conserva para compatibilidad.
"""

from pathlib import Path
import yaml

def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / 'pyproject.toml').exists() or (cur / '.git').exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve().parents[2]

ROOT = _find_repo_root(Path(__file__).parent)
CONFIG_DIR = ROOT / 'configs'

def load_config(name: str = 'train.yaml'):
    """Carga y devuelve un dict desde configs/<name> (YAML)."""
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f'No se encontró {path}. Asegura configs/{name}.')
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- Legacy (se conserva tu dict original para compatibilidad) ---
# Configuración migrada de train.yaml

CONFIGS = {
    "train": {
        "target": "usage_kwh",
        "target": "usage_kwh",
        "models": {
            "Ridge": {
                "type": "ridge",
                "params": {
                    "alpha": 1.0,
                    "fit_intercept": True,
                    "random_state": 42,
                },
            },
            "LinearRegression": {
                "type": "linear_regression",
                "params": {
                    "fit_intercept": True,
                    "copy_X": True,
                    "n_jobs": 1,
                },
            },
            "kNN": {
                "type": "knn",
                "params": {
                    "n_neighbors": 50,
                    "weights": "distance",
                    "metric": "minkowski",
                    "p": 2,
                },
            },
            "RegressionTree": {
                "type": "decision_tree",
                "params": {
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                },
            },
            "RandomForestRegressor": {
                "type": "random_forest",
                "params": {
                    "n_estimators": 15,
                    "max_depth": 4,
                },
                "metrics": ["accuracy", "f1"],
            },
        },
        "test_size": 0.12,
        "val_size": 0.2,
        "random_state": 42,
        "val_random_state": 43,
    },
    "features": {
        "scaling": "standard",
        "one_hot": True,
        "input": "data/raw/steel_energy_model_fullgrid.csv",
        "numeric_features": [
            "lagging_current_reactive_power_kvarh",
            "leading_current_reactive_power_kvarh",
            "co2_tco2",
            "lagging_current_power_factor",
            "leading_current_power_factor",
            "nsm",
            "mixed_type_col",
        ],
        "categorical_features": [
            "week_status",
            "day_of_week",
            "load_type",
        ],
        "target": "usage_kwh",
    },
}


__all__ = ["load_config", "CONFIG_DIR", "ROOT", "CONFIGS"]
