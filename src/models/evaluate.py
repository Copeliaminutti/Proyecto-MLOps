import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="white")
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Valores reales")
    plt.ylabel("Predicciones")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residual_histogram(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=30, color="#ff7f0e", edgecolor="black", alpha=0.75)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Residual")
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def load_split(path_str: str, target: str) -> Dict[str, Any]:
    split_df = pd.read_csv(path_str)
    y = split_df[target].to_numpy()
    X = split_df.drop(columns=[target])
    return {"X": X, "y": y}


def main(args: Optional[Any] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=str, default="artifacts/model.pkl")
    parser.add_argument("--plots-dir", type=str, default="artifacts/eval/plots")
    parser.add_argument("--report-json", type=str, default="artifacts/eval/metrics.json")
    ns = parser.parse_args(args)

    payload = joblib.load(ns.model_artifact)
    best_model_name: str = payload["best_model"]
    models: Dict[str, Any] = payload["models"]
    metrics: Dict[str, Dict[str, Dict[str, float]]] = payload["metrics"]
    split_paths: Dict[str, str] = payload["split_paths"]
    target = payload["config"]["target"]

    val_split = load_split(split_paths["val"], target)
    test_split = load_split(split_paths["test"], target)

    best_model = models[best_model_name]

    y_val_pred = best_model.predict(val_split["X"])
    y_test_pred = best_model.predict(test_split["X"])

    val_metrics = compute_metrics(val_split["y"], y_val_pred)
    test_metrics = compute_metrics(test_split["y"], y_test_pred)
    train_metrics = metrics[best_model_name]["train"]

    plots_dir = Path(ns.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_actual_vs_predicted(
        val_split["y"],
        y_val_pred,
        f"{best_model_name} - Validación",
        plots_dir / f"{best_model_name}_val_actual_vs_pred.png",
    )
    plot_residual_histogram(
        val_split["y"],
        y_val_pred,
        f"{best_model_name} - Residuales (Val)",
        plots_dir / f"{best_model_name}_val_residuals.png",
    )
    plot_actual_vs_predicted(
        test_split["y"],
        y_test_pred,
        f"{best_model_name} - Test",
        plots_dir / f"{best_model_name}_test_actual_vs_pred.png",
    )
    plot_residual_histogram(
        test_split["y"],
        y_test_pred,
        f"{best_model_name} - Residuales (Test)",
        plots_dir / f"{best_model_name}_test_residuals.png",
    )

    report = {
        "best_model": best_model_name,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "all_models_metrics": metrics,
        "split_paths": split_paths,
    }

    report_path = Path(ns.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Evaluación (val) del mejor modelo:")
    print(f"  RMSE: {val_metrics['rmse']:.3f}")
    print(f"  MAE : {val_metrics['mae']:.3f}")
    print(f"  R2  : {val_metrics['r2']:.3f}")
    print("\nEvaluación (test) del mejor modelo:")
    print(f"  RMSE: {test_metrics['rmse']:.3f}")
    print(f"  MAE : {test_metrics['mae']:.3f}")
    print(f"  R2  : {test_metrics['r2']:.3f}")
    print(f"\nReporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
