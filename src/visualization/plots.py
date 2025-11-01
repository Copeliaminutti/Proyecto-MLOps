import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import mlflow
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MetricsCalculator:
    """Calcula métricas de regresión."""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula R2 Score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    @classmethod
    def compute_all_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula todas las métricas disponibles."""
        return {
            "rmse": cls.rmse(y_true, y_pred),
            "mae": cls.mae(y_true, y_pred),
            "r2": cls.r2(y_true, y_pred),
        }


class ModelPlotter:
    """Genera visualizaciones para evaluación de modelos."""
    
    def __init__(self, output_dir: Path, dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_actual_vs_predicted(
        self, y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str
    ) -> Path:
        """Genera gráfico de valores reales vs predicciones."""
        output_path = self.output_dir / filename
        
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="white")
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Valores reales")
        plt.ylabel("Predicciones")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def plot_residual_histogram(
        self, y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str
    ) -> Path:
        """Genera histograma de residuales."""
        output_path = self.output_dir / filename
        residuals = y_true - y_pred
        
        plt.figure(figsize=(6, 5))
        plt.hist(residuals, bins=30, color="#ff7f0e", edgecolor="black", alpha=0.75)
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Residual")
        plt.ylabel("Frecuencia")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def generate_all_plots(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, split_name: str
    ) -> Dict[str, Path]:
        """Genera todos los gráficos para un modelo y split."""
        plots = {}
        
        # Gráfico de predicciones vs reales
        plots['actual_vs_pred'] = self.plot_actual_vs_predicted(
            y_true, y_pred,
            f"{model_name} - {split_name}",
            f"{model_name}_{split_name}_actual_vs_pred.png"
        )
        
        # Histograma de residuales
        plots['residuals'] = self.plot_residual_histogram(
            y_true, y_pred,
            f"{model_name} - Residuales ({split_name})",
            f"{model_name}_{split_name}_residuals.png"
        )
        
        return plots


class DataLoader:
    """Carga y prepara datos para evaluación."""
    
    @staticmethod
    def load_split(path_str: str, target: str) -> Dict[str, Any]:
        """Carga un split de datos (train/val/test)."""
        split_df = pd.read_csv(path_str)
        y = split_df[target].to_numpy()
        X = split_df.drop(columns=[target])
        return {"X": X, "y": y}
    
    @staticmethod
    def load_model_payload(artifact_path: str) -> Dict[str, Any]:
        """Carga el payload completo del modelo entrenado."""
        return joblib.load(artifact_path)

class ModelEvaluationPipeline:
    """Pipeline completo de evaluación de modelos."""
    
    def __init__(
        self,
        model_artifact_path: str = "models/best_model.pkl",
        plots_dir: str = "reports/figures",
        report_json_path: str = "artifacts/eval/metrics.json",
        log_to_mlflow: bool = True
    ):
        self.model_artifact_path = model_artifact_path
        self.plots_dir = plots_dir
        self.report_json_path = report_json_path
        self.log_to_mlflow = log_to_mlflow
        
        self.data_loader = DataLoader()
        self.metrics_calculator = MetricsCalculator()
        self.plotter = None  # Se inicializa en run()
    
    def run(self) -> None:
        """Ejecuta el pipeline completo de evaluación."""
        # Cargar payload del modelo
        payload = self.data_loader.load_model_payload(self.model_artifact_path)
        best_model_name: str = payload["best_model"]
        models: Dict[str, Any] = payload["models"]
        metrics: Dict[str, Dict[str, Dict[str, float]]] = payload["metrics"]
        split_paths: Dict[str, str] = payload["split_paths"]
        target = payload["config"]["target"]
        mlflow_run_id = payload.get("mlflow_run_id")
        
        # Usar el mismo run_id del entrenamiento si existe
        if self.log_to_mlflow and mlflow_run_id:
            with mlflow.start_run(run_id=mlflow_run_id):
                self._run_evaluation(payload, best_model_name, models, metrics, split_paths, target)
        else:
            self._run_evaluation(payload, best_model_name, models, metrics, split_paths, target)
    
    def _run_evaluation(
        self, payload, best_model_name, models, metrics, split_paths, target
    ) -> None:
        """Ejecuta la evaluación (con o sin MLflow)."""
        # Cargar datos de validación y test
        val_split = self.data_loader.load_split(split_paths["val"], target)
        test_split = self.data_loader.load_split(split_paths["test"], target)
        
        # Obtener mejor modelo
        best_model = models[best_model_name]
        
        # Generar predicciones
        y_val_pred = best_model.predict(val_split["X"])
        y_test_pred = best_model.predict(test_split["X"])
        
        # Calcular métricas
        val_metrics = self.metrics_calculator.compute_all_metrics(val_split["y"], y_val_pred)
        test_metrics = self.metrics_calculator.compute_all_metrics(test_split["y"], y_test_pred)
        train_metrics = metrics[best_model_name]["train"]
        
        # Loguear métricas en MLflow
        if self.log_to_mlflow:
            mlflow.log_metrics({
                "eval_val_rmse": val_metrics["rmse"],
                "eval_val_mae": val_metrics["mae"],
                "eval_val_r2": val_metrics["r2"],
                "eval_test_rmse": test_metrics["rmse"],
                "eval_test_mae": test_metrics["mae"],
                "eval_test_r2": test_metrics["r2"],
            })
        
        # Generar visualizaciones
        self.plotter = ModelPlotter(output_dir=self.plots_dir)
        val_plots = self.plotter.generate_all_plots(val_split["y"], y_val_pred, best_model_name, "val")
        test_plots = self.plotter.generate_all_plots(test_split["y"], y_test_pred, best_model_name, "test")
        
        # Loguear gráficos en MLflow
        if self.log_to_mlflow:
            for plot_name, plot_path in {**val_plots, **test_plots}.items():
                mlflow.log_artifact(str(plot_path))
            print(f"✅ Gráficos logueados en MLflow")
        
        # Construir reporte
        report = {
            "best_model": best_model_name,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "val_metrics": val_metrics,
            "all_models_metrics": metrics,
            "split_paths": split_paths,
        }
        
        # Guardar reporte
        eval_metrics_path = Path(self.report_json_path)
        eval_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        eval_metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        
        # Loguear reporte en MLflow
        if self.log_to_mlflow:
            mlflow.log_artifact(str(eval_metrics_path))
            print(f"✅ Reporte logueado en MLflow")
        
        # Imprimir resultados
        self._print_results(val_metrics, test_metrics, eval_metrics_path)
    
    def _print_results(
        self, val_metrics: Dict[str, float], test_metrics: Dict[str, float], report_path: Path
    ) -> None:
        """Imprime los resultados de la evaluación."""
        print("Evaluación (val) del mejor modelo:")
        print(f"  RMSE: {val_metrics['rmse']:.3f}")
        print(f"  MAE : {val_metrics['mae']:.3f}")
        print(f"  R2  : {val_metrics['r2']:.3f}")
        print("\nEvaluación (test) del mejor modelo:")
        print(f"  RMSE: {test_metrics['rmse']:.3f}")
        print(f"  MAE : {test_metrics['mae']:.3f}")
        print(f"  R2  : {test_metrics['r2']:.3f}")
        print(f"\nReporte guardado en: {report_path}")


def main(args: Optional[Any] = None) -> None:
    """Función main para compatibilidad con CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=str, default="models/best_model.pkl")
    parser.add_argument("--plots-dir", type=str, default="reports/figures")
    parser.add_argument("--report-json", type=str, default="artifacts/eval/metrics.json")
    ns = parser.parse_args(args)
    
    pipeline = ModelEvaluationPipeline(
        model_artifact_path=ns.model_artifact,
        plots_dir=ns.plots_dir,
        report_json_path=ns.report_json
    )
    pipeline.run()

if __name__ == "__main__":
    main()

# --- Compatibilidad: API programática ---------------------------------
def plots(*args, **kwargs):
    """Wrapper para usar este módulo desde Python (sin CLI).
    Equivalente a ejecutar `main()` sin argumentos desde código.
    """
    return main([])

if __name__ == "__main__":
    main()
