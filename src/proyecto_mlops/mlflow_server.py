#!/usr/bin/env python3
"""Script para iniciar el servidor de MLflow UI."""
import subprocess
import sys
from pathlib import Path


def start_mlflow_ui(tracking_uri: str = "./mlruns", port: int = 5000):
    """Inicia el servidor de MLflow UI."""
    mlruns_path = Path(tracking_uri).resolve()
    mlruns_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"🚀 Iniciando MLflow UI en http://localhost:{port}")
    print(f"📁 Tracking URI: {mlruns_path}")
    print("="*60)
    print("\n💡 Presiona Ctrl+C para detener el servidor\n")
    
    try:
        subprocess.run([
            "mlflow", "ui",
            "--backend-store-uri", f"file://{mlruns_path}",
            "--port", str(port),
            "--host", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Servidor MLflow detenido")
        sys.exit(0)
    except FileNotFoundError:
        print("❌ No se encontró 'mlflow'. Instálalo con: pip install mlflow")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inicia MLflow UI")
    parser.add_argument("--port", type=int, default=5000, help="Puerto del servidor (default: 5000)")
    parser.add_argument("--tracking-uri", type=str, default="./mlruns", help="Ruta de mlruns (default: ./mlruns)")
    args = parser.parse_args()
    
    start_mlflow_ui(args.tracking_uri, args.port)
