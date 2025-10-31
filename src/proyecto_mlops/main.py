import subprocess
import sys
import os
from pathlib import Path


def main():
    # Configurar MLflow tracking URI
    mlruns_path = Path("./mlruns").resolve()
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_path}"
    
    try:
        print("="*60)
        print("🚀 Ejecutando pipeline con DVC + MLflow tracking")
        print("="*60)
        print(f"📁 MLflow tracking URI: {mlruns_path}\n")
        
        result = subprocess.run(["dvc", "repro"], check=False)
        
        if result.returncode != 0:
            print(f"\n❌ dvc repro terminó con errores (returncode = {result.returncode})")
            sys.exit(result.returncode)
        
        print("\n" + "="*60)
        print("✅ Pipeline completado correctamente")
        print("="*60)
        print("\n📊 Para ver resultados en MLflow UI, ejecuta:")
        print("   python src/proyecto_mlops/mlflow_server.py")
        print("\n   Luego abre en tu navegador: http://localhost:5000")
        print("="*60)
        
    except FileNotFoundError:
        print("❌ No se encontró 'dvc'. Instálalo con: pip install dvc")
        sys.exit(1)


if __name__ == "__main__":
    main()
