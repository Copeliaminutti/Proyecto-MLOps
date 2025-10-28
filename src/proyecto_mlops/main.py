import subprocess
import sys

def main():
    try:
        print("▶Ejecutando pipeline con DVC: dvc repro")
        result = subprocess.run(["dvc", "repro"], check=False)
        if result.returncode != 0:
            print("dvc repro terminó con errores (returncode =", result.returncode, ")")
            sys.exit(result.returncode)
        print("Pipeline completado correctamente")
    except FileNotFoundError:
        print("No se encontró 'dvc'. Instálalo con: pip install dvc")
        sys.exit(1)

if __name__ == "__main__":
    main()
