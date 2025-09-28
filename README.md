# Tu Proyecto MLOps

Monorepo para un proyecto de ML de punta a punta (EDA → features → entrenamiento → evaluación → despliegue → monitoreo).

## Estructura
```
tu-proyecto-mlops/
├─ README.md
├─ .gitignore
├─ .gitattributes
├─ requirements.txt
├─ environment.yml
├─ Makefile
├─ dvc.yaml
├─ notebooks/
├─ src/
├─ configs/
├─ tests/
├─ data/              # versionar con DVC/LFS (no subir crudos a Git)
└─ .github/workflows/
```

## Cómo empezar
```bash
# Clonar (reemplaza usuario)
git clone https://github.com/tu-usuario/tu-proyecto-mlops.git
cd tu-proyecto-mlops

# (opcional) crear y activar entorno
python -m venv .venv && source .venv/bin/activate  # win: .venv\Scripts\activate

# instalar dependencias
pip install -r requirements.txt

# inicializar DVC en tu entorno local
dvc init
# (opcional) configura un remoto:
# dvc remote add -d storage gdrive://<id_carpeta>

# ejecutar pre-commit localmente
pre-commit install
pre-commit run --all-files
```

## Flujo de trabajo
- Ramas: `main` (estable), `dev` (integración) y ramas por feature: `feat/eda`, `feat/features`, `feat/train`, etc.
- Pull Request con revisión cruzada y descripción de rol + resultados.
- Artefactos y datasets versionados con DVC.

## Fases (mapa)
1. EDA → `notebooks/00_eda.ipynb` + `src/data/*`
2. Features → `src/features/*`
3. Entrenamiento → `src/models/train.py`
4. Evaluación → `src/models/evaluate.py`
5. Despliegue → `src/serving/*`
6. Monitoreo → `src/serving/monitoring/*` (añadir según necesidad)
