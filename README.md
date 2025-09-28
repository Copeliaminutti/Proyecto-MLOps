# Proyecto MLOps

Monorepo del proyecto de ML de punta a punta (EDA → features → entrenamiento → evaluación → despliegue → monitoreo).

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
├─ data/              # versionar con DVC/LFS 
└─ .github/workflows/
```

```bash
# Clonar 
git clone https://github.com/tu-usuario/tu-proyecto-mlops.git
cd proyecto-mlops

# (opcional) 
python -m venv .venv && source .venv/bin/activate  # win: .venv\Scripts\activate

# Dependencias
pip install -r requirements.txt

# inicializar DVC en entorno local
dvc init
# (opcional remoto)
# dvc remote add -d storage gdrive://<id_carpeta>

# ejecutar pre-commit localmente
pre-commit install
pre-commit run --all-files
```

## Flujo de trabajo
- Ramas: `main` (estable), `dev` (integración) y ramas por feature: `feat/eda`, `feat/features`, `feat/train`, etc.
- Pull Request con revisión cruzada y descripción de rol + resultados.
- Artefactos y datasets versionados con DVC.

## Fases 
1. EDA → `notebooks/00_eda.ipynb` + `src/data/*`
2. Features → `src/features/*`
3. Entrenamiento → `src/models/train.py`
4. Evaluación → `src/models/evaluate.py`
5. Despliegue → `src/serving/*`
6. Monitoreo → `src/serving/monitoring/*` 
