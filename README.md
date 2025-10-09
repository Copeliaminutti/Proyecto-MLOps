# Proyecto MLOps

Monorepo del proyecto de ML de punta a punta (EDA → features → entrenamiento → evaluación → despliegue → monitoreo).

## Estructura
```
proyecto-mlops/
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
