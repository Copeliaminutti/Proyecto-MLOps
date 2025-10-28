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

---

## 01. Estructura Cookiecutter (Fase 2)

El proyecto adopta la estructura estándar propuesta por Cookiecutter Data Science, 
lo que garantiza organización, colaboración y escalabilidad en el desarrollo de proyectos de Machine Learning.

**Principales directorios:**
- `src/proyecto_mlops/`: código modular del pipeline (data, features, modelling, utils, serving, visualization).
- `configs/`: archivos YAML de configuración y parámetros de entrenamiento.
- `data/`: conjunto de datos en sus etapas (raw, interim, processed, external).
- `notebooks/`: análisis exploratorio y experimentos.
- `reports/`: resultados, métricas y visualizaciones finales.
- `models/`: artefactos de modelos entrenados.
- `docs/`: documentación del proyecto.

**Importancia:**
Una estructura estandarizada facilita:
1. la colaboración entre equipos,  
2. el mantenimiento y versionado,  
3. la reproducibilidad de experimentos, y  
4. la integración de herramientas como DVC y MLflow.

---

