# Proyecto MLOps

Este proyecto sigue la estructura estandarizada de Cookiecutter Data Science (CCDS) y aplica las mejores prácticas de MLOps para garantizar escalabilidad, trazabilidad y reproducibilidad del ciclo de vida de Machine Learning.

## Estructura
proyecto-mlops/
├─ README.md
├─ .gitignore
├─ .gitattributes
├─ requirements.txt
├─ environment.yml
├─ Makefile
├─ dvc.yaml
├─ notebooks/                  # Análisis exploratorio y pruebas experimentales
├─ src/
│  └─ proyecto_mlops/
│     ├─ data/                 # Scripts de limpieza y preparación de datos
│     ├─ features/             # Ingeniería y construcción de características
│     ├─ modelling/            # Entrenamiento y evaluación de modelos
│     ├─ models/               # Artefactos del modelo entrenado (.pkl)
│     ├─ serving/              # API o despliegue del modelo
│     ├─ utils/                # Funciones auxiliares y rutas comunes
│     └─ visualization/        # Gráficos y reportes automáticos
├─ configs/                    # Archivos YAML con parámetros y configuraciones
├─ tests/                      # Pruebas unitarias y validaciones
├─ data/                       # Datasets (versionados con DVC)
│   ├─ raw/                    # Datos originales
│   ├─ processed/              # Datos limpios
│   └─ interim/                # Datos intermedios para modelado
└─ .github/workflows/          # Automatizaciones (CI/CD)

```

```bash

## Flujo de trabajo
l flujo de trabajo está definido en dvc.yaml, donde cada etapa se ejecuta de forma modular:

EDA / Limpieza → src/proyecto_mlops/data/clean.py

Features → src/proyecto_mlops/features/build_features.py

Entrenamiento → src/proyecto_mlops/modelling/train.py

Evaluación → src/proyecto_mlops/modelling/evaluate.py

Cada etapa genera sus salidas (data/processed/, data/interim/, artifacts/model.pkl, etc.), y el flujo puede ejecutarse automáticamente con:
dvc repro
# o desde Python
python src/proyecto_mlops/main.py


## Fases 
EDA → notebooks/00_eda.ipynb + src/proyecto_mlops/data/*

Features → src/proyecto_mlops/features/*

Entrenamiento → src/proyecto_mlops/modelling/train.py

Evaluación → src/proyecto_mlops/modelling/evaluate.py

Despliegue → src/proyecto_mlops/serving/*

Monitoreo → src/proyecto_mlops/serving/monitoring/*
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

