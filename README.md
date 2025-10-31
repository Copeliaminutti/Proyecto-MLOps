# Proyecto MLOps  

Este proyecto sigue la estructura estandarizada de Cookiecutter Data Science (CCDS) y aplica las mejores prácticas de MLOps para garantizar escalabilidad, trazabilidad y reproducibilidad del ciclo de vida de Machine Learning.

---

## Estructura del Proyecto

```text
proyecto-mlops/
│
├── README.md                # Documentación principal del proyecto
├── .gitignore               # Archivos y carpetas ignoradas por Git
├── requirements.txt         # Librerías y dependencias del proyecto
├── environment.yml          # Entorno reproducible (conda/env)
├── Makefile                 # Comandos automatizados
├── dvc.yaml                 # Definición de etapas del pipeline (DVC)
│
├── notebooks/               # Análisis exploratorio y experimentación
│
├── src/                     # Código fuente del proyecto
│   └── proyecto_mlops/
│       ├── data/            # Limpieza y preparación de datos
│       ├── features/        # Ingeniería y construcción de variables
│       ├── modelling/       # Entrenamiento y evaluación de modelos
│       ├── models/          # Artefactos del modelo entrenado (.pkl/.joblib)
│       ├── serving/         # API / inferencia / despliegue
│       ├── utils/           # Funciones auxiliares y utilidades comunes
│       └── visualization/   # Gráficos y reportes automáticos
│
├── configs/                 # Archivos YAML con parámetros y configuración
│
├── tests/                   # Pruebas unitarias / validaciones
│
├── data/                    # Datasets versionados (DVC)
│   ├── raw/                 # Datos crudos / originales
│   ├── processed/           # Datos limpios listos para modelar
│   └── interim/             # Datos intermedios (splits, features, etc.)
│
└── .github/workflows/       # Automatizaciones (CI/CD)


```

---

### Nivel raíz  

Contiene la configuración y control del proyecto completo:  

**README.md**, **.gitignore**, **requirements.txt**, **environment.yml**, **Makefile**, **dvc.yaml**  
→ Son la documentación, dependencias, automatización y definición del pipeline.  

---

### `notebooks/`  

Espacio para el **análisis exploratorio (EDA)** y pruebas de los experimentos.  
Aquí se trabajaron las fases iniciales de exploración y limpieza manual.  

---

### `src/proyecto_mlops/`  

Es el **núcleo del código** y sigue el estándar de **Cookiecutter Data Science**:  

- `data/` → Limpieza, carga y partición de datos.  
- `features/` → Ingeniería de características.  
- `modelling/` → Entrenamiento y evaluación de modelos (donde trabaja el equipo de modelado).  
- `models/` → Artefactos entrenados (`.pkl`, `.joblib`).  
- `serving/` → Scripts o API de despliegue.  
- `utils/` → Funciones auxiliares (rutas, E/S, helpers).  
- `visualization/` → Gráficas y reportes automáticos.  

Además incluye **`main.py`** como punto de entrada del pipeline,  
que ejecuta `dvc repro` para reproducir todas las etapas automáticamente.  

---

### `configs/`  

Archivos **.yaml** con los parámetros del modelo, datos, partición y evaluación.  
Es clave para mantener **reproducibilidad y flexibilidad** sin modificar el código fuente.  

---

###  `tests/`  

Donde se ubican las **pruebas unitarias y validaciones**,  
importantes para asegurar la integridad del pipeline (fase CI/CD).  

---

### `data/`  

Estructura **versionada con DVC**, separando claramente los estados de los datos:  

- `raw/` → Datos originales.  
- `processed/` → Datos limpios.  
- `interim/` → Datos intermedios o generados durante el pipeline.  

---

##  Flujo de Trabajo del Proyecto MLOps  

Este flujo describe cómo se conecta cada etapa del proyecto, desde la preparación de datos hasta el seguimiento del modelo, aplicando las mejores prácticas de **MLOps** y **Cookiecutter Data Science (CCDS)**.  

---

### 1. Ingesta y versionamiento de datos  
- Los datasets originales se almacenan en `data/raw/`.  
- Se versionan con **DVC** para garantizar trazabilidad y reproducibilidad.  
- Cada cambio en los datos queda documentado y puede revertirse.  

---

### 2. Limpieza y preparación (`data/`)  
- Se ejecutan los scripts de limpieza y preprocesamiento.  
- Los resultados se guardan en `data/processed/` o `data/interim/`.  
- Esta etapa se define como la primera fase del pipeline en `dvc.yaml`.  

---

### 3. Ingeniería de características (`features/`)  
- Se generan nuevas variables a partir de los datos procesados.  
- Se escalan, transforman o codifican según la configuración del archivo YAML.  
- Los datasets enriquecidos se guardan nuevamente en `data/interim/`.  

---

### 4. Entrenamiento y evaluación del modelo (`modelling/` y `models/`)  
- Los modelos se entrenan usando los parámetros definidos en `configs/train.yaml`.  
- El entrenamiento se automatiza con **DVC** y se ejecuta mediante `main.py`.  
- Los artefactos generados (modelos entrenados, métricas) se guardan en `models/` y `reports/`.  
- Se registran métricas de desempeño como `accuracy`, `f1`, `r2`, etc.  

---

### 5. Visualización y análisis de resultados (`visualization/`)  
- Se generan gráficos automáticos para comparar resultados y métricas.  
- Apoyan la interpretación del rendimiento y la comunicación de hallazgos.  

---

### 6. Control de versiones y automatización  
- **Git** gestiona el control de versiones del código.  
- **DVC** controla los datos y los artefactos del modelo.  
- El pipeline puede reproducirse completo con un solo comando:  
  ```bash
  python src/proyecto_mlops/main.py


----
Además, el archivo **`main.py`** actúa como punto de entrada del pipeline,  
permitiendo ejecutar todas las etapas con el comando:  

```bash
python src/proyecto_mlops/main.py



