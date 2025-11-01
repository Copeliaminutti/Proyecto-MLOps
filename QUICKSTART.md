# 🚀 Guía Rápida - Proyecto MLOps

## ⚡ Inicio Rápido

### **Opción 1: Script automatizado (Recomendado)**

```bash
# Ver opciones disponibles
./run_pipeline.sh help

# Ejecutar entrenamiento
./run_pipeline.sh train

# Levantar MLflow UI (después del entrenamiento)
./run_pipeline.sh mlflow

# Hacer todo en un comando (entrena + abre MLflow)
./run_pipeline.sh full
```

### **Opción 2: Comandos manuales**

```bash
# 1. Ejecutar pipeline de entrenamiento
dvc repro

# 2. Ver resultados en MLflow UI
mlflow ui --port 5000
# Luego abre: http://localhost:5000
```

### **Opción 3: Con Python**

```bash
# Ejecuta el pipeline usando main.py
python src/proyecto_mlops/main.py

# En otra terminal, levanta MLflow
python src/proyecto_mlops/mlflow_server.py
```

---

## 📊 ¿Qué hace cada comando?

| Comando | Descripción |
|---------|-------------|
| `./run_pipeline.sh train` | Ejecuta features → train → evaluate con DVC |
| `./run_pipeline.sh mlflow` | Abre interfaz web de MLflow en puerto 5000 |
| `./run_pipeline.sh full` | Entrena y pregunta si abrir MLflow |
| `./run_pipeline.sh clean` | Limpia outputs (artifacts, reports, models) |

---

## 🔄 Flujo de trabajo típico

### **Durante desarrollo:**

```bash
# Terminal 1: Levantar MLflow UI (ver experimentos en tiempo real)
./run_pipeline.sh mlflow

# Terminal 2: Entrenar modelos
./run_pipeline.sh train
```

### **Para experimentar:**

```bash
# 1. Modificar configs/train.yaml (cambiar hiperparámetros)
# 2. Ejecutar
./run_pipeline.sh train

# 3. Ver resultados en MLflow UI
./run_pipeline.sh mlflow
```

### **Para producción:**

```bash
# Ejecutar pipeline completo
dvc repro

# Modelo guardado en:
# src/proyecto_mlops/models/best_model.pkl
```

---

## 📁 Archivos generados

Después de ejecutar el pipeline:

```
├── src/proyecto_mlops/models/
│   └── best_model.pkl          # ← Mejor modelo entrenado
├── reports/
│   ├── metrics.json            # ← Métricas del mejor modelo
│   └── figures/                # ← Gráficos de evaluación
├── artifacts/eval/
│   ├── splits/                 # ← Train/val/test splits
│   └── metrics.json            # ← Métricas detalladas
└── mlruns/                     # ← Experimentos de MLflow
```

---

## 🎯 Ver experimentos en MLflow

1. Abre http://localhost:5000
2. Selecciona el experimento
3. Compara runs
4. Filtra por métricas
5. Descarga modelos

---

## 🐛 Troubleshooting

### Error: "dvc command not found"
```bash
pip install dvc
```

### Error: "mlflow command not found"
```bash
pip install mlflow
```

### Puerto 5000 ocupado
```bash
# Usar otro puerto
mlflow ui --port 5001
```

### Pipeline falla
```bash
# Ver logs detallados
dvc repro -v

# Limpiar y volver a correr
./run_pipeline.sh clean
./run_pipeline.sh train
```

---

## 📚 Más información

- **DVC**: Versionado de datos y pipeline
- **MLflow**: Tracking de experimentos y modelos
- **Pipeline stages**: features → train → evaluate

¿Preguntas? Revisa la documentación en `docs/`
