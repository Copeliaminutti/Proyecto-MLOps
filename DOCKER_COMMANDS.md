# Dockerfile para API de Predicción de Consumo Energético

## Construcción de la imagen
docker build -t energy-prediction-api:v1.0 .

## Ejecutar el contenedor
docker run -d -p 8000:8000 --name energy-api energy-prediction-api:v1.0

## Ver logs del contenedor
docker logs -f energy-api

## Detener el contenedor
docker stop energy-api

## Eliminar el contenedor
docker rm energy-api

## Probar la API
curl http://localhost:8000/

## Hacer una predicción
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "lagging_current_reactive_power_kvarh": 15.5,
    "leading_current_reactive_power_kvarh": 8.2,
    "co2_tco2": 0.05,
    "lagging_current_power_factor": 25.3,
    "leading_current_power_factor": 18.7,
    "nsm": 43200.0,
    "mixed_type_col": 1.0,
    "week_status": "Weekday",
    "day_of_week": "Monday",
    "load_type": "Medium_Load"
  }'
