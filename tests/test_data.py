import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.config import load_config


@pytest.fixture
def model_input_data():
    """Carga los datos que son la entrada al modelo (output de features.py)"""
    df = pd.read_csv("data/interim/features.csv")
    return df


@pytest.fixture
def feature_config():
    """Carga la configuración de features"""
    return load_config("features.yaml")


@pytest.fixture
def data_schema(model_input_data, feature_config):
    """Define el esquema esperado para los datos basándose en estadísticas del dataset"""
    target = feature_config.get('target', 'usage_kwh').lower()
    schema = {}
    
    # Para cada columna numérica, definir rangos esperados
    numeric_cols = model_input_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == target:
            continue
        
        # Calcular percentiles para definir rangos razonables
        q1 = model_input_data[col].quantile(0.01)
        q99 = model_input_data[col].quantile(0.99)
        
        schema[col] = {
            'range': {
                'min': q1 - (q99 - q1) * 1,  # Margen de error del 50%
                'max': q99 + (q99 - q1) * 0.5
            },
            'dtype': model_input_data[col].dtype
        }
    
    return schema


def test_input_data_has_no_missing_values(model_input_data):
    """Verifica que no haya valores faltantes en los datos de entrada al modelo"""
    missing_count = model_input_data.isna().sum().sum()
    assert missing_count == 0, f"Los datos de entrada tienen {missing_count} valores faltantes"


def test_input_data_has_target_column(model_input_data, feature_config):
    """Verifica que la columna target esté presente"""
    target = feature_config.get('target', 'usage_kwh').lower()
    assert target in model_input_data.columns, f"La columna target '{target}' debe estar presente"


def test_input_data_target_is_numeric(model_input_data, feature_config):
    """Verifica que la columna target sea numérica"""
    target = feature_config.get('target', 'usage_kwh').lower()
    assert pd.api.types.is_numeric_dtype(model_input_data[target]), \
        f"La columna target '{target}' debe ser numérica"


def test_input_data_ranges(model_input_data, data_schema):
    """Verifica que los valores de las features estén dentro de rangos esperados"""
    max_values = model_input_data.max()
    min_values = model_input_data.min()
    
    for feature in data_schema.keys():
        if feature not in model_input_data.columns:
            continue
            
        assert max_values[feature] <= data_schema[feature]['range']['max'], \
            f"El valor máximo de '{feature}' ({max_values[feature]}) excede el rango esperado ({data_schema[feature]['range']['max']})"
        
        assert min_values[feature] >= data_schema[feature]['range']['min'], \
            f"El valor mínimo de '{feature}' ({min_values[feature]}) está por debajo del rango esperado ({data_schema[feature]['range']['min']})"


def test_input_data_types(model_input_data, feature_config):
    """Verifica que todas las columnas (excepto target) sean numéricas"""
    target = feature_config.get('target', 'usage_kwh').lower()
    data_types = model_input_data.dtypes
    
    for column in model_input_data.columns:
        if column == target:
            continue
        assert pd.api.types.is_numeric_dtype(data_types[column]), \
            f"La columna '{column}' debe ser numérica, pero es {data_types[column]}"


def test_input_data_has_sufficient_rows(model_input_data):
    """Verifica que haya suficientes filas para entrenar un modelo"""
    min_rows = 100  # Mínimo razonable para entrenamiento
    assert len(model_input_data) >= min_rows, \
        f"El dataset tiene solo {len(model_input_data)} filas, se requieren al menos {min_rows}"


def test_input_data_has_no_infinite_values(model_input_data):
    """Verifica que no haya valores infinitos en los datos"""
    numeric_cols = model_input_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        infinite_count = np.isinf(model_input_data[col]).sum()
        assert infinite_count == 0, f"La columna '{col}' tiene {infinite_count} valores infinitos"


def test_input_data_has_variance(model_input_data, feature_config):
    """Verifica que las features tengan varianza (no sean constantes)"""
    target = feature_config.get('target', 'usage_kwh').lower()
    numeric_cols = model_input_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == target:
            continue
        variance = model_input_data[col].var()
        assert variance > 0, f"La columna '{col}' tiene varianza cero (es constante)"


def test_input_data_no_duplicate_rows(model_input_data):
    """Verifica que no haya filas duplicadas en los datos"""
    duplicates = model_input_data.duplicated().sum()
    duplicate_pct = (duplicates / len(model_input_data)) * 100
    
    # Permitimos hasta 1% de duplicados (puede ser normal en algunos casos)
    assert duplicate_pct < 1.0, \
        f"Se encontraron {duplicates} filas duplicadas ({duplicate_pct:.2f}%)"


def test_input_data_target_has_positive_values(model_input_data, feature_config):
    """Verifica que el target tenga valores positivos (para consumo de energía)"""
    target = feature_config.get('target', 'usage_kwh').lower()
    min_target = model_input_data[target].min()
    assert min_target >= 0, f"El target '{target}' tiene valores negativos (mínimo: {min_target})"


def test_input_data_has_multiple_features(model_input_data, feature_config):
    """Verifica que haya suficientes features para entrenar un modelo"""
    target = feature_config.get('target', 'usage_kwh').lower()
    feature_count = len(model_input_data.columns) - 1  # Excluir target
    
    min_features = 3  # Mínimo razonable
    assert feature_count >= min_features, \
        f"El dataset tiene solo {feature_count} features, se requieren al menos {min_features}"


def test_input_data_no_extreme_outliers(model_input_data, feature_config):
    """Verifica que no haya outliers extremos (más allá de 5 desviaciones estándar)"""
    target = feature_config.get('target', 'usage_kwh').lower()
    numeric_cols = model_input_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == target:
            continue
        
        mean = model_input_data[col].mean()
        std = model_input_data[col].std()
        
        if std == 0:  # Evitar división por cero
            continue
        
        z_scores = np.abs((model_input_data[col] - mean) / std)
        extreme_outliers = (z_scores > 5).sum()
        outlier_pct = (extreme_outliers / len(model_input_data)) * 100
        
        # Permitimos hasta 0.1% de outliers extremos
        assert outlier_pct < 0.1, \
            f"La columna '{col}' tiene {extreme_outliers} outliers extremos ({outlier_pct:.2f}%)"


