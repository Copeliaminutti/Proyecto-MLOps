import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.features import build, run_features
from src.config import load_config


@pytest.fixture
def raw_data():
    """Carga los datos limpios que son la entrada de features.py"""
    df = pd.read_csv("data/processed/clean.csv")
    return df


@pytest.fixture
def feature_config():
    """Carga la configuración de features"""
    return load_config("features.yaml")


@pytest.fixture
def processed_features(raw_data, feature_config):
    """Genera las features procesadas"""
    return build(raw_data, feature_config)


def test_features_removes_date_column(processed_features):
    """Verifica que la columna 'date' sea removida después del procesamiento"""
    assert 'date' not in processed_features.columns, "La columna 'date' debe ser removida"


def test_features_has_no_missing_values_in_numeric_columns(processed_features, feature_config):
    """Verifica que no haya valores faltantes en columnas numéricas después de la imputación"""
    target = feature_config.get('target', 'usage_kwh').lower()
    numeric_cols = processed_features.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target]
    
    for col in numeric_cols:
        missing_count = processed_features[col].isna().sum()
        assert missing_count == 0, f"La columna numérica '{col}' tiene {missing_count} valores faltantes"


def test_features_target_has_no_missing_values(processed_features, feature_config):
    """Verifica que la columna target no tenga valores faltantes"""
    target = feature_config.get('target', 'usage_kwh').lower()
    if target in processed_features.columns:
        missing_count = processed_features[target].isna().sum()
        assert missing_count == 0, f"La columna target '{target}' tiene {missing_count} valores faltantes"


def test_features_shape_is_consistent(raw_data, processed_features):
    """Verifica que el número de filas se mantenga o disminuya (por eliminación de NaNs en target)"""
    assert len(processed_features) <= len(raw_data), \
        "El número de filas procesadas debe ser menor o igual al de entrada"
    assert len(processed_features) > 0, "No debe haber cero filas después del procesamiento"


def test_features_one_hot_encoding_applied(processed_features, feature_config):
    """Verifica que se aplique one-hot encoding si está configurado"""
    if feature_config.get('one_hot', False):
        # Verifica que no haya columnas de tipo object (categóricas sin codificar)
        target = feature_config.get('target', 'usage_kwh').lower()
        non_numeric_cols = processed_features.select_dtypes(include=['object', 'category']).columns
        non_numeric_cols = [col for col in non_numeric_cols if col != target]
        
        assert len(non_numeric_cols) == 0, \
            f"Después de one-hot encoding no debe haber columnas categóricas: {list(non_numeric_cols)}"


def test_features_all_numeric_after_processing(processed_features, feature_config):
    """Verifica que todas las columnas (excepto target si es categórico) sean numéricas"""
    target = feature_config.get('target', 'usage_kwh').lower()
    non_target_cols = [col for col in processed_features.columns if col != target]
    
    for col in non_target_cols:
        assert pd.api.types.is_numeric_dtype(processed_features[col]), \
            f"La columna '{col}' debe ser numérica después del procesamiento"


def test_features_no_infinite_values(processed_features):
    """Verifica que no haya valores infinitos en las features numéricas"""
    numeric_cols = processed_features.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        infinite_count = np.isinf(processed_features[col]).sum()
        assert infinite_count == 0, f"La columna '{col}' tiene {infinite_count} valores infinitos"


def test_features_output_file_created():
    """Verifica que run_features cree el archivo de salida correctamente"""
    input_path = "data/processed/clean.csv"
    output_path = "data/interim/features.csv"
    
    # Ejecuta el procesamiento
    df_result = run_features(input_path, output_path)
    
    # Verifica que el archivo exista
    import os
    assert os.path.exists(output_path), f"El archivo de salida {output_path} no fue creado"
    
    # Verifica que el DataFrame retornado no esté vacío
    assert len(df_result) > 0, "El DataFrame de features no debe estar vacío"
    assert len(df_result.columns) > 1, "El DataFrame debe tener más de una columna"


def test_features_preserves_target_column(processed_features, feature_config):
    """Verifica que la columna target se preserve en el dataset procesado"""
    target = feature_config.get('target', 'usage_kwh').lower()
    assert target in processed_features.columns, f"La columna target '{target}' debe estar presente"


def test_features_imputation_reduces_nulls(raw_data, processed_features):
    """Verifica que la imputación reduzca el número de valores nulos"""
    raw_nulls = raw_data.isna().sum().sum()
    processed_nulls = processed_features.isna().sum().sum()
    
    assert processed_nulls <= raw_nulls, \
        f"La imputación debe reducir los nulos (antes: {raw_nulls}, después: {processed_nulls})"
