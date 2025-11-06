import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import scipy.stats as stats
from src.config import load_config


@pytest.fixture(scope="module")
def features_dataset():
    """Carga los datos procesados de features.py (entrada a train.py)"""
    df = pd.read_csv("data/interim/features.csv")
    return df


@pytest.fixture(scope="module")
def feature_config():
    """Carga la configuración de features"""
    return load_config("features.yaml")


def test_feature_normality(features_dataset, feature_config):
    """Verifica que las features sigan una distribución normal usando Shapiro-Wilk test"""
    target = feature_config.get('target', 'usage_kwh').lower()
    
    # Obtener todas las columnas numéricas excepto el target
    numeric_cols = features_dataset.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [col for col in numeric_cols if col != target]
    
    for feature in feature_cols:
        _, p_value = stats.shapiro(features_dataset[feature])
        assert p_value < 0.05, f"{feature} fails normality test with p-value: {p_value}"


def test_sepal_length_between_species(features_dataset, feature_config):
    """Verifica diferencias significativas entre grupos usando ANOVA (f_oneway)"""
    target = feature_config.get('target', 'usage_kwh').lower()
    
    # Obtener todas las columnas numéricas excepto el target
    numeric_cols = features_dataset.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [col for col in numeric_cols if col != target]
    
    # Si hay al menos 3 features, comparar las primeras 3 como grupos
    # (Equivalente a comparar especies en el ejemplo de Iris)
    if len(feature_cols) >= 3:
        group1 = features_dataset[feature_cols[0]]
        group2 = features_dataset[feature_cols[1]]
        group3 = features_dataset[feature_cols[2]]
        
        f_value, p_value = stats.f_oneway(group1, group2, group3)
        assert p_value < 0.05  # Un P-value bajo indica diferencias significativas
