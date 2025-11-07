import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.modelling.train import ModelTrainer, ModelSpecLoader
from src.config import load_config


@pytest.fixture
def train_config():
    """Carga la configuración de entrenamiento"""
    return load_config("train.yaml")


@pytest.fixture
def model_trainer(train_config):
    """Crea una instancia del ModelTrainer con configuración cargada"""
    trainer = ModelTrainer(cfg_path="train.yaml")
    trainer.load_configuration()
    return trainer


@pytest.fixture
def trained_models(model_trainer):
    """Entrena los modelos y retorna el payload completo"""
    df = pd.read_csv("data/interim/features.csv")
    payload = model_trainer.train(df)
    return payload


# Configuraciones permitidas para cada tipo de modelo
ALLOWED_MODEL_CONFIGS = {
    'ridge': {
        'alpha': [0.1, 0.5, 1.0, 10.0],
        'fit_intercept': [True, False],
    },
    'linear_regression': {
        'fit_intercept': [True, False],
        'n_jobs': [-1, 1, 2, 4],
    },
    'knn': {
        'n_neighbors': range(3, 101),  # Entre 3 y 100
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    },
    'decision_tree': {
        'max_depth': [None, 3, 5, 10, 15, 20],
        'min_samples_split': range(2, 21),  # Entre 2 y 20
        'min_samples_leaf': range(1, 11),  # Entre 1 y 10
    },
    'random_forest': {
        'n_estimators': range(10, 201),  # Entre 10 y 200
        'max_depth': [None, 3, 4, 5, 10, 15, 20],
    },
    'gradient_boosting': {
        'n_estimators': range(10, 201),
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': range(2, 11),
    },
}


def test_all_configured_models_exist_in_registry(train_config):
    """Verifica que todos los modelos configurados estén registrados en MODEL_REGISTRY"""
    from src.modelling.train import MODEL_REGISTRY
    
    model_specs = ModelSpecLoader.load_from_config(train_config)
    
    for spec in model_specs:
        assert spec.model_type in MODEL_REGISTRY, \
            f"El tipo de modelo '{spec.model_type}' no está registrado en MODEL_REGISTRY"


def test_model_hyperparameters_are_valid(train_config):
    """Verifica que los hiperparámetros configurados sean válidos para cada tipo de modelo"""
    model_specs = ModelSpecLoader.load_from_config(train_config)
    
    for spec in model_specs:
        model_type = spec.model_type
        params = spec.params
        
        # Verificar que el tipo de modelo tenga configuraciones permitidas definidas
        if model_type in ALLOWED_MODEL_CONFIGS:
            allowed_config = ALLOWED_MODEL_CONFIGS[model_type]
            
            for param_name, param_value in params.items():
                # Skip parameters that are not in the allowed config (permite flexibilidad)
                if param_name not in allowed_config:
                    continue
                
                allowed_values = allowed_config[param_name]
                
                # Si es un rango, verificar que el valor esté en el rango
                if isinstance(allowed_values, range):
                    assert param_value in allowed_values, \
                        f"El parámetro '{param_name}' del modelo '{spec.name}' ({param_value}) no está en el rango permitido {allowed_values}"
                # Si es una lista, verificar que el valor esté en la lista
                elif isinstance(allowed_values, list):
                    assert param_value in allowed_values, \
                        f"El parámetro '{param_name}' del modelo '{spec.name}' ({param_value}) no está en los valores permitidos {allowed_values}"


def test_train_config_has_required_paths(train_config):
    """Verifica que la configuración tenga las rutas requeridas"""
    assert 'paths' in train_config, "La configuración debe tener la sección 'paths'"
    assert 'train' in train_config['paths'], "La configuración debe tener 'paths.train'"
    
    # Verificar que el archivo de entrenamiento exista
    train_path = train_config['paths']['train']
    assert os.path.exists(train_path), f"El archivo de entrenamiento {train_path} no existe"


def test_train_config_has_target_column(train_config):
    """Verifica que la configuración especifique una columna target"""
    assert 'target' in train_config, "La configuración debe especificar una columna 'target'"
    assert isinstance(train_config['target'], str), "El target debe ser un string"
    assert len(train_config['target']) > 0, "El target no puede estar vacío"


def test_train_config_has_valid_split_sizes(train_config):
    """Verifica que los tamaños de split sean válidos"""
    test_size = train_config.get('test_size', 0.2)
    val_size = train_config.get('val_size', 0.2)
    
    assert 0 < test_size < 1, f"test_size debe estar entre 0 y 1, pero es {test_size}"
    assert 0 < val_size < 1, f"val_size debe estar entre 0 y 1, pero es {val_size}"
    assert test_size + val_size < 1, \
        f"La suma de test_size ({test_size}) y val_size ({val_size}) debe ser menor a 1"


def test_train_config_has_valid_cv_parameters(train_config):
    """Verifica que los parámetros de cross-validation sean válidos"""
    cv_n_splits = train_config.get('cv_n_splits', 5)
    cv_n_repeats = train_config.get('cv_n_repeats', 3)
    
    assert cv_n_splits >= 2, f"cv_n_splits debe ser al menos 2, pero es {cv_n_splits}"
    assert cv_n_splits <= 10, f"cv_n_splits no debe exceder 10, pero es {cv_n_splits}"
    assert cv_n_repeats >= 1, f"cv_n_repeats debe ser al menos 1, pero es {cv_n_repeats}"
    assert cv_n_repeats <= 10, f"cv_n_repeats no debe exceder 10, pero es {cv_n_repeats}"


def test_trained_models_have_valid_structure(trained_models):
    """Verifica que el payload de modelos entrenados tenga la estructura correcta"""
    assert 'models' in trained_models, "El payload debe contener 'models'"
    assert 'best_model' in trained_models, "El payload debe contener 'best_model'"
    assert 'test_metrics' in trained_models, "El payload debe contener 'test_metrics'"
    
    # Verificar que best_model sea uno de los modelos entrenados
    assert trained_models['best_model'] in trained_models['models'], \
        "El best_model debe estar en la lista de modelos entrenados"


def test_all_trained_models_have_metrics(trained_models):
    """Verifica que todos los modelos entrenados tengan métricas"""
    for model_name in trained_models['models'].keys():
        assert model_name in trained_models['test_metrics'], \
            f"El modelo '{model_name}' debe tener métricas de test"
        
        test_metrics = trained_models['test_metrics'][model_name]
        assert 'rmse' in test_metrics, f"El modelo '{model_name}' debe tener métrica RMSE"
        assert 'r2' in test_metrics, f"El modelo '{model_name}' debe tener métrica R2"
