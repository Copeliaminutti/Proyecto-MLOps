import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modelling.train import ModelTrainer
import pandas as pd

@pytest.fixture
def trained_model():
    # Carga datos de prueba y entrena el modelo
    df = pd.read_csv("data/interim/features.csv")
    trainer = ModelTrainer(cfg_path="train.yaml")
    trainer.load_configuration()
    payload = trainer.train(df)
    return payload

def test_model_accuracy(trained_model):
    # Obtén el mejor modelo y datos de test
    best_model = trained_model["models"][trained_model["best_model"]]
    test_metrics = trained_model["test_metrics"][trained_model["best_model"]]
    # Por ejemplo, verifica que el R2 de test sea mayor a 0.5
    assert test_metrics["r2"] > 0.5

def test_model_rmse(trained_model):
    test_metrics = trained_model["test_metrics"][trained_model["best_model"]]
    # Verifica que el RMSE sea menor a un umbral
    assert test_metrics["rmse"] < 1000