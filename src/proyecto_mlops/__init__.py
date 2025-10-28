from proyecto_mlops.features import run_features
from proyecto_mlops.modelling.train import run_train

def main_pipeline():
	# 1. Ejecutar features y obtener el DataFrame procesado
	df_feat = run_features(input_path="src/data/processed/clean.csv")
	run_train(df_feat)

if __name__ == "__main__":
	main_pipeline()
