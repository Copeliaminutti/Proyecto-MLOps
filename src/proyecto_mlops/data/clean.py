# -*- coding: utf-8 -*-
"""Limpieza de datos con Programación Orientada a Objetos.

En esta etapa aplicamos POO para organizar la lógica de limpieza de datos.
El objetivo es pasar de un código funcional a una estructura basada en clases,
manteniendo compatibilidad con el pipeline de DVC.

Este módulo:
- Lee los datos crudos desde data/raw/
- Aplica reglas de limpieza (valores nulos, duplicados, outliers)
- Guarda los datos limpios en data/processed/
"""

from pathlib import Path
import pandas as pd
import yaml


class DataCleaner:
    """Clase que encapsula el proceso de limpieza de datos.

    Esta clase se encarga de:
    - Cargar los datos
    - Aplicar reglas de limpieza
    - Guardar los resultados
    """

    def __init__(self, cfg_path: str):
        self.cfg_path = Path(cfg_path)
        self.cfg = self._load_cfg(self.cfg_path)
        self.df = None

    def _load_cfg(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load(self, raw_csv_path: str):
        raw_csv_path = Path(raw_csv_path)
        if not raw_csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de entrada: {raw_csv_path}")
        print(f"Cargando datos desde {raw_csv_path} ...")
        self.df = pd.read_csv(raw_csv_path)
        print(f"Filas: {self.df.shape[0]}, Columnas: {self.df.shape[1]}")
        return self

    def drop_missing(self):
        if self.cfg.get("drop_na", True):
            before = len(self.df)
            self.df = self.df.dropna()
            after = len(self.df)
            print(f"Filas eliminadas por valores nulos: {before - after}")
        return self

    def drop_duplicates(self):
        if self.cfg.get("duplicates", True):
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            after = len(self.df)
            print(f"Filas duplicadas eliminadas: {before - after}")
        return self

    def remove_outliers(self):
        rules = self.cfg.get("outliers", {})
        if rules.get("method") != "iqr":
            print("Eliminación de outliers desactivada o método no configurado.")
            return self

        factor = float(rules.get("factor", 1.5))
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        before = len(self.df)
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            low = q1 - factor * iqr
            high = q3 + factor * iqr
            mask = (self.df[col] >= low) & (self.df[col] <= high)
            self.df = self.df[mask]

        after = len(self.df)
        print(f"Filas eliminadas por outliers (IQR): {before - after}")
        return self

    def save(self, output_csv_path: str):
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_csv_path, index=False)
        print(f"Archivo limpio guardado en {output_csv_path} ({len(self.df)} filas)")
        return self

    def run(self, raw_csv_path: str, output_csv_path: str):
        (self
            .load(raw_csv_path)
            .drop_missing()
            .drop_duplicates()
            .remove_outliers()
            .save(output_csv_path)
        )
        return self


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="configs/eda.yaml",
                        help="Ruta del archivo de configuración de limpieza (YAML).")
    parser.add_argument("--input", type=str, default="data/raw/steel_energy_clean_fullgrid.csv",
                        help="Ruta del archivo CSV de entrada.")
    parser.add_argument("--output", type=str, default="data/processed/clean.csv",
                        help="Ruta de salida del archivo limpio.")
    args = parser.parse_args()

    cleaner = DataCleaner(cfg_path=args.cfg)
    cleaner.run(raw_csv_path=args.input, output_csv_path=args.output)


if __name__ == "__main__":
    main()
