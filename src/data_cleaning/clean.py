#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Limpieza de datos del Steel Industry Energy Consumption Dataset.

Este script implementa un pipeline completo de limpieza de datos utilizando
Programación Orientada a Objetos, basado en el análisis del notebook
Fase_1_Limpieza_EDA_git_1.ipynb.


Funcionalidades principales:
- Normalización de nombres de columnas a snake_case
- Conversión de fecha a formato datetime coherente (dd/mm/yyyy HH:MM)
- Conversión de columnas numéricas a tipo float
- Limpieza y validación de categóricas (day_of_week, week_status, load_type)
- Eliminación de filas sin fecha y duplicados por timestamp
- Recálculo de NSM (segundos desde medianoche) a múltiplos de 900 (15 min)
- Aplicación de reglas físicas (no negatividad, rangos válidos)
- Imputación jerárquica de faltantes (load_type+day → day → global)
- Imputación de CO2 usando factor de emisión
- Capping de outliers por IQR
- Manejo de columnas mixtas (separación numérica/categórica)
- Imputación de load_type NA por moda del día de la semana
- Generación opcional de grilla temporal completa (15 minutos)
- Creación de variable numérica day_status_num (Weekday=0, Weekend=1)
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class SteelEnergyDataCleaner:
    """
    Clase para la limpieza de datos de consumo energético en la industria del acero.
    
    Attributes:
        df: DataFrame principal con los datos crudos
        df_clean: DataFrame con los datos limpios
        dup_strategy: Estrategia para manejar duplicados ('first' o 'aggregate')
    """
    
    def __init__(self, dup_strategy: str = "first"):
        """
        Inicializa el limpiador de datos.
        
        Args:
            dup_strategy: Estrategia para duplicados. 'first' mantiene la primera 
                         aparición, 'aggregate' promedia numéricas y toma el primero
                         para categóricas.
        """
        self.df = None
        self.df_clean = None
        self.dup_strategy = dup_strategy
        self.stats = {
            "n_original": 0,
            "n_na_date": 0,
            "n_duplicates": 0,
            "n_final": 0
        }
    
    @staticmethod
    def normalize_colname(name: str) -> str:
        """
        Normaliza nombres de columnas a snake_case.
        
        Args:
            name: Nombre de columna original
            
        Returns:
            Nombre normalizado en snake_case
        """
        # Quitar acentos
        s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
        # Separar CamelCase
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        s = s.strip().lower()
        # Reemplazar separadores por guión bajo
        s = re.sub(r"[^\w]+", "_", s)
        s = re.sub(r"__+", "_", s).strip("_")
        return s
    
    def load_data(self, filepath: str) -> 'SteelEnergyDataCleaner':
        """
        Carga los datos desde un archivo CSV.
        
        Args:
            filepath: Ruta al archivo CSV de entrada
            
        Returns:
            self para encadenamiento de métodos
        """
        na_values = ["", " ", "nan", "NaN", "NAN", "null", "NULL", "N/A", "NA"]
        
        self.df = pd.read_csv(
            filepath,
            na_values=na_values,
            keep_default_na=True,
            parse_dates=["date"],
            dayfirst=True,
            encoding="utf-8-sig"
        )
        
        print(f"Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
        return self
    
    def normalize_columns(self) -> 'SteelEnergyDataCleaner':
        """
        Normaliza los nombres de columnas y maneja duplicados.
        
        Returns:
            self para encadenamiento de métodos
        """
        colmap = {c: self.normalize_colname(c) for c in self.df.columns}
        seen = {}
        new_cols = []
        
        for c in self.df.columns:
            nc = colmap[c]
            if nc in seen:
                seen[nc] += 1
                nc = f"{nc}_{seen[nc]}"
            else:
                seen[nc] = 0
            new_cols.append(nc)
        
        self.df.columns = new_cols
        print(f"Columnas normalizadas: {len(new_cols)} columnas")
        return self
    
    def fix_column_names(self) -> 'SteelEnergyDataCleaner':
        """
        Corrige nombres de columnas que quedaron mal separados.
        
        Returns:
            self para encadenamiento de métodos
        """
        rename_fix = {
            "usage_k_wh": "usage_kwh",
            "co2_t_co2": "co2_tco2",
            "lagging_current_reactive_power_k_varh": "lagging_current_reactive_power_kvarh",
            "leading_current_reactive_power_k_varh": "leading_current_reactive_power_kvarh",
        }
        self.df.rename(columns=rename_fix, inplace=True)
        return self
    
    def convert_numeric_columns(self) -> 'SteelEnergyDataCleaner':
        """
        Convierte columnas numéricas a tipo float.
        
        Returns:
            self para encadenamiento de métodos
        """
        # Forzar formato de fecha
        self.df["date"] = pd.to_datetime(self.df["date"], format="%d/%m/%Y %H:%M", errors="coerce")
        
        num_cols = [
            "usage_kwh",
            "lagging_current_reactive_power_kvarh",
            "leading_current_reactive_power_kvarh",
            "co2_tco2",
            "lagging_current_power_factor",
            "leading_current_power_factor",
            "nsm",
        ]
        
        for c in num_cols:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        
        print("Columnas numéricas convertidas")
        return self
    
    def clean_categorical_columns(self) -> 'SteelEnergyDataCleaner':
        """
        Limpia y normaliza columnas categóricas (day_of_week, week_status, load_type).
        
        Returns:
            self para encadenamiento de métodos
        """
        self.df_clean = self.df.copy()
        
        # Corregir nombres por si acaso
        rename_fix = {
            "usage_k_wh": "usage_kwh",
            "co2_t_co2": "co2_tco2",
            "lagging_current_reactive_power_k_varh": "lagging_current_reactive_power_kvarh",
            "leading_current_reactive_power_k_varh": "leading_current_reactive_power_kvarh",
        }
        self.df_clean.rename(
            columns={k: v for k, v in rename_fix.items() if k in self.df_clean.columns},
            inplace=True
        )
        
        # day_of_week: derivar desde la fecha
        self.df_clean["day_of_week_raw"] = self.df_clean.get(
            "day_of_week", pd.Series(index=self.df_clean.index, dtype="string")
        )
        self.df_clean["day_of_week"] = self.df_clean["date"].dt.day_name().astype("string")
        
        # week_status: derivar desde day_of_week
        def _week_from_day(d):
            if pd.isna(d):
                return pd.NA
            return "Weekend" if d in ("Saturday", "Sunday") else "Weekday"
        
        self.df_clean["week_status_raw"] = self.df_clean.get(
            "week_status", pd.Series(index=self.df_clean.index, dtype="string")
        )
        self.df_clean["week_status"] = self.df_clean["day_of_week"].map(_week_from_day).astype("string")
        
        # load_type: normalizar variantes
        def _norm_load_type(x):
            if pd.isna(x):
                return pd.NA
            s = str(x).strip().upper().replace(" ", "").replace("-", "_")
            s = s.replace("__", "_")
            if "LIGHT" in s:
                return "Light_Load"
            if "MEDIUM" in s:
                return "Medium_Load"
            if "MAX" in s:
                return "Maximum_Load"
            return pd.NA
        
        self.df_clean["load_type_raw"] = self.df_clean.get(
            "load_type", pd.Series(index=self.df_clean.index, dtype="string")
        )
        self.df_clean["load_type"] = self.df_clean["load_type_raw"].map(_norm_load_type).astype("string")
        
        # Convertir a categorías ordenadas
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        self.df_clean["day_of_week"] = pd.Categorical(
            self.df_clean["day_of_week"], categories=dow_order, ordered=True
        )
        self.df_clean["week_status"] = pd.Categorical(
            self.df_clean["week_status"], categories=["Weekday", "Weekend"], ordered=True
        )
        self.df_clean["load_type"] = pd.Categorical(
            self.df_clean["load_type"],
            categories=["Light_Load", "Medium_Load", "Maximum_Load"],
            ordered=True
        )
        
        print("Columnas categóricas limpiadas y normalizadas")
        return self
    
    def remove_missing_dates_and_duplicates(self) -> 'SteelEnergyDataCleaner':
        """
        Elimina filas sin fecha y maneja duplicados por timestamp.
        
        Returns:
            self para encadenamiento de métodos
        """
        self.stats["n_original"] = len(self.df_clean)
        self.stats["n_na_date"] = int(self.df_clean["date"].isna().sum())
        
        # Eliminar filas sin fecha
        self.df_clean = self.df_clean.loc[self.df_clean["date"].notna()].copy()
        
        # Manejar duplicados
        self.df_clean = self.df_clean.sort_values("date")
        
        if self.dup_strategy == "first":
            self.stats["n_duplicates"] = int(
                self.df_clean.duplicated(subset=["date"], keep="first").sum()
            )
            self.df_clean = self.df_clean.drop_duplicates(subset=["date"], keep="first")
        else:
            # Estrategia aggregate: media para numéricas, first para categóricas
            num_cols = self.df_clean.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = self.df_clean.select_dtypes(
                exclude=[np.number, "datetime64[ns]"]
            ).columns.tolist()
            agg = {**{c: "mean" for c in num_cols}, **{c: "first" for c in cat_cols}}
            self.stats["n_duplicates"] = int(
                self.df_clean.duplicated(subset=["date"], keep=False).sum()
            )
            self.df_clean = self.df_clean.groupby("date", as_index=False).agg(agg)
        
        print(f"Filas sin fecha eliminadas: {self.stats['n_na_date']}")
        print(f"Duplicados removidos: {self.stats['n_duplicates']}")
        return self
    
    def recalculate_nsm(self) -> 'SteelEnergyDataCleaner':
        """
        Recalcula NSM (segundos desde medianoche) coherente con la fecha.
        
        Returns:
            self para encadenamiento de métodos
        """
        nsm_calc = (
            self.df_clean["date"].dt.hour * 3600
            + self.df_clean["date"].dt.minute * 60
            + self.df_clean["date"].dt.second
        )
        # Redondear a múltiplos de 900 (15 minutos)
        self.df_clean["nsm"] = ((nsm_calc / 900).round() * 900).astype("Int64")
        
        print("NSM recalculado (múltiplos de 900)")
        return self
    
    def apply_physical_rules(self) -> 'SteelEnergyDataCleaner':
        """
        Aplica reglas físicas: no negatividad y rangos válidos.
        
        Returns:
            self para encadenamiento de métodos
        """
        num_cols_base = [
            "usage_kwh",
            "lagging_current_reactive_power_kvarh",
            "leading_current_reactive_power_kvarh",
            "co2_tco2",
            "lagging_current_power_factor",
            "leading_current_power_factor",
            "nsm",
        ]
        
        for c in num_cols_base:
            if c in self.df_clean.columns:
                self.df_clean[c] = pd.to_numeric(self.df_clean[c], errors="coerce")
        
        # No negativos para energías y CO2
        for c in [
            "usage_kwh",
            "lagging_current_reactive_power_kvarh",
            "leading_current_reactive_power_kvarh",
            "co2_tco2",
        ]:
            if c in self.df_clean.columns:
                self.df_clean.loc[self.df_clean[c] < 0, c] = np.nan
        
        # Factores de potencia en [0, 100]
        for c in ["lagging_current_power_factor", "leading_current_power_factor"]:
            if c in self.df_clean.columns:
                self.df_clean.loc[
                    (self.df_clean[c] < 0) | (self.df_clean[c] > 100), c
                ] = np.nan
        
        print("Reglas físicas aplicadas (no negatividad, rangos válidos)")
        return self
    
    def impute_missing_values(self) -> 'SteelEnergyDataCleaner':
        """
        Imputa valores faltantes usando jerarquía (load_type, day_of_week) -> day_of_week -> global.
        
        Returns:
            self para encadenamiento de métodos
        """
        def impute_numeric_col(df_, col):
            if col not in df_.columns:
                return df_
            s = df_[col]
            if s.isna().sum() == 0:
                return df_
            
            # Jerarquía de imputación
            med1 = df_.groupby(["load_type", "day_of_week"], observed=True)[col].transform("median")
            med2 = df_.groupby(["day_of_week"], observed=True)[col].transform("median")
            medg = s.median()
            df_[col] = s.fillna(med1).fillna(med2).fillna(medg)
            return df_
        
        # Imputar CO2 usando factor de emisión
        if "co2_tco2" in self.df_clean.columns and "usage_kwh" in self.df_clean.columns:
            mask_pair = (self.df_clean["co2_tco2"] > 0) & (self.df_clean["usage_kwh"] > 0)
            if mask_pair.any():
                emission_factor = (
                    self.df_clean.loc[mask_pair, "co2_tco2"]
                    / self.df_clean.loc[mask_pair, "usage_kwh"]
                ).median()
                mask_impute = self.df_clean["co2_tco2"].isna() & self.df_clean["usage_kwh"].notna()
                self.df_clean.loc[mask_impute, "co2_tco2"] = (
                    self.df_clean.loc[mask_impute, "usage_kwh"] * emission_factor
                )
        
        # Imputar todas las numéricas relevantes
        for col in [
            "usage_kwh",
            "lagging_current_reactive_power_kvarh",
            "leading_current_reactive_power_kvarh",
            "co2_tco2",
            "lagging_current_power_factor",
            "leading_current_power_factor",
        ]:
            self.df_clean = impute_numeric_col(self.df_clean, col)
        
        print("Valores faltantes imputados (jerarquía)")
        return self
    
    def cap_outliers(self, k: float = 1.5) -> 'SteelEnergyDataCleaner':
        """
        Aplica capping de outliers usando IQR.
        
        Args:
            k: Factor multiplicador del IQR para definir límites
            
        Returns:
            self para encadenamiento de métodos
        """
        def iqr_cap(s, k_factor):
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            low = q1 - k_factor * iqr
            high = q3 + k_factor * iqr
            return s.clip(lower=low, upper=high)
        
        for c in [
            "usage_kwh",
            "lagging_current_reactive_power_kvarh",
            "leading_current_reactive_power_kvarh",
            "co2_tco2",
        ]:
            if c in self.df_clean.columns:
                self.df_clean[c] = iqr_cap(self.df_clean[c], k)
        
        print(f"Outliers limitados (IQR con k={k})")
        return self
    
    def handle_mixed_columns(self) -> 'SteelEnergyDataCleaner':
        """
        Separa columnas mixtas en parte numérica y etiqueta.
        
        Returns:
            self para encadenamiento de métodos
        """
        if "mixed_type_col" in self.df_clean.columns:
            num_parsed = pd.to_numeric(self.df_clean["mixed_type_col"], errors="coerce")
            self.df_clean["mixed_type_col_num"] = num_parsed
            self.df_clean["mixed_type_col_label"] = (
                self.df_clean["mixed_type_col"]
                .where(num_parsed.isna(), other=pd.NA)
                .astype("string")
            )
            print("Columnas mixtas separadas")
        
        return self
    
    def impute_load_type_na(self) -> 'SteelEnergyDataCleaner':
        """
        Imputa valores NA en load_type usando la moda por día de la semana.
        
        Returns:
            self para encadenamiento de métodos
        """
        mode_by_dow = (
            self.df_clean.dropna(subset=["load_type"])
            .groupby("day_of_week", observed=True)["load_type"]
            .agg(lambda x: x.mode().iat[0] if not x.mode().empty else pd.NA)
        )
        self.df_clean.loc[self.df_clean["load_type"].isna(), "load_type"] = (
            self.df_clean.loc[self.df_clean["load_type"].isna(), "day_of_week"].map(mode_by_dow)
        )
        
        na_remaining = self.df_clean["load_type"].isna().sum()
        print(f"load_type NA imputados. Restantes: {na_remaining}")
        return self
    
    def create_day_status_num(self) -> 'SteelEnergyDataCleaner':
        """
        Crea variable numérica day_status_num a partir de week_status.
        Weekday=0, Weekend=1
        
        Returns:
            self para encadenamiento de métodos
        """
        status_map = {"Weekday": 0, "Weekend": 1}
        self.df_clean["day_status_num"] = (
            self.df_clean["week_status"].map(status_map).astype("Int8")
        )
        print("Variable day_status_num creada (Weekday=0, Weekend=1)")
        return self
    
    def reorder_columns_for_modeling(self) -> 'SteelEnergyDataCleaner':
        """
        Reordena columnas para tener primero las más relevantes para modelado.
        
        Returns:
            self para encadenamiento de métodos
        """
        cols_first = ["date", "week_status", "day_status_num", "day_of_week", "load_type"]
        cols_first = [c for c in cols_first if c in self.df_clean.columns]
        cols_rest = [c for c in self.df_clean.columns if c not in cols_first]
        self.df_clean = self.df_clean[cols_first + cols_rest]
        print("Columnas reordenadas para modelado")
        return self
    
    def create_full_grid(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Crea una versión con grilla temporal completa (intervalos de 15 minutos).
        
        Args:
            output_path: Ruta opcional para guardar el CSV con grilla completa
            
        Returns:
            DataFrame con grilla temporal completa
        """
        df_full = self.df_clean.set_index("date").sort_index()
        
        idx_full = pd.date_range(
            df_full.index.min().floor("15min"),
            df_full.index.max().ceil("15min"),
            freq="15min"
        )
        df_full = df_full.reindex(idx_full)
        df_full.index.name = "date"
        
        # Recalcular day_of_week, week_status, NSM
        df_full["day_of_week"] = pd.Series(
            df_full.index.day_name(), index=df_full.index, dtype="string"
        )
        df_full["week_status"] = df_full["day_of_week"].map(
            lambda d: "Weekend" if d in ("Saturday", "Sunday") else "Weekday"
        ).astype("string")
        df_full["nsm"] = (
            ((df_full.index.hour * 3600 + df_full.index.minute * 60) / 900).round() * 900
        ).astype("Int64")
        
        # Crear day_status_num (Weekday=0, Weekend=1)
        status_map = {"Weekday": 0, "Weekend": 1}
        df_full["day_status_num"] = df_full["week_status"].map(status_map).astype("Int8")
        
        # Imputar load_type dentro de cada día (ffill/bfill diario)
        df_full["load_type"] = df_full.groupby(pd.Grouper(freq="D"))["load_type"].ffill().bfill()
        
        # Imputación jerárquica numéricas
        def impute_numeric_col(df_, col):
            s = df_[col]
            med1 = df_.groupby(["load_type", "day_of_week"], observed=True)[col].transform("median")
            med2 = df_.groupby(["day_of_week"], observed=True)[col].transform("median")
            medg = s.median()
            return s.fillna(med1).fillna(med2).fillna(medg)
        
        for col in [
            "usage_kwh",
            "lagging_current_reactive_power_kvarh",
            "leading_current_reactive_power_kvarh",
            "co2_tco2",
            "lagging_current_power_factor",
            "leading_current_power_factor",
        ]:
            df_full[col] = impute_numeric_col(df_full, col)
        
        df_full = df_full.reset_index().rename(columns={"index": "date"})
        
        # Reordenar columnas para modelado
        cols_first = ["date", "week_status", "day_status_num", "day_of_week", "load_type"]
        cols_first = [c for c in cols_first if c in df_full.columns]
        cols_rest = [c for c in df_full.columns if c not in cols_first]
        df_full = df_full[cols_first + cols_rest]
        
        print(f"Grilla completa generada: {len(df_full)} filas")
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df_full.to_csv(output_path, index=False)
            print(f"Grilla completa guardada en: {output_path}")
        
        return df_full
    
    def save(self, output_path: str) -> 'SteelEnergyDataCleaner':
        """
        Guarda el DataFrame limpio en un archivo CSV.
        
        Args:
            output_path: Ruta de salida para el CSV limpio
            
        Returns:
            self para encadenamiento de métodos
        """
        self.stats["n_final"] = len(self.df_clean)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df_clean.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print("RESUMEN DE LIMPIEZA")
        print(f"{'='*60}")
        print(f"Filas originales: {self.stats['n_original']:,}")
        print(f"Filas sin fecha eliminadas: {self.stats['n_na_date']:,}")
        print(f"Duplicados removidos: {self.stats['n_duplicates']:,}")
        print(f"Filas finales: {self.stats['n_final']:,}")
        print(f"\nDataset limpio guardado en: {output_path}")
        print(f"{'='*60}")
        
        return self
    
    def run_full_pipeline(
        self,
        input_path: str,
        output_path: str,
        create_fullgrid: bool = False,
        fullgrid_output_path: Optional[str] = None
    ) -> 'SteelEnergyDataCleaner':
        """
        Ejecuta el pipeline completo de limpieza.
        
        Args:
            input_path: Ruta del CSV de entrada
            output_path: Ruta del CSV de salida limpio
            create_fullgrid: Si True, genera también la versión con grilla completa
            fullgrid_output_path: Ruta para guardar la grilla completa (si create_fullgrid=True)
            
        Returns:
            self para encadenamiento de métodos
        """
        (self
            .load_data(input_path)
            .normalize_columns()
            .fix_column_names()
            .convert_numeric_columns()
            .clean_categorical_columns()
            .remove_missing_dates_and_duplicates()
            .recalculate_nsm()
            .apply_physical_rules()
            .impute_missing_values()
            .cap_outliers()
            .handle_mixed_columns()
            .impute_load_type_na()
            .create_day_status_num()
            .reorder_columns_for_modeling()
            .save(output_path)
        )
        
        if create_fullgrid:
            if fullgrid_output_path is None:
                fullgrid_output_path = str(Path(output_path).parent / "clean_fullgrid.csv")
            self.create_full_grid(fullgrid_output_path)
        
        return self


def main():
    """Función principal para ejecutar el script desde línea de comandos."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description="Limpieza de datos del Steel Industry Energy Consumption Dataset"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Ruta del archivo de configuración YAML (debe incluir input_path y output_path)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Ruta del archivo CSV de entrada"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta del archivo CSV de salida"
    )
    parser.add_argument(
        "--dup-strategy",
        type=str,
        choices=["first", "aggregate"],
        default="first",
        help="Estrategia para manejar duplicados (default: first)"
    )
    parser.add_argument(
        "--fullgrid",
        action="store_true",
        help="Generar también versión con grilla temporal completa (15 min)"
    )
    parser.add_argument(
        "--fullgrid-output",
        type=str,
        default=None,
        help="Ruta para la versión con grilla completa"
    )
    
    args = parser.parse_args()
    
    # Determinar rutas de entrada y salida
    input_path = args.input
    output_path = args.output
    dup_strategy = args.dup_strategy
    create_fullgrid = args.fullgrid
    fullgrid_output = args.fullgrid_output
    
    # Si se proporciona archivo de configuración, leer de ahí
    if args.cfg:
        cfg_path = Path(args.cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"No existe el archivo de configuración: {args.cfg}")
        
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # Leer rutas desde configuración si no se proporcionaron como argumentos
        if input_path is None:
            input_path = cfg.get("input_path", "data/raw/steel_energy_clean_fullgrid.csv")
        if output_path is None:
            output_path = cfg.get("output_path", "data/processed/clean.csv")
        if args.dup_strategy == "first":  # Solo sobrescribir si es el valor por defecto
            dup_strategy = cfg.get("dup_strategy", "first")
        if not args.fullgrid:  # Solo sobrescribir si no se especificó en CLI
            create_fullgrid = cfg.get("create_fullgrid", False)
        if fullgrid_output is None:
            fullgrid_output = cfg.get("fullgrid_output_path", None)
    
    # Validar que tengamos al menos input_path
    if input_path is None:
        raise ValueError("Se requiere --input o un archivo de configuración con input_path")
    
    # Usar valor por defecto para output si no se especificó
    if output_path is None:
        output_path = "data/processed/clean.csv"
    
    cleaner = SteelEnergyDataCleaner(dup_strategy=dup_strategy)
    cleaner.run_full_pipeline(
        input_path=input_path,
        output_path=output_path,
        create_fullgrid=create_fullgrid,
        fullgrid_output_path=fullgrid_output
    )
    
    print("\n✓ Limpieza completada exitosamente")


if __name__ == "__main__":
    main()
