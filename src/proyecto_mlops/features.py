import argparse
import pandas as pd
from pathlib import Path
from proyecto_mlops.config import CONFIGS

def build(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Remove 'date' column explicitly if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    date_keywords = ['date', 'fecha', 'time', 'timestamp', 'hour', 'minute', 'second']
    num_cols = [c.lower() for c in cfg.get('numeric_features', []) if not any(k in c.lower() for k in date_keywords)]
    cat_cols = [c.lower() for c in cfg.get('categorical_features', []) if not any(k in c.lower() for k in date_keywords)]
    target = cfg.get('target', None)

    # Clean problematic string values in all columns except target
    import numpy as np
    problematic_values = ['bad', 'unknown', 'n/a', 'na', 'none', 'null', 'missing', '?', '--']
    for col in df.columns:
        if col == target:
            continue
        df[col] = df[col].replace(problematic_values, np.nan).infer_objects(copy=False)
        # Try to convert to numeric (if possible, coerce errors to NaN)
        numeric_attempt = pd.to_numeric(df[col], errors='coerce')
        if not numeric_attempt.isna().all():  # Only convert if there are valid numeric values
            df[col] = numeric_attempt

    # Drop columns that are entirely empty (all NaN) except target
    cols_to_drop = [col for col in df.columns if col != target and df[col].isna().all()]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Impute missing values
    from sklearn.impute import SimpleImputer
    # Numeric columns (excluding target)
    num_cols = [col for col in df.columns if col != target and pd.api.types.is_numeric_dtype(df[col])]
    if num_cols:
        imputer_num = SimpleImputer(strategy='mean')
        df[num_cols] = pd.DataFrame(
            imputer_num.fit_transform(df[num_cols]),
            columns=num_cols,
            index=df.index
        )
    # Categorical columns (object/category, excluding target)
    cat_cols_auto = [col for col in df.columns if col != target and df[col].dtype in ['object', 'category']]
    if cat_cols_auto:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols_auto] = pd.DataFrame(
            imputer_cat.fit_transform(df[cat_cols_auto]),
            columns=cat_cols_auto,
            index=df.index
        )

    # One-hot encode ALL object/category columns except the target
    if cfg.get('one_hot', False):
        if cat_cols_auto:
            df = pd.get_dummies(df, columns=cat_cols_auto, drop_first=False)
    return df


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="src/proyecto_mlops/data/processed/clean.csv")
    parser.add_argument("--output", type=str, default="data/interim/features.csv")
    ns = parser.parse_args(args)

    df_feat = run_features(ns.input, ns.output)
    print(f"Guardado: {ns.output} -> filas: {len(df_feat)}")

def run_features(input_path: str, output_path: str = None) -> pd.DataFrame:
    cfg = CONFIGS["features"]
    df = pd.read_csv(input_path)
    df_feat = build(df, cfg)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_csv(output_path, index=False)
    return df_feat

if __name__ == "__main__":
    main()
