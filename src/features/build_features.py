import argparse
import pandas as pd
import yaml
from pathlib import Path

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

    # One-hot encoding
    if cfg.get('one_hot', False) and cat_cols:
        # Only use categorical columns that exist in the DataFrame
        cat_cols_existing = [c for c in cat_cols if c in df.columns]
        if cat_cols_existing:
            df = pd.get_dummies(df, columns=cat_cols_existing, drop_first=False)
            # After encoding, update num_cols to include new dummies
            new_cols = [col for col in df.columns if col != target]
            num_cols = [col for col in new_cols if col != target]

    return df

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/features.yaml")
    parser.add_argument("--input", type=str, default="data/processed/clean.csv")
    parser.add_argument("--output", type=str, default="data/interim/features.csv")
    ns = parser.parse_args(args)

    cfg = yaml.safe_load(Path(ns.cfg).read_text(encoding="utf-8"))
    df = pd.read_csv(ns.input)
    df_feat = build(df, cfg)
    Path(ns.output).parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(ns.output, index=False)
    print(f"Guardado: {ns.output} -> filas: {len(df_feat)}")

if __name__ == "__main__":
    main()
