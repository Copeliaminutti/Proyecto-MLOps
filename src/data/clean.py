import argparse
import pandas as pd
import yaml
from pathlib import Path

def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if cfg.get("drop_na", True):
        df = df.dropna()
    if cfg.get("duplicates", True):
        df = df.drop_duplicates()
    # TODO: outliers según cfg
    return df

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/eda.yaml")
    parser.add_argument("--input", type=str, default="data/raw/data.csv")
    parser.add_argument("--output", type=str, default="data/processed/clean.csv")
    ns = parser.parse_args(args)

    cfg = yaml.safe_load(Path(ns.cfg).read_text(encoding="utf-8"))
    df = pd.read_csv(ns.input)
    df_clean = clean(df, cfg)
    Path(ns.output).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(ns.output, index=False)
    print(f"Guardado: {ns.output} -> filas: {len(df_clean)}")

if __name__ == "__main__":
    main()
