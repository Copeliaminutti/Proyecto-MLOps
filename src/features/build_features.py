import argparse
import pandas as pd
import yaml
from pathlib import Path

def build(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # TODO: implementar escalado/one-hot según cfg
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
