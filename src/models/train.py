import argparse
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/train.yaml")
    parser.add_argument("--input", type=str, default="data/interim/features.csv")
    parser.add_argument("--out", type=str, default="artifacts/model.pkl")
    ns = parser.parse_args(args)

    cfg = yaml.safe_load(Path(ns.cfg).read_text(encoding='utf-8'))
    target = cfg.get("target", "target")

    df = pd.read_csv(ns.input)
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("test_size", 0.2), random_state=cfg.get("random_state", 42)
    )

    model = RandomForestClassifier(**cfg.get("params", {}))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    print({"accuracy": acc, "f1": f1})

    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ns.out)
    print(f"Modelo guardado en {ns.out}")

if __name__ == "__main__":
    main()
