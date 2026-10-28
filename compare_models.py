#!/usr/bin/env python3
"""Score several prediction files against val.csv and print one comparison table.

Each prediction file is a CSV with columns id,prediction (what ensemble.py,
predict_with_ollama.py, predict_with_gpt4.py and train_transformer.py write).
Some runs (like the GPT-4 one) only cover a sample of the validation set, so
the table shows how many rows each model was scored on. Use --common to score
every model on only the rows they all share, for a fair head-to-head.

Examples:
    python compare_models.py
    python compare_models.py --preds gpt4o-mini=val_preds_gpt4.csv mbert=preds_bert-base-multilingual-cased.csv
    python compare_models.py --by-language --confusion
"""
import argparse
import glob
import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABEL_NAMES = ["contra", "neutral", "entail"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", default="val.csv", help="Labelled validation file (id, language, label)")
    p.add_argument("--preds", nargs="*", default=None,
                   help="NAME=FILE pairs, or bare files. Default: every *preds*.csv in this folder")
    p.add_argument("--common", action="store_true", help="Score all models on the ids they have in common")
    p.add_argument("--by-language", action="store_true", help="Also print accuracy per language")
    p.add_argument("--confusion", action="store_true", help="Also print a confusion matrix per model")
    return p.parse_args()


def resolve_pred_files(specs):
    if not specs:
        specs = sorted(glob.glob("*preds*.csv"))
    pairs = []
    for spec in specs:
        name, _, path = spec.partition("=") if "=" in spec else ("", "", spec)
        name = name or os.path.splitext(os.path.basename(path))[0]
        pairs.append((name, path))
    return pairs


def load_preds(path):
    df = pd.read_csv(path, dtype={"id": str})
    if df.empty or not {"id", "prediction"} <= set(df.columns):
        return None
    df = df.dropna(subset=["prediction"])
    df["prediction"] = df["prediction"].astype(int)
    return df[["id", "prediction"]]


def main():
    args = parse_args()
    gold = pd.read_csv(args.gold, dtype={"id": str})
    gold["label"] = gold["label"].astype(int)

    runs = {}
    for name, path in resolve_pred_files(args.preds):
        if not os.path.exists(path):
            print(f"skip {name}: {path} not found", file=sys.stderr)
            continue
        preds = load_preds(path)
        if preds is None:
            print(f"skip {name}: {path} has no predictions", file=sys.stderr)
            continue
        runs[name] = gold.merge(preds, on="id", how="inner")

    if not runs:
        sys.exit("No usable prediction files found.")

    if args.common:
        shared = set.intersection(*(set(df["id"]) for df in runs.values()))
        runs = {n: df[df["id"].isin(shared)] for n, df in runs.items()}
        print(f"Scoring every model on the {len(shared)} ids they share.\n")

    rows = []
    for name, df in runs.items():
        en = df[df["language"] == "English"]
        other = df[df["language"] != "English"]
        rows.append({
            "model": name,
            "rows": len(df),
            "accuracy": accuracy_score(df["label"], df["prediction"]),
            "macro_f1": f1_score(df["label"], df["prediction"], average="macro"),
            "english_acc": accuracy_score(en["label"], en["prediction"]) if len(en) else float("nan"),
            "non_english_acc": accuracy_score(other["label"], other["prediction"]) if len(other) else float("nan"),
        })
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.by_language:
        cells = {}
        for name, df in runs.items():
            g = df.assign(correct=df["label"] == df["prediction"]).groupby("language")["correct"]
            cells[name] = g.mean().map("{:.3f}".format) + " (" + g.size().astype(str) + ")"
        per_lang = pd.DataFrame(cells).fillna("-")
        per_lang.insert(0, "val_rows", gold.groupby("language").size())
        print("\nAccuracy by language, with the number of rows scored in brackets:")
        print(per_lang.sort_values("val_rows", ascending=False).to_string())

    if args.confusion:
        for name, df in runs.items():
            cm = confusion_matrix(df["label"], df["prediction"], labels=[0, 1, 2])
            print(f"\nConfusion matrix: {name} (rows = true, columns = predicted)")
            print(pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES).to_string())


if __name__ == "__main__":
    main()
