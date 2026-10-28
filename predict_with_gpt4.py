#!/usr/bin/env python3
"""Zero-shot NLI with an OpenAI chat model on a random sample of val.csv.

Uses the openai>=1.0 client, retries on rate limits, and saves predictions as
id,prediction so the run can be scored with compare_models.py.

Usage:
    export OPENAI_API_KEY=...
    python predict_with_gpt4.py --n 100 --model gpt-4o-mini
"""
import argparse
import csv
import random
import re
import time

from openai import APIError, OpenAI, RateLimitError
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

PROMPT = """
Task
You will be given two short texts: a Premise and a Hypothesis. Decide whether the Hypothesis is
contradicted by the Premise (label 0), logically unrelated / possible but not guaranteed (label 1 = neutral),
or logically entailed by the Premise (label 2).
Return only the single digit 0, 1, or 2, no extra words.

Decision Guide

0 (Contradiction)    The Hypothesis cannot be true if the Premise is true.
1 (Neutral)          Both texts can be true at the same time, but the Premise does not guarantee the Hypothesis.
2 (Entailment)       If the Premise is true, the Hypothesis must also be true.

Format
Premise: {premise_text}
Hypothesis: {hypothesis_text}
Answer:
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="val.csv", help="Validation file with a label column")
    p.add_argument("--output", default=None, help="Default: preds_<model>.csv")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--n", type=int, default=100, help="Number of random rows to label (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def classify(client, model, premise, hypothesis, retries=5):
    prompt = PROMPT.format(premise_text=premise.strip(), hypothesis_text=hypothesis.strip()).strip()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            text = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\b([012])\b", text)
            return int(m.group(1)) if m else 1  # fall back to neutral
        except (RateLimitError, APIError):
            time.sleep(2 ** attempt)
    return 1


def main():
    args = parse_args()
    with open(args.input, newline="", encoding="utf-8") as fin:
        all_rows = list(csv.DictReader(fin))

    random.seed(args.seed)
    rows = all_rows if args.n <= 0 or args.n >= len(all_rows) else random.sample(all_rows, args.n)

    args.output = args.output or f"preds_{args.model}.csv"
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    y_true, y_pred = [], []
    with open(args.output, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "prediction"])
        for row in tqdm(rows, desc=f"{args.model} on {len(rows)} rows", unit="row"):
            pred = classify(client, args.model, row["premise"], row["hypothesis"])
            writer.writerow([row["id"], pred])
            y_pred.append(pred)
            y_true.append(int(row["label"]))

    print(f"\nAccuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Macro-F1 : {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
