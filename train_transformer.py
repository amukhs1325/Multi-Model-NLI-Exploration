#!/usr/bin/env python3
"""Fine-tune a multilingual transformer on the Watson NLI data from the command line.

This is the notebook's fine-tuning code as a reusable script. It reproduces the
notebook's preprocessing exactly (drop empty rows, flip labels 0<->2, stratified
90/10 split with random_state=42), so the held-out rows are the same 1,212 rows
as val.csv. Predictions are written as id,prediction so they can be scored next
to the LLM runs with compare_models.py.

Examples:
    python train_transformer.py --model bert-base-multilingual-cased
    python train_transformer.py --model joeddav/xlm-roberta-large-xnli --epochs 3 --lr 1e-5 --batch-size 8
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL_NAMES = ["contradiction", "neutral", "entailment"]
# The Kaggle file uses 0=entailment, 2=contradiction. Flip so 0=contra, 1=neutral, 2=entail.
FLIP = {0: 2, 1: 1, 2: 0}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bert-base-multilingual-cased", help="Hugging Face checkpoint to fine-tune")
    p.add_argument("--train-csv", default="train.csv", help="Original Kaggle train.csv")
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--patience", type=int, default=2, help="Early-stopping patience in epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None, help="Where checkpoints go (default: runs/<model name>)")
    p.add_argument("--preds-out", default=None, help="Prediction CSV (default: preds_<model name>.csv)")
    return p.parse_args()


def load_splits(path, seed):
    df = pd.read_csv(path).dropna(subset=["premise", "hypothesis", "label"])
    df["label"] = df["label"].astype(int).map(FLIP)
    return train_test_split(df, test_size=0.10, stratify=df["label"], random_state=seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    short_name = args.model.split("/")[-1]
    output_dir = args.output_dir or os.path.join("runs", short_name)
    preds_out = args.preds_out or f"preds_{short_name}.csv"

    train_df, val_df = load_splits(args.train_csv, args.seed)
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=args.max_length)

    cols = ["premise", "hypothesis", "label"]
    train_ds = Dataset.from_pandas(train_df[cols], preserve_index=False).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[cols], preserve_index=False).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=dict(enumerate(LABEL_NAMES)),
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
        ignore_mismatched_sizes=True,  # allows checkpoints whose classifier head has a different size
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    logits = trainer.predict(val_ds).predictions
    preds = np.argmax(logits, axis=-1)
    print(classification_report(val_df["label"], preds, target_names=LABEL_NAMES, digits=3))

    pd.DataFrame({"id": val_df["id"].values, "prediction": preds}).to_csv(preds_out, index=False)
    print(f"Wrote {len(preds)} predictions to {preds_out}")


if __name__ == "__main__":
    main()
