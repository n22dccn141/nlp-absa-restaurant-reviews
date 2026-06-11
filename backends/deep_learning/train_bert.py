"""
Fine-tune a transformer model for restaurant ABSA.

Recommended Windows/RTX 3060 command:
    python backends/deep_learning/train_bert.py ^
      --model microsoft/deberta-v3-base ^
      --epochs 5 ^
      --batch-size 8 ^
      --gradient-accumulation-steps 2

Quick smoke test before full training:
    python backends/deep_learning/train_bert.py --quick

Input dataset:
    backends/data/processed/restaurant_absa_aspect_rows.csv

Saved outputs:
    backends/deep_learning/bert_absa_model/
    backends/deep_learning/bert_training_report.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


BACKENDS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = (
    BACKENDS_DIR / "data" / "processed" / "restaurant_absa_aspect_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "bert_absa_model"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "bert_training_report.json"

ASPECTS = [
    "Food",
    "Service",
    "Price",
    "Eating Environment / Ambiance",
]

LABEL2ID = {
    "Positive": 0,
    "Negative": 1,
    "Unknown": 2,
}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer ABSA model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-rows", type=int, default=1000)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dataset(path: Path, quick: bool, quick_rows: int, seed: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {path}. Run backends/data/preprocess_dataset.ipynb first."
        )

    df = pd.read_csv(path, dtype={"id": str})
    expected_columns = ["id", "review", "aspect", "sentiment"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, got {list(df.columns)}")

    bad_aspects = sorted(set(df["aspect"]) - set(ASPECTS))
    if bad_aspects:
        raise ValueError(f"Unexpected aspects: {bad_aspects}")

    bad_labels = sorted(set(df["sentiment"]) - set(LABEL2ID))
    if bad_labels:
        raise ValueError(f"Unexpected sentiment labels: {bad_labels}")

    df = df.dropna(subset=["id", "review", "aspect", "sentiment"]).copy()
    df["review"] = df["review"].astype(str).str.strip()
    df["aspect"] = df["aspect"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df = df[df["review"].str.len() > 0].copy()
    df["label"] = df["sentiment"].map(LABEL2ID)

    if quick:
        sample_size = min(quick_rows, len(df))
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    return df


def split_by_review_id(
    df: pd.DataFrame,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by review ID to avoid the same review appearing in multiple splits."""
    review_ids = pd.Series(df["id"].unique())
    train_val_ids, test_ids = train_test_split(
        review_ids,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    adjusted_validation_size = validation_size / (1.0 - test_size)
    train_ids, validation_ids = train_test_split(
        train_val_ids,
        test_size=adjusted_validation_size,
        random_state=seed,
        shuffle=True,
    )

    train_df = df[df["id"].isin(train_ids)].reset_index(drop=True)
    validation_df = df[df["id"].isin(validation_ids)].reset_index(drop=True)
    test_df = df[df["id"].isin(test_ids)].reset_index(drop=True)
    return train_df, validation_df, test_df


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        df[["id", "review", "aspect", "sentiment", "label"]],
        preserve_index=False,
    )


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["review"],
            batch["aspect"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    return tokenized.remove_columns(["id", "review", "aspect", "sentiment"])


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(ID2LABEL),
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(ID2LABEL),
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
    }


def build_training_args(args: argparse.Namespace, fp16: bool) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "logging_steps": 50,
        "report_to": "none",
        "seed": args.seed,
        "fp16": fp16,
    }

    # Transformers renamed evaluation_strategy to eval_strategy in newer versions.
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    kwargs["save_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def detailed_report(
    trainer: Trainer,
    test_dataset: Dataset,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict[str, Any]:
    prediction_output = trainer.predict(test_dataset)
    y_true = prediction_output.label_ids
    y_pred = np.argmax(prediction_output.predictions, axis=-1)

    per_label_precision, per_label_recall, per_label_f1, per_label_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(ID2LABEL),
            zero_division=0,
        )
    )

    per_label = {}
    for label_id, label_name in ID2LABEL.items():
        per_label[label_name] = {
            "precision": float(per_label_precision[label_id]),
            "recall": float(per_label_recall[label_id]),
            "f1": float(per_label_f1[label_id]),
            "support": int(per_label_support[label_id]),
        }

    test_metrics = compute_metrics((prediction_output.predictions, y_true))
    test_metrics = {key: float(value) for key, value in test_metrics.items()}

    examples = test_df[["id", "review", "aspect", "sentiment"]].copy()
    examples["prediction"] = [ID2LABEL[int(label)] for label in y_pred]
    examples = examples.head(25).to_dict(orient="records")

    return {
        "model": args.model,
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "device": detect_device(),
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "rows": {
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df)),
        },
        "training_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size
            * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "seed": args.seed,
            "quick": args.quick,
        },
        "test_metrics": test_metrics,
        "per_label": per_label,
        "confusion_matrix_labels": [ID2LABEL[index] for index in ID2LABEL],
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(ID2LABEL),
        ).tolist(),
        "sample_predictions": examples,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"Device: {detect_device()}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")

    df = load_dataset(args.dataset, args.quick, args.quick_rows, args.seed)
    train_df, validation_df, test_df = split_by_review_id(
        df,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    print(f"Rows: train={len(train_df)}, validation={len(validation_df)}, test={len(test_df)}")
    print("Train label counts:")
    print(train_df["sentiment"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = tokenize_dataset(to_hf_dataset(train_df), tokenizer, args.max_length)
    validation_dataset = tokenize_dataset(
        to_hf_dataset(validation_df),
        tokenizer,
        args.max_length,
    )
    test_dataset = tokenize_dataset(to_hf_dataset(test_df), tokenizer, args.max_length)

    fp16 = torch.cuda.is_available() and not args.no_fp16
    training_args = build_training_args(args, fp16=fp16)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    report = detailed_report(
        trainer=trainer,
        test_dataset=test_dataset,
        test_df=test_df,
        args=args,
        train_df=train_df,
        validation_df=validation_df,
    )
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved model to: {args.output_dir}")
    print(f"Saved report to: {args.report_path}")
    print("Test metrics:")
    for key, value in report["test_metrics"].items():
        print(f"- {key}: {value:.4f}")


if __name__ == "__main__":
    main()
