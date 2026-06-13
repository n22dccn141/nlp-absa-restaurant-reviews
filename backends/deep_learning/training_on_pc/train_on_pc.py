"""
Standalone Windows PC training script for the ABSA deep learning model.

This file is designed so you can copy only the training_on_pc folder to your
Windows RTX 3060 machine and train there.

Default input:
    restaurant_absa_aspect_rows.csv

Default outputs:
    model_output/bert_absa_model/
    model_output/bert_training_report.json
    model_output/bert_absa_model.zip
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import shutil
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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "restaurant_absa_aspect_rows.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "model_output"
DEFAULT_MODEL_DIR = DEFAULT_OUTPUT_ROOT / "bert_absa_model"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_ROOT / "bert_training_report.json"
DEFAULT_ZIP_BASE = DEFAULT_OUTPUT_ROOT / "bert_absa_model"

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
    parser = argparse.ArgumentParser(description="Train ABSA transformer model on PC.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
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
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-rows", type=int, default=1000)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--class-weighted-loss",
        action="store_true",
        help="Use inverse-frequency class weights in cross entropy loss.",
    )
    parser.add_argument(
        "--max-unknown-to-known-ratio",
        type=float,
        default=0.0,
        help=(
            "Downsample Unknown labels in the training split only. "
            "0 disables downsampling. Example: 1.5 keeps at most 1.5x as many "
            "Unknown samples as Positive+Negative samples."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)}"
    return "cpu"


def load_data(path: Path, quick: bool, quick_rows: int, seed: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, dtype={"id": str})
    expected_columns = ["id", "review", "aspect", "sentiment"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Expected {expected_columns}, got {list(df.columns)}")

    bad_aspects = sorted(set(df["aspect"]) - set(ASPECTS))
    if bad_aspects:
        raise ValueError(f"Unexpected aspects: {bad_aspects}")

    bad_labels = sorted(set(df["sentiment"]) - set(LABEL2ID))
    if bad_labels:
        raise ValueError(f"Unexpected sentiment labels: {bad_labels}")

    df = df.dropna(subset=["id", "review", "aspect", "sentiment"]).copy()
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"].str.len() > 0].copy()
    df["label"] = df["sentiment"].map(LABEL2ID).astype(int)

    if quick:
        df = df.sample(n=min(quick_rows, len(df)), random_state=seed).reset_index(drop=True)

    return df


def split_by_review_id(
    df: pd.DataFrame,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def downsample_unknown_train_rows(
    train_df: pd.DataFrame,
    max_unknown_to_known_ratio: float,
    seed: int,
) -> pd.DataFrame:
    if max_unknown_to_known_ratio <= 0:
        return train_df

    known_df = train_df[train_df["sentiment"] != "Unknown"]
    unknown_df = train_df[train_df["sentiment"] == "Unknown"]
    max_unknown = int(len(known_df) * max_unknown_to_known_ratio)

    if len(known_df) == 0 or len(unknown_df) <= max_unknown:
        return train_df

    sampled_unknown_df = unknown_df.sample(n=max_unknown, random_state=seed)
    return (
        pd.concat([known_df, sampled_unknown_df])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    label_counts = train_df["label"].value_counts().reindex(list(ID2LABEL), fill_value=0)
    if (label_counts == 0).any():
        missing = [ID2LABEL[index] for index, count in label_counts.items() if count == 0]
        raise ValueError(f"Cannot compute class weights. Missing labels: {missing}")

    total = label_counts.sum()
    weights = total / (len(ID2LABEL) * label_counts)
    return torch.tensor(weights.sort_index().to_numpy(dtype=np.float32))


def to_dataset(df: pd.DataFrame) -> Dataset:
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

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(ID2LABEL),
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(ID2LABEL),
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
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
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "logging_steps": 50,
        "report_to": "none",
        "seed": args.seed,
        "fp16": fp16,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


class WeightedLossTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: AutoModelForSequenceClassification,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        labels = inputs.pop("labels", None)
        if labels is None:
            labels = inputs.pop("label")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fn(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


def make_report(
    trainer: Trainer,
    tokenized_test: Dataset,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    args: argparse.Namespace,
    class_weights: torch.Tensor | None,
    original_train_rows: int,
) -> dict[str, Any]:
    prediction_output = trainer.predict(tokenized_test)
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

    sample_predictions = test_df[["id", "review", "aspect", "sentiment"]].copy()
    sample_predictions["prediction"] = [ID2LABEL[int(label)] for label in y_pred]

    return {
        "model": args.model,
        "device": detect_device(),
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "rows": {
            "original_train": int(original_train_rows),
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df)),
        },
        "training_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "quick": args.quick,
            "seed": args.seed,
            "class_weighted_loss": args.class_weighted_loss,
            "max_unknown_to_known_ratio": args.max_unknown_to_known_ratio,
        },
        "class_weights": (
            {
                ID2LABEL[index]: float(value)
                for index, value in enumerate(class_weights.tolist())
            }
            if class_weights is not None
            else None
        ),
        "test_metrics": test_metrics,
        "per_label": per_label,
        "confusion_matrix_labels": [ID2LABEL[index] for index in ID2LABEL],
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(ID2LABEL),
        ).tolist(),
        "sample_predictions": sample_predictions.head(25).to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("Device:", detect_device())
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
    print("Model:", args.model)
    print("Dataset:", args.dataset)

    df = load_data(args.dataset, args.quick, args.quick_rows, args.seed)
    train_df, validation_df, test_df = split_by_review_id(
        df,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    original_train_rows = len(train_df)
    train_df = downsample_unknown_train_rows(
        train_df,
        max_unknown_to_known_ratio=args.max_unknown_to_known_ratio,
        seed=args.seed,
    )
    print(f"Rows: train={len(train_df)}, validation={len(validation_df)}, test={len(test_df)}")
    print("Training label counts:")
    print(train_df["sentiment"].value_counts())
    print("Validation label counts:")
    print(validation_df["sentiment"].value_counts())

    class_weights = compute_class_weights(train_df) if args.class_weighted_loss else None
    if class_weights is not None:
        print("Class weights:")
        for index, value in enumerate(class_weights.tolist()):
            print(f"- {ID2LABEL[index]}: {value:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    tokenized_train = tokenize_dataset(to_dataset(train_df), tokenizer, args.max_length)
    tokenized_validation = tokenize_dataset(to_dataset(validation_df), tokenizer, args.max_length)
    tokenized_test = tokenize_dataset(to_dataset(test_df), tokenizer, args.max_length)

    fp16 = torch.cuda.is_available() and not args.no_fp16
    training_args = build_training_args(args, fp16=fp16)
    trainer_class = WeightedLossTrainer if args.class_weighted_loss else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        **({"class_weights": class_weights} if args.class_weighted_loss else {}),
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    report = make_report(
        trainer=trainer,
        tokenized_test=tokenized_test,
        test_df=test_df,
        train_df=train_df,
        validation_df=validation_df,
        args=args,
        class_weights=class_weights,
        original_train_rows=original_train_rows,
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.no_zip:
        zip_path = shutil.make_archive(
            str(DEFAULT_ZIP_BASE),
            "zip",
            root_dir=args.output_dir.parent,
            base_dir=args.output_dir.name,
        )
        print("Saved zipped model:", zip_path)

    print("Saved model folder:", args.output_dir)
    print("Saved report:", args.report_path)
    print("Test metrics:")
    for key, value in report["test_metrics"].items():
        print(f"- {key}: {value:.4f}")


if __name__ == "__main__":
    main()
