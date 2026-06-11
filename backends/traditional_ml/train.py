"""
Train the traditional ML ABSA model.

This script uses the processed SemEval restaurant dataset created by
backends/data/preprocess_dataset.ipynb. It trains one TF-IDF + Logistic
Regression classifier per project aspect and saves all four models together.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BACKENDS_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BACKENDS_DIR / "data" / "processed" / "restaurant_absa_4_aspects.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
REPORT_PATH = Path(__file__).resolve().parent / "training_report.json"

ASPECTS = [
    "Food",
    "Service",
    "Price",
    "Eating Environment / Ambiance",
]

LABELS = ["Positive", "Negative", "Unknown"]


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {DATASET_PATH}. "
            "Run backends/data/preprocess_dataset.ipynb first."
        )

    df = pd.read_csv(DATASET_PATH, dtype={"id": str})
    expected_columns = ["id", "review", *ASPECTS]
    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Unexpected dataset columns: {list(df.columns)}. "
            f"Expected: {expected_columns}"
        )

    if df["review"].isna().any() or df["review"].str.len().eq(0).any():
        raise ValueError("Dataset contains empty review text.")

    for aspect in ASPECTS:
        bad_labels = sorted(set(df[aspect]) - set(LABELS))
        if bad_labels:
            raise ValueError(f"{aspect} has invalid labels: {bad_labels}")

    return df


def train() -> dict:
    df = load_dataset()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    models: dict[str, Pipeline] = {}
    report: dict[str, object] = {
        "dataset": str(DATASET_PATH),
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "aspects": ASPECTS,
        "labels": LABELS,
        "metrics": {},
    }

    for aspect in ASPECTS:
        pipeline = build_pipeline()
        pipeline.fit(train_df["review"], train_df[aspect])
        models[aspect] = pipeline

        y_true = test_df[aspect]
        y_pred = pipeline.predict(test_df["review"])
        report["metrics"][aspect] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro")),
            "weighted_f1": float(
                f1_score(y_true, y_pred, labels=LABELS, average="weighted")
            ),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=LABELS,
                output_dict=True,
                zero_division=0,
            ),
        }

    artifact = {
        "method": "traditional_ml",
        "model_type": "tfidf_logistic_regression_per_aspect",
        "aspects": ASPECTS,
        "labels": LABELS,
        "models": models,
    }

    joblib.dump(artifact, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    training_report = train()
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved training report: {REPORT_PATH}")
    print("\nEvaluation summary:")
    for aspect, metrics in training_report["metrics"].items():
        print(
            f"- {aspect}: accuracy={metrics['accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}, "
            f"weighted_f1={metrics['weighted_f1']:.3f}"
        )
