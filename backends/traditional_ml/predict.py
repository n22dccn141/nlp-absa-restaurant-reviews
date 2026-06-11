"""
Predict aspect-based sentiment with the trained traditional ML model.

Usage:
    python predict.py "The food was delicious but the service was slow."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib


MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}. "
            "Run backends/traditional_ml/train.py first."
        )
    return joblib.load(MODEL_PATH)


def predict(review: str) -> dict:
    review = review.strip()
    if not review:
        raise ValueError("Review text must not be empty.")

    artifact = load_model()
    aspects = artifact["aspects"]
    models = artifact["models"]

    results = []
    for aspect in aspects:
        sentiment = models[aspect].predict([review])[0]
        results.append({"aspect": aspect, "sentiment": sentiment})

    return {
        "review": review,
        "method": artifact.get("method", "traditional_ml"),
        "results": results,
    }


def display(output: dict) -> None:
    print("\nReview:", output["review"])
    print("Method:", output["method"])
    print("-" * 62)
    for item in output["results"]:
        print(f"{item['aspect']:<35} {item['sentiment']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "The food was delicious. The service was bad. The environment was nice."

    prediction = predict(text)
    display(prediction)
    print("\nJSON output:")
    print(json.dumps(prediction, indent=2, ensure_ascii=False))
