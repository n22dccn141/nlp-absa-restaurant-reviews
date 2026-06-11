"""
Predict restaurant ABSA sentiment with the trained transformer model.

Usage:
    python backends/deep_learning/predict_bert.py "The food was delicious."

Expected local model folder:
    backends/deep_learning/bert_absa_model/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parent / "bert_absa_model"

ASPECTS = [
    "Food",
    "Service",
    "Price",
    "Eating Environment / Ambiance",
]

ID2LABEL = {
    0: "Positive",
    1: "Negative",
    2: "Unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ABSA sentiment with BERT.")
    parser.add_argument("review", nargs="*", help="Restaurant review text.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BertAbsaPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR, max_length: int = 160) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Missing BERT model folder: {model_dir}. "
                "Copy or unzip bert_absa_model into backends/deep_learning/ first."
            )

        self.model_dir = model_dir
        self.max_length = max_length
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

        config_id2label = getattr(self.model.config, "id2label", None) or ID2LABEL
        self.id2label = {int(key): value for key, value in config_id2label.items()}

    def predict(self, review: str) -> dict[str, Any]:
        review = review.strip()
        if not review:
            raise ValueError("Review text must not be empty.")

        encoded = self.tokenizer(
            [review] * len(ASPECTS),
            ASPECTS,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)
            if not torch.isfinite(output.logits).all():
                raise RuntimeError(
                    "The BERT model produced non-finite logits. "
                    "The saved model is likely numerically unstable from training. "
                    "Retrain with --no-fp16 and a lower learning rate such as 1e-5."
                )
            probabilities = torch.softmax(output.logits, dim=-1).cpu()
            predictions = probabilities.argmax(dim=-1).tolist()

        results = []
        for aspect, label_id, probs in zip(ASPECTS, predictions, probabilities):
            confidence = float(probs[label_id])
            results.append(
                {
                    "aspect": aspect,
                    "sentiment": self.id2label[int(label_id)],
                    "confidence": round(confidence, 4),
                }
            )

        return {
            "review": review,
            "method": "bert",
            "results": results,
        }


def display(output: dict[str, Any]) -> None:
    print("\nReview:", output["review"])
    print("Method:", output["method"])
    print("-" * 74)
    for item in output["results"]:
        print(
            f"{item['aspect']:<35} "
            f"{item['sentiment']:<10} "
            f"confidence={item['confidence']:.4f}"
        )


def predict(review: str, model_dir: Path = MODEL_DIR, max_length: int = 160) -> dict[str, Any]:
    predictor = BertAbsaPredictor(model_dir=model_dir, max_length=max_length)
    return predictor.predict(review)


if __name__ == "__main__":
    args = parse_args()
    review_text = " ".join(args.review).strip()
    if not review_text:
        review_text = "The food was delicious but the service was slow."

    predictor = BertAbsaPredictor(model_dir=args.model_dir, max_length=args.max_length)
    prediction = predictor.predict(review_text)

    if not args.json_only:
        display(prediction)
        print("\nJSON output:")
    print(json.dumps(prediction, indent=2, ensure_ascii=False))
