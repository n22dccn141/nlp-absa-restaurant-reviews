"""
backends/rule_based/predict.py
================================
Entry point — run and demo the rule-based ABSA method.
Member 2 (Baseline Model).

Usage:
    python predict.py
    python predict.py "The food was amazing but the service was rude."
"""

import sys
import json
from rule_based_model import predict, ALL_ASPECTS


def display(output: dict) -> None:
    """Pretty-print one predict() output."""
    ICONS = {"Positive": "✅", "Negative": "❌", "Unknown": "➖"}
    W = 65
    print("\n" + "═" * W)
    print(f'  Review: "{output["review"]}"')
    print("─" * W)
    print(f"  {'Aspect':<35}  Sentiment")
    print(f"  {'─'*35}  {'─'*12}")
    for item in output["results"]:
        icon = ICONS.get(item["sentiment"], "➖")
        print(f"  {item['aspect']:<35}  {icon} {item['sentiment']}")
    print("═" * W)


if __name__ == "__main__":
    print("[INFO] NLP backend : spaCy (en_core_web_sm)")
    print("[INFO] Output format: Section 22.1 of NLP_Official_Documents_v03\n")

    if len(sys.argv) > 1:
        review = " ".join(sys.argv[1:])
        output = predict(review)
        display(output)
        print("\n[JSON output]")
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Section 30 — Example Full System Demonstration
        demo_reviews = [
            "The burger was delicious.",
            "The waiter was rude and the food was cold.",
            "The restaurant was clean, but the food was expensive.",
            "The music was too loud, but the staff was helpful.",
            "The pasta tasted great, the bill was reasonable, and the place was comfortable.",
            "I visited the restaurant yesterday.",  # all Unknown

            # Standard test cases
            "The food was delicious but the service was slow.",
            "The pasta was not good and the price was not reasonable at all.",
            "Coffee was absolutely terrible, however the ambiance was cozy.",

            # ── Điền câu của bạn vào đây ──────────────────────────
            "Your custom sentence goes here.",
        ]

        print("=" * 65)
        print("  DEMO — Rule-Based ABSA  (backends/rule_based/predict.py)")
        print("  Section 30: Example Full System Demonstration")
        print("=" * 65)

        for review in demo_reviews:
            output = predict(review)
            display(output)