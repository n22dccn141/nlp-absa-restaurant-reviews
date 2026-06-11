"""
backends/rule_based/predict.py
================================
Entry point — run and demo the rule-based ABSA method.
Member 2 (Baseline Model).

Usage:
    python predict.py
    python predict.py "The food was amazing but the service was rude."
    python predict.py "The food was amazing but the service was rude." --debug
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

    # Tách --debug flag khỏi review text
    debug = "--debug" in sys.argv
    args  = [a for a in sys.argv[1:] if a != "--debug"]

    if args:
        review = " ".join(args)
        output = predict(review, debug=debug)
        display(output)
        if not debug:
            print("\n[JSON output]")
            print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        demo_reviews = [
            "The burger was delicious.",
            "The waiter was rude and the food was cold.",
            "The restaurant was clean, but the food was expensive.",
            "The music was too loud, but the staff was helpful.",
            "The pasta tasted great, the bill was reasonable, and the place was comfortable.",
            "I visited the restaurant yesterday.",
            "The food was delicious but the service was slow.",
            "The pasta was not good and the price was not reasonable at all.",
            "Coffee was absolutely terrible, however the ambiance was cozy.",
            "The price was not reasonable.",
            "The restaurant was not clean.",
            "The place was not comfortable.",
            "The food was not expensive.",
            "Food was so good."
            # ── Điền câu của bạn vào đây ──────────────────────────
        ]

        print("=" * 65)
        print("  DEMO — Rule-Based ABSA  (backends/rule_based/predict.py)")
        print("  Section 30: Example Full System Demonstration")
        print("=" * 65)

        for review in demo_reviews:
            output = predict(review, debug=debug)
            display(output)