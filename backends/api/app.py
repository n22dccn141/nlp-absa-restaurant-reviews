"""
Local HTTP API for the Flutter ABSA app.

Run from the project root:
    python3 backends/api/app.py

Endpoints:
    GET  /health
    POST /analyze

POST body:
    {"review": "The food was delicious.", "method": "traditional_ml"}
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


BACKENDS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKENDS_DIR.parent

for path in [
    BACKENDS_DIR / "rule_based",
    BACKENDS_DIR / "traditional_ml",
    BACKENDS_DIR / "deep_learning",
]:
    sys.path.insert(0, str(path))

from rule_based_model import predict as predict_rule_based  # noqa: E402
from predict import predict as predict_traditional_ml  # noqa: E402
from predict_bert import BertAbsaPredictor  # noqa: E402


ALLOWED_METHODS = {"rule_based", "traditional_ml", "bert"}
_bert_predictor: BertAbsaPredictor | None = None


def normalize_results(output: dict[str, Any], method: str) -> dict[str, Any]:
    results = output.get("results", [])
    if isinstance(results, dict):
        results = [
            {"aspect": aspect, "sentiment": sentiment}
            for aspect, sentiment in results.items()
        ]

    return {
        "review": output.get("review", ""),
        "method": method,
        "results": results,
    }


def analyze(review: str, method: str) -> dict[str, Any]:
    global _bert_predictor

    review = review.strip()
    if not review:
        raise ValueError("Review text must not be empty.")

    if method == "rule_based":
        return normalize_results(predict_rule_based(review), method)

    if method == "traditional_ml":
        return normalize_results(predict_traditional_ml(review), method)

    if method == "bert":
        if _bert_predictor is None:
            _bert_predictor = BertAbsaPredictor()
        return normalize_results(_bert_predictor.predict(review), method)

    raise ValueError(f"Unsupported method: {method}")


class AbsaRequestHandler(BaseHTTPRequestHandler):
    server_version = "ABSAHTTP/1.0"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "methods": sorted(ALLOWED_METHODS),
                },
            )
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/analyze":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            body = self._read_json_body()
            review = str(body.get("review", ""))
            method = str(body.get("method", "rule_based"))
            response = analyze(review, method)
        except Exception as exc:  # Keep API errors JSON-readable for Flutter.
            self._send_json(400, {"error": str(exc)})
            return

        self._send_json(200, response)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local ABSA API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AbsaRequestHandler)
    print(f"ABSA API running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ABSA API.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
