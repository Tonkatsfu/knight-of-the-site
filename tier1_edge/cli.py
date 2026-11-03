"""Command-line interface for the Tier 1 Edge URL Classifier.

Usage:
    python -m tier1_edge.cli https://www.google.com/
    python -m tier1_edge.cli http://192.168.10.5:8080/paypal/login@verify//secure --json
    python -m tier1_edge.cli --sync
    python -m tier1_edge.cli --flush-queue
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from tier1_edge.inference_client import DEFAULT_TIER2_URL, ModelSyncError, Tier1EdgeClient

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tier1_v1.onnx"

_VALUE_LABELS = {1: "Legitimate (+1)", 0: "Suspicious ( 0)", -1: "Phishing   (-1)"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tier1_edge.cli",
        description="Knight of the Site — Tier 1 Mobile/Edge ONNX URL Scanner",
    )
    parser.add_argument("url", nargs="?", help="URL to classify")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to Tier 1 .onnx model")
    parser.add_argument("--tier2-url", default=DEFAULT_TIER2_URL, help="Tier 2 Cloud base URL")
    parser.add_argument("--report", action="store_true", help="Force escalation to Tier 2 as user-reported")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON output")
    parser.add_argument("--sync", action="store_true", help="Check Tier 2 for an OTA ONNX model update")
    parser.add_argument("--flush-queue", action="store_true", help="Flush queued offline escalations to Tier 2")
    return parser


def _print_human(result: dict) -> None:
    verdict = result["verdict"].upper()
    print(f"URL           : {result['url']}")
    print(f"Verdict       : {verdict} (source: {result['verdict_source']})")
    print(f"Routing       : {result['decision']} | P(phishing) = {result['p_phishing']:.4f}")
    print(
        f"Latency       : extract={result['extract_ms']:.3f} ms | "
        f"onnx={result['inference_ms']:.3f} ms | network={result['network_ms']:.2f} ms"
    )
    print(f"Model Version : {result['model_version']}")
    print("9 Lexical Features:")
    for name, val in result["features"].items():
        print(f"  - {name:26s}: {_VALUE_LABELS.get(val, str(val))}")


def main(argv: Sequence[str] | None = None, *, client: Tier1EdgeClient | None = None) -> int:
    args = _build_parser().parse_args(argv)
    owns_client = client is None
    edge = client or Tier1EdgeClient(args.model, tier2_url=args.tier2_url)
    try:
        if args.sync:
            try:
                outcome = edge.sync_model()
            except ModelSyncError as exc:
                print(f"OTA sync failed: {exc}", file=sys.stderr)
                return 1
            print(json.dumps(outcome) if args.as_json else f"OTA sync: {outcome}")
            if not args.url:
                return 0

        if args.flush_queue:
            delivered = edge.flush_offline_queue()
            msg = {"delivered": delivered, "pending": edge.pending_queue_size()}
            print(json.dumps(msg) if args.as_json else f"Queue flush: {msg}")
            if not args.url:
                return 0

        if not args.url:
            print("Provide a URL or --sync / --flush-queue.", file=sys.stderr)
            return 2

        result = edge.evaluate_url(args.url, user_reported=args.report)
        if args.as_json:
            print(json.dumps(result))
        else:
            _print_human(result)
        return 0
    finally:
        if owns_client:
            edge.close()


if __name__ == "__main__":
    raise SystemExit(main())
