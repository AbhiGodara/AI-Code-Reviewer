#!/usr/bin/env python3
"""
Read each diff from /diffs, POST to the running API, and save ReviewReport JSON to /reviews.

Usage:
    # Start the server first:
    #   uvicorn app.main:app --reload
    #
    # Then in a second terminal:
    python run_reviews.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

# Windows consoles default to cp1252 and crash on non-ASCII output. Make stdout
# UTF-8 where possible; the markers below are ASCII anyway as a hard guarantee.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://localhost:8000"
DIFFS_DIR = Path(__file__).parent / "diffs"
REVIEWS_DIR = Path(__file__).parent / "reviews"

DIFFS = [
    {
        "file": "diff1_python.txt",
        "language": "python",
        "output": "diff1_review.json",
        "context": "PR adds a refund endpoint and fixes the transaction lookup logic in a payment service.",
    },
    {
        "file": "diff2_javascript.txt",
        "language": "javascript",
        "output": "diff2_review.json",
        "context": "PR adds a bulk user-fetch endpoint and updates the password reset flow in a Node.js/Express controller.",
    },
    {
        "file": "diff3_typescript.txt",
        "language": "typescript",
        "output": "diff3_review.json",
        "context": "PR adds order cancellation logic and a status polling mechanism to a TypeScript order service.",
    },
]


def run_review(diff_info: dict, client: httpx.Client) -> dict:
    diff_path = DIFFS_DIR / diff_info["file"]
    if not diff_path.exists():
        raise FileNotFoundError(f"Diff file not found: {diff_path}")

    diff_text = diff_path.read_text(encoding="utf-8")
    payload = {
        "diff": diff_text,
        "language": diff_info["language"],
        "context": diff_info["context"],
    }

    print(f"  -> POST {BASE_URL}/review  ({len(diff_text)} chars)", flush=True)
    response = client.post(f"{BASE_URL}/review", json=payload)
    response.raise_for_status()
    return response.json()


def main() -> None:
    REVIEWS_DIR.mkdir(exist_ok=True)

    # Generous timeout: 5 agents plus the merge can take a minute
    with httpx.Client(timeout=180.0) as client:
        # Quick health check before burning time
        try:
            health = client.get(f"{BASE_URL}/health", timeout=5.0)
            health.raise_for_status()
            print(f"Server is up. Groq configured: {health.json().get('groq_configured')}\n")
        except Exception as exc:
            print(f"ERROR: Server not reachable at {BASE_URL}: {exc}", file=sys.stderr)
            print("Start the server with:  uvicorn app.main:app --reload", file=sys.stderr)
            sys.exit(1)

        for diff_info in DIFFS:
            print(f"Reviewing {diff_info['file']} ({diff_info['language']})...")
            try:
                review = run_review(diff_info, client)
                output_path = REVIEWS_DIR / diff_info["output"]
                output_path.write_text(
                    json.dumps(review, indent=2, default=str), encoding="utf-8"
                )
                verdict = review.get("verdict", "?")
                severity = review.get("overall_severity", "?")
                n_findings = len(review.get("findings", []))
                print(f"  [ok] Saved -> {output_path.name}")
                print(f"    verdict={verdict}  severity={severity}  findings={n_findings}\n")
            except Exception as exc:
                print(f"  [FAILED] {exc}\n", file=sys.stderr)

    print("Done. Review reports saved to reviews/")


if __name__ == "__main__":
    main()
