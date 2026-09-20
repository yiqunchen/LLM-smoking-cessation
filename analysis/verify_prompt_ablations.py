#!/usr/bin/env python3
"""Verify that one Reviewer 3 model run has all canonical dt10-k7 rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data/splits" / "canonical" / "test_dt10_k7.json"
CONDITIONS = ("pp_cbtact", "full_pp_no_cbtact", "history_ratings_only", "history_text_only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    result_dir = (ROOT / args.results_dir).resolve()
    if ROOT not in result_dir.parents:
        parser.error("--results-dir must stay within the repository root")
    test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    expected = {str(index) for index in range(len(test))}
    manifest = result_dir / "manifest_dt10_k7.json"
    if not manifest.exists():
        raise SystemExit(f"Missing run manifest: {manifest}")
    for condition in CONDITIONS:
        path = result_dir / f"{condition}_dt10_k7.json"
        if not path.exists():
            raise SystemExit(f"Missing {condition} result: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, dict) or set(rows) != expected:
            raise SystemExit(f"Incomplete {condition}: {len(rows) if isinstance(rows, dict) else 'invalid'}/{len(test)}")
        for index, item in enumerate(test):
            row = rows[str(index)]
            if row.get("response_id") != item.get("response_id") or row.get("input_message") != item.get("input_message"):
                raise SystemExit(f"{condition} row {index} is not aligned to dt10-k7")
    model = json.loads(manifest.read_text(encoding="utf-8")).get("model", "unknown")
    print(f"PASS: {model}; four conditions x {len(test)} canonical dt10-k7 rows.")


if __name__ == "__main__":
    main()
