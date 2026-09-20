#!/usr/bin/env python3
"""
Apples-to-apples variance workflow for the dt7030 comparison.

This workflow:
- waits for the fresh generic dt7030 outputs to appear;
- computes the exact shared subset on the digital-twin split;
- writes that subset so all variance reruns can use the same messages;
- summarizes repeated-run variance on a chosen domain.

Default domain: Content.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from revision_utils import MODEL_CONFIGS, load_results_aligned


DOMAINS = ["content", "coping", "quitting"]
RATING_MAP = {
    "Very poor": 1,
    "Poor": 2,
    "Acceptable": 3,
    "Good": 4,
    "Very good": 5,
    "Not at all helpful": 1,
    "Somewhat helpful": 2,
    "Moderately helpful": 3,
    "Very helpful": 4,
    "Extremely helpful": 5,
    "Not Helpful": 1,
}

GENERIC_DT7030_METHOD_FILES = {
    "Zero-shot (all)": "generic_llm_1_zero_shot_dt7030.json",
    "Zero-shot (select)": "generic_llm_2_zero_shot_select_dt7030.json",
    "Few-shot (all)": "generic_llm_3_few_shot_dt7030.json",
    "Few-shot (select)": "generic_llm_4_few_shot_select_dt7030.json",
}
PERSONALIZED_METHOD_FILES = {
    "PP": "digital_twin_4_cbtact_7030.json",
    "Hybrid RF+PP": "hybrid",
}

MODEL_PROVIDER = {
    "gpt-4o-mini": "openai",
    "gpt-5": "openai",
    "deepseek_deepseek-r1-0528": "openrouter",
    "x-ai_grok-4-fast": "openrouter",
    "gemini-2.5-pro": "gemini",
}

METHOD_PROMPT_CONFIG = {
    "Zero-shot (all)": "zero-shot",
    "Zero-shot (select)": "zero-shot-feature-select",
    "Few-shot (all)": "few-shot",
    "Few-shot (select)": "few-shot-feature-select",
}

TARGET_DATA_FILE = Path("data_splits/canonical/test_digital_twin_7030.json")
COMMON_SUBSET_DIR = Path("revision/figures")


def normalize_message(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def item_key(row: pd.Series) -> str:
    ratings = row.get("ratings", {}) if isinstance(row, dict) else row.get("ratings", {})
    if not isinstance(ratings, dict):
        ratings = {}

    def _rating_value(name: str) -> str:
        ground_truth_field = row.get(f"ground_truth_{name}", None)
        if ground_truth_field not in [None, "", "NA"]:
            return str(ground_truth_field)
        return str(ratings.get(name, "NA"))

    return "||".join([
        str(row.get("response_id", "")),
        normalize_message(row.get("input_message", "")),
        _rating_value("content"),
        _rating_value("design"),
        _rating_value("coping"),
        _rating_value("quitting"),
    ])


def attach_item_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Item_Key"] = out.apply(item_key, axis=1)
    return out.drop_duplicates(subset=["Item_Key"]).reset_index(drop=True)


def expected_dt7030_inputs() -> list[Path]:
    """Fresh dt7030 inputs that this watcher should wait on."""
    paths: list[Path] = []
    for cfg in MODEL_CONFIGS.values():
        for method_file in GENERIC_DT7030_METHOD_FILES.values():
            paths.append(Path(cfg["dir"]) / method_file)
    return paths


def wait_for_dt7030_inputs(poll_seconds: int = 60) -> None:
    expected = expected_dt7030_inputs()
    while True:
        missing = [path for path in expected if not path.exists()]
        if not missing:
            print(f"All expected dt7030 inputs are present ({len(expected)} files).")
            return
        print(f"Waiting on {len(missing)} dt7030 inputs...")
        for path in missing[:10]:
            print(f"  missing: {path}")
        time.sleep(poll_seconds)


def _load_required_results() -> dict[tuple[str, str], pd.DataFrame]:
    raw_results: dict[tuple[str, str], pd.DataFrame] = {}
    for model_id, cfg in MODEL_CONFIGS.items():
        model_name = cfg["display"]
        for method_name, method_file in {
            **GENERIC_DT7030_METHOD_FILES,
            **PERSONALIZED_METHOD_FILES,
        }.items():
            df = load_results_aligned(model_id, method_file)
            if df is None:
                raise FileNotFoundError(
                    f"Missing required result file for {model_name} / {method_name}: {method_file}"
                )
            raw_results[(model_name, method_name)] = attach_item_keys(df)
    return raw_results


def collect_common_subset(domain: str) -> tuple[pd.DataFrame, set[str]]:
    domain = domain.lower()
    raw_results = _load_required_results()

    shared_sets: list[set[str]] = []
    for df in raw_results.values():
        gt_col = f"gt_{domain}_num"
        pred_col = f"pred_{domain}_num"
        valid = df[gt_col].notna() & df[pred_col].notna()
        shared_sets.append(set(df.loc[valid, "Item_Key"]))

    shared = set.intersection(*shared_sets) if shared_sets else set()

    rows = []
    for (model_name, method_name), df in raw_results.items():
        gt_col = f"gt_{domain}_num"
        pred_col = f"pred_{domain}_num"
        valid = df[gt_col].notna() & df[pred_col].notna()
        rows.append(
            {
                "Model": model_name,
                "Method": method_name,
                "Domain": domain.capitalize(),
                "N_Available": int(valid.sum()),
                "N_Common": len(shared),
            }
        )

    return pd.DataFrame(rows), shared


def build_common_subset_file(domain: str, output_path: Path) -> pd.DataFrame:
    manifest_df, shared_keys = collect_common_subset(domain)

    with TARGET_DATA_FILE.open("r", encoding="utf-8") as handle:
        source_items = json.load(handle)
    subset_items = []
    for item in source_items:
        if not isinstance(item, dict):
            continue
        if item_key(pd.Series(item)) in shared_keys:
            subset_items.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(subset_items, handle, indent=2)

    return manifest_df


def parse_result_file(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    rows = [
        item
        for item in results.values()
        if isinstance(item, dict) and item.get("predicted_content") != "ERROR"
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["Item_Key"] = df.apply(item_key, axis=1)
    return df


def summarize_variance(pattern: str, domain: str) -> pd.DataFrame:
    paths = sorted(Path().glob(pattern))
    rows = []
    gt_map = RATING_MAP
    for path in paths:
        df = parse_result_file(path)
        if df.empty:
            continue
        rep_match = re.search(r"_rep(\d+)\.json$", path.name)
        rep = int(rep_match.group(1)) if rep_match else -1
        model_dir = path.parent.name
        method_name = "Unknown"
        for display_name, stem in GENERIC_DT7030_METHOD_FILES.items():
            if stem.replace(".json", "") in path.name:
                method_name = display_name
                break
        gt_col = f"ground_truth_{domain}"
        pred_col = f"predicted_{domain}"
        valid = df[gt_col].notna() & df[pred_col].notna()
        df = df.loc[valid].copy()
        if df.empty:
            continue
        gt = df[gt_col].map(gt_map)
        pred = df[pred_col].map(gt_map)
        mask = gt.notna() & pred.notna()
        gt = gt.loc[mask].astype(int)
        pred = pred.loc[mask].astype(int)
        rows.append(
            {
                "File": str(path),
                "Model_Dir": model_dir,
                "Method": method_name,
                "Rep": rep,
                "N": int(len(gt)),
                "Accuracy": float(accuracy_score(gt, pred)),
                "F1": float(f1_score(gt, pred, average="macro", zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    subset = sub.add_parser("subset", help="Report the exact common subset")
    subset.add_argument("--domain", default="content", choices=DOMAINS)
    subset.add_argument("--wait", action="store_true", help="Wait until dt7030 inputs exist")
    subset.add_argument("--save", type=str, default=None, help="Optional path to save the exact common subset JSON")

    summarize = sub.add_parser("summarize", help="Summarize repeated-run variance")
    summarize.add_argument("--pattern", required=True, help="Glob pattern for repeated-run JSON files")
    summarize.add_argument("--domain", default="content", choices=DOMAINS)

    args = parser.parse_args()

    if args.cmd == "subset":
        if args.wait:
            wait_for_dt7030_inputs()
        if args.save:
            df = build_common_subset_file(args.domain, Path(args.save))
        else:
            df, _ = collect_common_subset(args.domain)
        manifest_path = COMMON_SUBSET_DIR / f"dt7030_common_subset_manifest_{args.domain}.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(manifest_path, index=False)
        print(f"Saved manifest to {manifest_path}")
        if args.save:
            print(f"Saved common subset JSON to {args.save}")
        print(df.to_string(index=False))
    elif args.cmd == "summarize":
        df = summarize_variance(args.pattern, args.domain)
        if df.empty:
            print("No repeated-run files found.")
        else:
            print(df.to_string(index=False))
            print()
            print(df.groupby(["Model_Dir", "Method"])[["Accuracy", "F1"]].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
