#!/usr/bin/env python3
"""Fail fast if the published figure directories contain retired PP artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_META = ROOT / "data_splits/canonical/metadata_dt10_k7.json"
PRIMARY_TRAIN = ROOT / "data_splits/canonical/train_dt10_k7.json"
PRIMARY_TEST = ROOT / "data_splits/canonical/test_dt10_k7.json"
PUBLISHED_DIRS = [ROOT / "figures", ROOT / "revision/figures"]
RETIRED_MARKERS = ("7030", "dt7030", "digital_twin_70")
FIGURE2_SOURCE = ROOT / "figures/figure2/figure2_dt10_source_table.csv"
FIGURE4_SOURCE = ROOT / "figures/figure4/supporting_message_selection_methods_k7.csv"
STRICT_SOURCE = ROOT / "revision/figures/progress_summary/ai_vs_ml_strict_fixed_feature.csv"


def main() -> None:
    with PRIMARY_META.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    primary_k = metadata.get("k_train")
    primary_n = metadata.get("test_samples")
    with PRIMARY_TRAIN.open(encoding="utf-8") as handle:
        train_rows = json.load(handle)
    with PRIMARY_TEST.open(encoding="utf-8") as handle:
        test_rows = json.load(handle)
    split_k = int(PRIMARY_META.stem.rsplit("k", 1)[1])
    allowed_k = {
        json.loads(path.read_text(encoding="utf-8"))["k_train"]
        for path in PRIMARY_META.parent.glob("metadata_dt10_k*.json")
    }
    if (primary_k != split_k or len(train_rows) != metadata.get("train_samples") or
            len(test_rows) != primary_n):
        raise SystemExit("Primary dt10 metadata and split files disagree.")

    retired = []
    for directory in PUBLISHED_DIRS:
        for path in directory.rglob("*"):
            if path.is_file() and any(marker in path.name.lower() for marker in RETIRED_MARKERS):
                retired.append(path.relative_to(ROOT))
    if retired:
        raise SystemExit("Retired published artifacts found:\n" + "\n".join(map(str, retired)))

    figure2 = pd.read_csv(FIGURE2_SOURCE)
    if set(figure2["split"]) != {f"dt10_k{primary_k}"}:
        raise SystemExit("Figure 2 source is not exclusively the declared primary dt10 split.")
    figure4 = pd.read_csv(FIGURE4_SOURCE)
    if set(figure4["k_train"]) != {primary_k}:
        raise SystemExit("Figure 4 supporting source is not exclusively the declared primary history size.")
    strict = pd.read_csv(STRICT_SOURCE)
    aggregate_primary = strict[(strict["k_train"] == primary_k) &
                                (strict["metric"].isin(["Accuracy", "Macro-F1", "QWK"]))]
    if (set(strict["k_train"]) - allowed_k or aggregate_primary.empty or
            set(aggregate_primary["N_min"]) != {primary_n}):
        raise SystemExit("Strict summary does not match the canonical dt10 design.")

    print(f"PASS: published artifacts match dt10 k={primary_k}, N_test={primary_n}.")


if __name__ == "__main__":
    main()
