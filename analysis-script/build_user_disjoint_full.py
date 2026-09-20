#!/usr/bin/env python3
"""Build user-disjoint splits that keep every rated message per user.

Two splits, **nested** (the 10/90 train users are a strict subset of the
30/70 train users so the OOD comparison ladders cleanly):

  - 30/70: 90 train users / 211 test users  (reuses participant_3070 partition)
  - 10/90: 30 train users / 271 test users  (subset of 30/70 train, with
            seed=42 random pick; remaining 60 of the 90 + 211 originals = 271 test)

Each split keeps every (user, message) tuple in the dt10 union (~10 msgs/user),
not the ~3 msgs/user the canonical participant_3070 used.

Outputs
-------
    data_splits/canonical/train_user_disjoint_3070_full.json
    data_splits/canonical/test_user_disjoint_3070_full.json
    data_splits/canonical/train_user_disjoint_1090_full.json
    data_splits/canonical/test_user_disjoint_1090_full.json

Usage:
    uv run python analysis-script/build_user_disjoint_full.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = PROJECT_ROOT / "data_splits" / "canonical"

# Source: union all dt10 rows (across k=1..7 train and test) and dedupe
# on (response_id, input_message).
DT10_FILES = [
    f"{kind}_dt10_k{k}.json"
    for kind in ("train", "test")
    for k in (1, 3, 5, 7)
]

# Reuse the exact user partition from participant_3070 to maintain
# narrative continuity with the existing analysis.
TRAIN_PARTITION = CANONICAL / "train_participant_3070.json"
TEST_PARTITION  = CANONICAL / "test_participant_3070.json"


def main():
    # 1. Pull the 90 / 211 user partition.
    with open(TRAIN_PARTITION) as f:
        tr_part = json.load(f)
    with open(TEST_PARTITION) as f:
        te_part = json.load(f)
    train_users = {r["response_id"] for r in tr_part}
    test_users  = {r["response_id"] for r in te_part}
    print(f"partition: train users={len(train_users)}, test users={len(test_users)}, "
          f"intersection={len(train_users & test_users)}")
    assert not (train_users & test_users), "user partition is not disjoint"

    # 2. Union dt10 rows, dedupe on (response_id, input_message).
    seen = set()
    pooled: list[dict] = []
    for fname in DT10_FILES:
        path = CANONICAL / fname
        if not path.exists():
            print(f"  skip (missing): {fname}")
            continue
        with open(path) as f:
            rows = json.load(f)
        n_added = 0
        for r in rows:
            key = (r.get("response_id"), r.get("input_message"))
            if key in seen or None in key:
                continue
            seen.add(key)
            pooled.append(r)
            n_added += 1
        print(f"  +{n_added:>4} rows from {fname}  (running total {len(pooled)})")

    # 3. Stats.
    by_user = defaultdict(list)
    for r in pooled:
        by_user[r["response_id"]].append(r)
    n_users = len(by_user)
    avg_per_user = sum(len(v) for v in by_user.values()) / max(1, n_users)
    print(f"\npooled: {len(pooled)} rows  |  users={n_users}  |  rows/user={avg_per_user:.2f}")

    # 4. Filter to the 30/70 partition.
    train_rows_3070 = [r for r in pooled if r["response_id"] in train_users]
    test_rows_3070  = [r for r in pooled if r["response_id"] in test_users]
    print(f"\nUSER-DISJOINT 30/70 split:")
    print(f"  train: {len(train_rows_3070):>5} rows  ({len(train_users)} users)")
    print(f"  test:  {len(test_rows_3070):>5} rows  ({len(test_users)} users)")

    # 5. Build NESTED 10/90 split: 30 train users sampled (seed=42) from the
    #    90 train users in 30/70; the remaining 60 + the original 211 test
    #    users go to test (271 total).
    rng = random.Random(42)
    train_users_1090 = set(rng.sample(sorted(train_users), 30))
    test_users_1090 = (train_users - train_users_1090) | test_users
    train_rows_1090 = [r for r in pooled if r["response_id"] in train_users_1090]
    test_rows_1090  = [r for r in pooled if r["response_id"] in test_users_1090]
    print(f"\nUSER-DISJOINT 10/90 split (nested in 30/70):")
    print(f"  train: {len(train_rows_1090):>5} rows  ({len(train_users_1090)} users)")
    print(f"  test:  {len(test_rows_1090):>5} rows  ({len(test_users_1090)} users)")
    assert train_users_1090 < train_users, "10/90 train should be subset of 30/70 train"
    assert not (train_users_1090 & test_users_1090), "10/90 user partition not disjoint"

    # 6. Save all four files.
    pairs = [
        ("3070", train_rows_3070, test_rows_3070),
        ("1090", train_rows_1090, test_rows_1090),
    ]
    for tag, tr_rows, te_rows in pairs:
        out_train = CANONICAL / f"train_user_disjoint_{tag}_full.json"
        out_test  = CANONICAL / f"test_user_disjoint_{tag}_full.json"
        with open(out_train, "w") as f: json.dump(tr_rows, f, indent=2)
        with open(out_test,  "w") as f: json.dump(te_rows, f, indent=2)
        print(f"\nwrote {out_train.relative_to(PROJECT_ROOT)}")
        print(f"wrote {out_test.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
