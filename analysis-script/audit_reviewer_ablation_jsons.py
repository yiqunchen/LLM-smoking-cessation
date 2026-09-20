#!/usr/bin/env python3
"""Row-level integrity audit for current Reviewer 3 ablation result JSONs.

Unlike a count-only checkpoint check, this validates every saved prediction
against the canonical dt10-k7 held-out record.  Incomplete DeepSeek files are
reported as VALID_PARTIAL rather than treated as results eligible for analysis.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data_splits" / "canonical" / "test_dt10_k7.json"
OUTDIR = ROOT / "revision" / "figures" / "reviewer_ablations_dt10"
CONDITIONS = ("pp_cbtact", "full_pp_no_cbtact", "history_ratings_only", "history_text_only")
RATING_DOMAINS = ("content", "coping", "quitting")
ALLOWED_LABELS = {
    "content": {"Very poor", "Poor", "Acceptable", "Good", "Very good"},
    "coping": {"Not at all helpful", "Somewhat helpful", "Moderately helpful", "Very helpful", "Extremely helpful"},
    "quitting": {"Not at all helpful", "Somewhat helpful", "Moderately helpful", "Very helpful", "Extremely helpful"},
}
MODELS = (
    ("GPT-4o-mini", "results_reviewer_ablations_openai_gpt-4o-mini", "openai/gpt-4o-mini"),
    ("GPT-5", "results_reviewer_ablations_openai_gpt-5", "openai/gpt-5"),
    ("DeepSeek-R1", "results_reviewer_ablations_deepseek-r1", "deepseek/deepseek-r1-0528"),
    ("Grok-4.3", "results_reviewer_ablations_x-ai_grok-4.3", "x-ai/grok-4.3"),
    ("Gemini-2.5-Pro", "results_reviewer_ablations_google_gemini-2.5-pro", "google/gemini-2.5-pro"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_file(path: Path, test: list[dict], condition: str) -> tuple[int, list[str]]:
    """Return saved-row count and all integrity violations for one condition file."""
    errors: list[str] = []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return 0, [f"unparseable JSON: {type(exc).__name__}: {exc}"]
    if not isinstance(rows, dict):
        return 0, ["top-level JSON is not an object keyed by canonical row index"]
    expected_keys = {str(index) for index in range(len(test))}
    keys = set(rows)
    unexpected = keys - expected_keys
    if unexpected:
        errors.append(f"unexpected row keys: {sorted(unexpected, key=int)[:5]}")
    for key, row in rows.items():
        if key not in expected_keys:
            continue
        if not isinstance(row, dict):
            errors.append(f"row {key}: not a JSON object")
            continue
        reference = test[int(key)]
        if row.get("response_id") != reference.get("response_id"):
            errors.append(f"row {key}: response_id mismatch")
        if row.get("input_message") != reference.get("input_message"):
            errors.append(f"row {key}: input_message mismatch")
        if row.get("condition") != condition:
            errors.append(f"row {key}: condition={row.get('condition')!r}")
        if row.get("n_profile_messages") != 7:
            errors.append(f"row {key}: n_profile_messages={row.get('n_profile_messages')!r}")
        ratings = reference.get("ratings", {}) or {}
        for domain in RATING_DOMAINS:
            if row.get(f"ground_truth_{domain}") != ratings.get(domain):
                errors.append(f"row {key}: ground_truth_{domain} mismatch")
            if row.get(f"predicted_{domain}") not in ALLOWED_LABELS[domain]:
                errors.append(f"row {key}: invalid predicted_{domain}={row.get(f'predicted_{domain}')!r}")
        if len(errors) >= 20:
            errors.append("additional violations omitted after first 20")
            break
    return len(rows), errors


def main() -> None:
    test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    if not test:
        raise SystemExit("Canonical test file is empty")
    expected_hash = sha256(TEST_PATH)
    report: list[dict] = []
    failures: list[str] = []
    for model, relative_dir, expected_model in MODELS:
        directory = ROOT / relative_dir
        manifest_path = directory / "manifest_dt10_k7.json"
        manifest_error = ""
        if not manifest_path.exists():
            manifest_error = "missing manifest"
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("model") != expected_model:
                    manifest_error = f"manifest model={manifest.get('model')!r}"
                elif manifest.get("test_sha256") != expected_hash:
                    manifest_error = "manifest canonical-test SHA-256 mismatch"
            except Exception as exc:
                manifest_error = f"invalid manifest: {type(exc).__name__}: {exc}"
        for condition in CONDITIONS:
            path = directory / f"{condition}_dt10_k7.json"
            if not path.exists():
                n_rows, errors = 0, ["missing condition file"]
            else:
                n_rows, errors = audit_file(path, test, condition)
            if manifest_error:
                errors.insert(0, manifest_error)
            complete = n_rows == len(test)
            if errors:
                status = "FAIL"
                failures.append(f"{model} / {condition}: {'; '.join(errors[:3])}")
            elif complete:
                status = "PASS"
            else:
                status = "VALID_PARTIAL"
            report.append({
                "model": model, "condition_key": condition, "n_saved_rows": n_rows,
                "n_expected_rows": len(test), "complete": complete, "status": status,
                "file_sha256": sha256(path) if path.exists() else "", "violations": "; ".join(errors),
            })
    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / "reviewer_ablation_json_integrity_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    status_counts = {status: sum(row["status"] == status for row in report) for status in ("PASS", "VALID_PARTIAL", "FAIL")}
    markdown = [
        "# Reviewer-ablation JSON integrity audit",
        "",
        f"Canonical test: {len(test)} rows; SHA-256: `{expected_hash}`.",
        "",
        f"- PASS: {status_counts['PASS']} complete model-condition files",
        f"- VALID_PARTIAL: {status_counts['VALID_PARTIAL']} DeepSeek checkpoint files (not eligible for analysis)",
        f"- FAIL: {status_counts['FAIL']} files",
        "",
        "Each saved row was checked for its canonical index, participant, message text, ground truth, allowed prediction label, condition identity, and seven-message profile count.",
    ]
    (OUTDIR / "reviewer_ablation_json_integrity_audit.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Audit: {status_counts['PASS']} complete PASS; {status_counts['VALID_PARTIAL']} valid partial; {status_counts['FAIL']} FAIL.")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
