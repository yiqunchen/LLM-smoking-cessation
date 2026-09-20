# Legacy analysis code (frozen)

Scripts from the original manuscript submission and early revision drafts.
Most of them read the retired 70/30-family splits or four-outcome result files
that are no longer in the repository (see `docs/DATA_POLICY.md`), so they are
kept for provenance only and are **not maintained or expected to run**.

Active code lives one level up in `analysis/`. If you must run something here:

```bash
PYTHONPATH=analysis .venv/bin/python analysis/legacy/<script>.py
```

Do not add new work to this directory.
