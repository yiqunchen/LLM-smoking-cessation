# LLM-Based Smoking Cessation Message Evaluation

Predicting participant ratings of smoking-cessation messages with LLMs
(generic prompts, personalized prompts (PP), hybrid RF+PP) and supervised
baselines. This branch holds the **revision deliverables** for the reviewer
response.

## Policy (read first)

- **Splits:** every PP result, table, and figure uses the ten-message `dt10`
  splits in `data_splits/canonical/`. Primary analysis = `dt10_k7`
  (7 history messages per participant, 898 held-out ratings, 301 participants).
  `k1/k3/k5` exist only for learning-curve sensitivity.
  The old 70/30 (`digital_twin_7030`, `1090/3070/5050/9010`, `participant_3070`)
  splits and everything derived from them are **retired**: not tracked, not
  plotted, not cited. Copies live outside Git under `archive/retired_*`.
- **Outcomes:** three domains only: **content, coping, quitting**. The
  `design` rating is excluded from all revision analyses and never appears in
  reviewer-ablation outputs. (`design` is still a raw field in the split JSONs
  and the original manuscript-era scripts; that is source data, not a result.)
- **No placeholders:** never fabricate or ship mock rows. A missing result is
  reported as missing with the rerun command.
- Details: `revision/DT10_SOURCE_OF_TRUTH.md`, `data_splits/canonical/README.md`,
  figure style in `agents.md`.

## Repository layout

| Path | What it is | Tracked |
|---|---|---|
| `analysis-script/` | All Python: runners, metrics, bootstrap, plotting | yes |
| `scripts/` | Shell entry points for the reviewer-ablation pipeline | yes |
| `data_splits/canonical/` | dt10 `k1/k3/k5/k7` train/test/metadata JSON | yes |
| `results_reviewer_ablations_<model>/` | Reviewer 3 prompt-ablation predictions (4 conditions x 898 rows + manifest) | yes |
| `results_manuscript_x-ai_grok-4-fast/digital_twin_dt10_k*.json` | Grok PP runs feeding the dt10 learning-curve / progress summary | yes |
| `results_manuscript_hybrid_rf_grok4/rf_predictions_all_features.json` | RF predictions used by hybrid figures | yes |
| `revision/*.md`, `revision/*.docx` | Reviewer responses, edit maps, narrative | yes |
| `revision/figures/progress_summary/` | Strict shared-row dt10 tables and figures | yes |
| `revision/figures/reviewer_ablations_dt10/` | Reviewer 3 ablation figures, CSV tables, bootstrap CIs | yes (cache ignored) |
| `figures/figure2/`, `figures/figure4/` | Main-text figure sources (dt10) | yes |
| `archive/data/` | Archival raw inputs (restricted; only embeddings tracked) | mostly no |
| `archive/retired_splits/`, `archive/retired_results/` | Retired 70/30-family artifacts, kept only so nothing is lost | no |
| `results_manuscript_*/` (other) | Manuscript-era runs not used by revision figures | no |
| `logs/` | Run logs | no |

Manuscript-era pipelines (`e2e_pipeline.sh`, `run_full_manuscript_pipeline.sh`,
`run_hybrid_rf_all_models.sh`, `Makefile`) are kept for provenance of the
original submission; they are not part of the revision workflow.

## Setup

```bash
uv sync                      # creates .venv (Python 3.12)
export OPENROUTER_API_KEY=... # all reviewer-ablation models run via OpenRouter
```

## Reviewer 3 prompt ablations (what to run)

Four prompt conditions, five models, canonical `dt10_k7`, 898 rows each:

| Condition | Participant metadata | 7 history texts | History ratings | CBT/ACT labels |
|---|:-:|:-:|:-:|:-:|
| `pp_cbtact` (baseline PP) | x | x | x | x |
| `full_pp_no_cbtact` | x | x | x | |
| `history_ratings_only` | | x | x | |
| `history_text_only` | | x | | |

Models: GPT-4o-mini, GPT-5, DeepSeek-R1 (0528), Grok-4.3, Gemini-2.5-Pro.

```bash
# 1. Launch / resume all runs (one tmux session per model x condition,
#    checkpointed, safe to re-run until every condition reports 898/898).
bash scripts/run_reviewer_ablations.sh          # or: ... gpt5 grok
tmux ls | grep rerun- ; tail -c 300 logs/rerun-*.log

# 2. Verify -> audit -> clustered bootstrap -> figures -> Word doc -> ZIP,
#    plus timestamped copies in ~/Downloads/.  Refuses incomplete inputs.
bash scripts/build_reviewer_ablation_package.sh
```

Per-model runner flags are set inside `scripts/run_reviewer_ablations.sh`
(GPT-5 / DeepSeek-R1: `--reasoning-effort low` with a larger output budget;
Gemini 2.5 Pro: `--max-output-tokens 0`, i.e. uncapped, because its hidden
thinking otherwise exhausts `max_tokens` and returns empty content).

Individual steps, if needed:

```bash
.venv/bin/python analysis-script/run_dt10_reviewer_ablations_grok.py --help
.venv/bin/python analysis-script/verify_reviewer_ablation_completion.py --results-dir results_reviewer_ablations_openai_gpt-5
.venv/bin/python analysis-script/audit_reviewer_ablation_jsons.py
.venv/bin/python analysis-script/bootstrap_dt10_reviewer_ablations.py   # --force to recompute caches
.venv/bin/python analysis-script/plot_dt10_reviewer_ablations_all_models.py
.venv/bin/python analysis-script/create_reviewer_bootstrap_word_doc.py
```

## Other revision figures

```bash
.venv/bin/python analysis-script/lc_dt10_rf.py                  # dt10 RF learning curves
.venv/bin/python analysis-script/build_progress_summary.py      # revision/figures/progress_summary/
.venv/bin/python analysis-script/make_figure4_supporting_dt10.py # figures/figure4/*_dt10.*
```

`revision/figures/README.md` maps each retained figure to its source CSV and
script.

## Metrics

Accuracy, macro-F1, Cohen's kappa, quadratic weighted kappa (QWK), and
Spearman's rho, each per domain. Reviewer-ablation CIs are participant-clustered
bootstraps (2000 resamples, seed 20260919).
