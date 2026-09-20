# LLM-Based Smoking Cessation Message Evaluation

Can a large language model predict how an individual smoker will rate a
cessation message? This repository evaluates that question with three
families of models on a dataset of participant-rated messages:

- **Generic LLM prompting** – zero-/few-shot, no participant information.
- **Personalized prompting (PP)** – the model sees a participant's
  characteristics and their ratings of past messages, then rates a new one.
- **Supervised baselines** – logistic regression / random forest on
  participant features, rating history, and message embeddings; plus
  hybrid RF+PP variants.

Outcomes are three 5-point ratings per message: **content** quality,
helpfulness for **coping** with cravings, and helpfulness for **quitting**.
Metrics: accuracy, macro-F1, Cohen's kappa, quadratic weighted kappa (QWK),
Spearman's rho, with participant-clustered bootstrap CIs.

Agents and contributors: read `AGENTS.md` first.

## Layout

```
analysis/       active Python (one script per step); analysis/legacy/ is frozen
scripts/        shell entry points
data/           splits/canonical (dt10 splits, tracked) · raw/ (restricted)
results/        prompt_ablations/<model>/ (tracked) · manuscript/ (dt10 Grok PP tracked)
figures/        figures + source CSVs; figures/README.md maps figure -> data -> script
reports/        generated Word report + ZIP for the prompt ablations
docs/           DATA_POLICY.md · prompt_templates.md · revision/ (reviewer-response drafts)
archive/        gitignored: retired splits/results/code
```

## Data

`data/splits/canonical/` holds the **dt10** splits: every participant's ten
rated messages are divided within-participant into `k` history messages
(training / personalization context) and the remainder held out.

| Split | History per participant | Held-out rows | Use |
|---|---:|---:|---|
| `dt10_k7` | 7 | 898 | **primary** evaluation (301 participants) |
| `dt10_k1`, `k3`, `k5` | 1 / 3 / 5 | split-specific | learning-curve sensitivity |

`metadata_dt10_k*.json` records seeds, counts, and SHA-256s. Earlier 70/30
splits from a three-message pool are retired and not in the repository
(`docs/DATA_POLICY.md`). Raw participant-level inputs are restricted and
gitignored under `data/raw/`.

## Setup

```bash
uv sync                        # Python 3.12 -> .venv
export OPENROUTER_API_KEY=...  # every model is called through OpenRouter
```

## Prompt ablations (main pipeline)

Which parts of a personalized prompt carry the signal? Four conditions, five
models (GPT-4o-mini, GPT-5, DeepSeek-R1, Grok-4.3, Gemini-2.5-Pro), all 898
`dt10_k7` held-out rows each:

| Condition | Participant metadata | 7 history texts | History ratings | CBT/ACT labels |
|---|:-:|:-:|:-:|:-:|
| `pp_cbtact` (full PP) | x | x | x | x |
| `full_pp_no_cbtact` | x | x | x | |
| `history_ratings_only` | | x | x | |
| `history_text_only` | | x | | |

```bash
# 1. Launch or resume every model x condition (tmux, checkpointed, idempotent).
bash scripts/run_prompt_ablations.sh            # subset: ... gpt5 grok
tmux ls | grep rerun- ; tail -c 300 logs/rerun-*.log

# 2. Verify -> audit -> clustered bootstrap -> figures -> Word report -> ZIP
#    (+ timestamped copies in ~/Downloads). Refuses incomplete inputs.
bash scripts/build_prompt_ablation_package.sh
```

Outputs: `results/prompt_ablations/<model>/<condition>_dt10_k7.json` (one
record per held-out row: ground truth, predictions, explanation) and
`manifest_dt10_k7.json`; `figures/prompt_ablations/` (figures, per-metric CSV
tables, bootstrap CIs); `reports/prompt_ablation_results.docx`.

Step by step, if you need one piece:

```bash
.venv/bin/python analysis/run_prompt_ablations.py --help       # runner (per model/condition flags live in scripts/run_prompt_ablations.sh)
.venv/bin/python analysis/verify_prompt_ablations.py --results-dir results/prompt_ablations/gpt-5
.venv/bin/python analysis/audit_prompt_ablations.py            # row-level integrity vs canonical test set
.venv/bin/python analysis/bootstrap_prompt_ablations.py        # --force recomputes cached CIs
.venv/bin/python analysis/plot_prompt_ablations.py
.venv/bin/python analysis/report_prompt_ablations_docx.py
```

## Other analyses

```bash
.venv/bin/python analysis/build_dt10_splits.py                 # rebuild data/splits/canonical from data/raw
.venv/bin/python analysis/run_dt10_grok.py --help              # PP runs on dt10 k1/k3/k5/k7 -> results/manuscript/x-ai_grok-4-fast
.venv/bin/python analysis/lc_dt10_rf.py                        # supervised learning curves over k
.venv/bin/python analysis/build_progress_summary.py            # figures/progress_summary: strict shared-row LLM-PP vs RF
.venv/bin/python analysis/make_figure2_dt10.py                 # figures/figure2
.venv/bin/python analysis/make_figure4_supporting_dt10.py      # figures/figure4
.venv/bin/python analysis/verify_dt10_artifacts.py             # checks committed dt10 artifacts are consistent
```

Shared modules: `analysis/revision_utils.py` (paths, rating maps, metric
helpers), `analysis/text_baselines.py`, `analysis/history_supervised_baselines.py`,
`analysis/filter_duplicates.py`.

## Manuscript-era code

`analysis/legacy/` and `scripts/legacy/` hold the original submission's
pipelines (generic-LLM methods, hybrid RF+PP, earlier figures). They target
retired splits and are kept for provenance only; see their READMEs.
