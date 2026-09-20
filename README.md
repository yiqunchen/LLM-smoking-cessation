# LLM-Based Smoking Cessation Message Evaluation

Can a large language model predict how an individual smoker will rate a
cessation message? We compare personalized LLM prompting (the model sees a
participant's characteristics and their ratings of past messages) against
generic prompting and supervised baselines (logistic regression / random
forest on participant features, rating history, and message embeddings).

Outcomes: three 5-point ratings per message — **content** quality, helpfulness
for **coping** with cravings, helpfulness for **quitting**. Metrics: accuracy,
macro-F1, quadratic weighted kappa (QWK), directional (low/neutral/high)
accuracy and macro-F1, with participant-clustered bootstrap CIs.

Read `AGENTS.md` before changing anything.

## Layout

```
analysis/                 Python, one script per step (see --help)
scripts/                  smoke_test_apis.sh · run_prompt_ablations.sh · build_prompt_ablation_package.sh
data/splits/canonical/    dt10 splits (k = 1/3/5/7 history messages per participant)
data/raw/                 restricted participant data (gitignored)
results/prompt_ablations/ predictions per model and prompt condition
results/manuscript/       dt10 personalized-prompt runs used by the main figures
figures/                  figure2 · figure4 · progress_summary · prompt_ablations (+ source CSVs)
reports/                  generated Word report
docs/                     DATA_POLICY.md · RESULTS_DT10.md · FEATURE_BLOCKS.md · prompt_templates.md
```

## Data

`dt10`: each participant's ten rated messages are split within-participant
into `k` history messages and the rest held out. `dt10_k7` (7 history, 898
held-out ratings, 301 participants) is the primary evaluation; `k1/k3/k5` are
for sensitivity. Definition and headline numbers: `docs/RESULTS_DT10.md`.

## Setup

```bash
uv sync                        # Python 3.12 -> .venv
export OPENROUTER_API_KEY=...  # every model runs through OpenRouter
```

## 1. Main dt10 analyses and sensitivity

```bash
.venv/bin/python analysis/build_dt10_splits.py             # data/splits/canonical from data/raw
.venv/bin/python analysis/run_dt10_grok.py --help          # personalized-prompt runs at k = 1/3/5/7
.venv/bin/python analysis/lc_dt10_rf.py                    # supervised baselines across k
.venv/bin/python analysis/build_progress_summary.py        # figures/progress_summary
.venv/bin/python analysis/make_figure2_dt10.py             # figures/figure2
.venv/bin/python analysis/make_figure4_supporting_dt10.py  # figures/figure4
.venv/bin/python analysis/verify_dt10_artifacts.py         # consistency check of committed outputs
```

Baselines and shared code: `history_supervised_baselines.py`,
`text_baselines.py`, `revision_utils.py` (paths, rating maps, metrics).

## 2. Prompt ablations

Which parts of the personalized prompt carry the signal? Four conditions,
five models (GPT-4o-mini, GPT-5, DeepSeek-R1, Grok-4.3, Gemini-2.5-Pro), all
898 `dt10_k7` rows:

| Condition | Metadata | History texts | History ratings | CBT/ACT labels |
|---|:-:|:-:|:-:|:-:|
| `pp_cbtact` (full PP) | x | x | x | x |
| `full_pp_no_cbtact` | x | x | x | |
| `history_ratings_only` | | x | x | |
| `history_text_only` | | x | | |

```bash
bash scripts/smoke_test_apis.sh                 # one request per model, writes nothing
bash scripts/run_prompt_ablations.sh            # launch/resume all runs (tmux, checkpointed)
bash scripts/build_prompt_ablation_package.sh   # verify -> bootstrap -> figures -> Word report
```

Outputs: `results/prompt_ablations/<model>/`, `figures/prompt_ablations/`,
`reports/prompt_ablation_results.docx`.
