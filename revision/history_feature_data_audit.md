# History Feature Data Audit

Date checked: 2026-06-01

## 2026-06-01 Correction: `Demographics` Was Previously Too Broad

Earlier Figure 2/strict-RF outputs used a source feature set named
`Demographics`, but that code path actually called the broader
`extract_features()` metadata extractor. That broader metadata block included
age, gender, race/ethnicity, Hispanic/Latino status, sexual orientation,
education, household income, smoking behavior, quit intention/motivation,
social support, household/friend smoking context, and psychosocial items.

The corrected Figure 2 pipeline now uses true demographics only:

- age
- gender
- race/ethnicity
- Hispanic/Latino status
- sexual orientation
- education
- household income

Important split caveat checked on 2026-06-08: the corrected `Demographics`
feature set does **not** include explicit history or embedding columns
(`history=False`, `embedding=False` in
`analysis-script/history_supervised_baselines.py`). However, the canonical PP
70/30 split is within-participant: all 301 cleaned test participants also appear
in the training rows. Demographic patterns can therefore act as participant
proxies. In the cleaned test set, 307/307 rows have a demographics pattern seen
in training, and 228/307 rows have a pattern that maps only to the same
participant in training. This is not direct history-feature leakage, but it is
not independent-participant generalization.

The Demographics RF line is also stronger on the rows where the demographic
pattern behaves like a participant proxy:

| Domain | Demographic pattern group | N | Accuracy | Macro-F1 | QWK |
|---|---|---:|---:|---:|---:|
| Content | unique participant-proxy pattern | 227 | 0.493 | 0.332 | 0.352 |
| Content | shared demographic pattern | 79 | 0.456 | 0.291 | 0.314 |
| Coping | unique participant-proxy pattern | 228 | 0.469 | 0.395 | 0.371 |
| Coping | shared demographic pattern | 79 | 0.430 | 0.347 | 0.282 |
| Quitting | unique participant-proxy pattern | 228 | 0.496 | 0.411 | 0.484 |
| Quitting | shared demographic pattern | 79 | 0.392 | 0.269 | 0.386 |

Interpretation: Demographics RF should be described as a same-participant
within-split supervised reference. It is not using explicit history features,
but part of its strength is consistent with demographic vectors serving as
participant identifiers in the PP split.

The corrected Figure 2 supervised baseline families are now only:

- `Demographics`
- `Demographics + History + Message Embedding`

For the corrected Figure 2 source table, true demographics-only RF is lower than
the earlier mislabeled broad-metadata row:

| Model | Feature set | Content acc. | Coping acc. | Quitting acc. | Content QWK | Coping QWK | Quitting QWK |
|---|---|---:|---:|---:|---:|---:|---:|
| RF | Demographics | 0.484 | 0.459 | 0.469 | 0.348 | 0.357 | 0.466 |
| LR | Demographics | 0.422 | 0.355 | 0.368 | 0.232 | 0.108 | 0.145 |
| RF | Demographics + History + Message Embedding | 0.363 | 0.355 | 0.349 | 0.042 | 0.065 | 0.073 |
| LR | Demographics + History + Message Embedding | 0.484 | 0.433 | 0.502 | 0.325 | 0.333 | 0.480 |

For corrected strict RF at `k_train = 7`, the strongest RF history result is
demographics plus coarse history, not the high-dimensional embedding block:

| Feature set | Mean accuracy | Mean macro-F1 | Mean QWK | Mean directional macro-F1 |
|---|---:|---:|---:|---:|
| Demographics | 0.474 | 0.403 | 0.527 | 0.547 |
| Demo+History | 0.446 | 0.368 | 0.549 | 0.549 |
| Demographics + History + Message Embedding | 0.364 | 0.254 | 0.212 | 0.399 |

Use this wording:

> Historical participant ratings improved ordinal personalization in the
> corrected supervised RF benchmark, while the full demographics + history +
> message-embedding feature block was more useful for LR than RF.

The older notes below are retained only as provenance for the earlier audit;
use the corrected 2026-06-01 values above for the revision.

## Current Guardrail

Do not use the older broad-metadata values in manuscript text or captions. Use
the corrected 2026-06-01 rows above for Figure 2 and for the history discussion.
