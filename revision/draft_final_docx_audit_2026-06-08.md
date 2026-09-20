# Draft Final Docx Audit

Date: 2026-06-08

Audited file: `/Users/yiqun/Downloads/Draft_final (1).docx`

The newer Downloads manuscript is **not revision-complete**. It still contains
old digital-twin framing, stale superiority claims, and does not include the
strict RF / LLM-PP / RF-anchor / LLM-anchor benchmark narrative.

## File Version

- `Draft_final (1).docx` is newer than `Draft_final.docx`:
  `2026-06-01 22:08` vs `2026-04-29 01:02`.
- The audit below uses `Draft_final (1).docx`.

## High-Level Status

- **Title:** Not updated. Current title remains
  `Personalized Prediction of Perceived Message Effectiveness Using Machine Learning and Large Language Models`.
  Replace with `Response-History-Informed Smoking-Cessation Message Evaluation`.
- **Abstract:** Not updated. It still claims LLM-based digital twins outperform
  supervised baselines.
- **PP terminology:** Incomplete. Reader-facing manuscript prose still has
  33 `digital twin` / `Digital Twin` hits outside references.
- **Strict benchmark:** Missing. The manuscript has no `RF-anchor`, no `QWK`,
  and no strict `0.474 / 0.462 / 0.473 / 0.461` accuracy summary.
- **Old/original split wording:** No hits for `old split`, `original split`,
  or `original-split`.
- **Figure captions:** Not updated. Captions still describe digital-twin
  models and old top-K LLM-only superiority.
- **Appendix A2/A3:** Not updated. It still presents an LLM-only hierarchy and
  reliable digital-twin onboarding language.

## Paragraphs Requiring Revision

### Title and Abstract

- **P1:** Title is too broad and not the requested short version.
- **P20-P24:** Replace the full abstract. Current P22 says:
  `LLM-based digital twins outperformed zero-/few-shot ... and supervised baselines`.
- **P25:** Keywords need response/rating history and hybrid models; also fix
  missing space after `LLMs);`.

### Introduction

- **P32:** Old digital-twin motivation paragraph. Replace with the narrower
  persona-conditioned/history-augmented framing.
- **P33:** Contributions still describe seven prompt strategies as the main
  contribution. Replace with shared-row benchmark contributions centered on
  response history, RF, LLM-PP, and hybrids.

### Methods

- **P40-P42:** Model overview still frames methods as traditional baselines,
  zero-/few-shot LLMs, and digital-twin approaches. Replace with supervised RF,
  LLM-PP, RF-anchor, and LLM-anchor as the primary comparison; generic LLMs
  should be contextual.
- **P43-P44:** Supervised section still says models are based on patient
  characteristics only and describes RF/LR as weak baselines. Replace with
  response-history-informed RF as the primary supervised benchmark.
- **P52-P53:** Section title and text still say `LLM-based digital twins`.
  Replace with `Persona-conditioned LLM and hybrid anchor methods`.
- **P55-P57:** Metrics do not include QWK or within-participant Spearman as
  revised diagnostics.
- **P58-P59:** Top-K section still describes Digital Twin prompting and LLM-only
  selection. Replace with supervised RF, LLM-PP, RF-anchor, and LLM-anchor
  message-selection gain over random.

### Results

- **P62-P65:** Results still claim personalized LLM/digital-twin superiority
  and old supervised-baseline ranges. Replace with strict shared-row
  RF/LLM-PP/RF-anchor/LLM-anchor results.
- **P66-P67:** Score-distribution section still treats wider digital-twin
  predictions as evidence of personalization. Reframe as class-imbalance and
  calibration context.
- **P68-P69:** Top-K results remain LLM-only. Add the supporting four-method
  message-selection benchmark and fixed feature block.

### Discussion and Conclusion

- **P71-P76:** Discussion still says personalized digital twin models
  substantially outperformed all approaches, achieved directional accuracies
  0.66-0.75, and wider distributions support individual heterogeneity. Replace
  with response history as strongest observed signal, RF/RF-anchor aggregate
  strength, and LLM-PP complementary role.
- **P78:** Conclusion still says digital twin models demonstrated highest
  accuracy. Replace with calibrated conclusion: supervised RF strongest
  aggregate benchmark, RF-anchor strongest ordinal agreement, LLM methods
  complementary for message selection.

### Captions and Appendix

- **P134-P137:** Figure captions still use digital-twin and LLM-only selection
  wording. Replace Figures 1-4 captions with the revision text.
- **P163-P170:** Appendix A1.6 still says `LLM-based Digital Twins`, basic
  profile, and enhanced profile. Rename to PP/persona-conditioned and
  Hybrid RF+PP while preserving function names where needed.
- **P171-P176:** Appendix A2 still claims a clear Digital Twin hierarchy.
  Replace with uncertainty for LLM-only prompt-family analyses and state that
  main claims use strict shared-row RF/LLM-PP/RF-anchor/LLM-anchor comparisons.
- **P420-P423:** Appendix A3 still says digital-twin configurations and
  reliable personalized digital-twin predictions. Replace with history-length
  sensitivity and caution that this is not evidence of reliable individual
  simulation.

## Acceptable Remaining Digital-Twin Text

Digital-twin wording in the reference list is acceptable because those are
cited paper titles or source names. The problem is reader-facing claims,
methods, captions, and appendix prose.

## Recommended Next Step

Create a revised copy of `/Users/yiqun/Downloads/Draft_final (1).docx` rather
than editing the original in place. The target copy should be named:

`/Users/yiqun/Downloads/Draft_final_PP_revised.docx`

