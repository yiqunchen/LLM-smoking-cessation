# Figure 1-4 In-Place Verification

Date: 2026-05-27

Checked source Word draft: `/Users/yiqun/Downloads/Draft_final (1).docx`

## Bottom Line

The updated figure files exist in the repo, but they are **not all inserted in
the current Word draft**. The Word draft still embeds old/compressed images for
several figure slots.

## Figure Status

| Figure | Updated repo file(s) found | In `Draft_final (1).docx`? | Action needed |
|---|---|---|---|
| Figure 1 | `figures/llm-message-paper-figure1.svg`, `.pdf`, `.png` | No. The docx still has the older Figure 1 with “Personalization Essential,” “Hybrid Best Performance,” and “Only 3-5 ratings needed.” | Replace Word Figure 1 with the repo Figure 1 file. |
| Figure 2 | `figures/figure2/figure2_main_top3_vertical_300dpi.png` plus `figures/figure2/figure2_appendix_other3_vertical_300dpi.png`, or the individual `figures/figure2/bars_all_methods_*` panels | No. The docx embeds old Figure 2 panels titled “All Models x All Methods,” without the cleaned 70/30 title and corrected `Demographics + History + Message Embedding` supervised comparator. | Replace Figure 2 with the vertical main Figure 2 PNG/PDF and move the vertical supplement panel to the appendix/supplement. |
| Figure 3 | `figures/figure3_score_distributions_content.png`, `figures/figure3_score_distributions_coping.png`, `figures/figure3_score_distributions_quitting.png` | No. The docx embeds older grouped histogram panels that do not match the current repo Figure 3 files. | Replace Figure 3 panels with the current repo Figure 3 files, then use the revised caption. |
| Figure 4 | `figures/llm_selection_quality.png`, `figures/top_k_agreement_line.png`; supporting strict benchmark: `revision/figures/message_selection_gain.png` | Partly. The docx contains the original top-K LLM selection figure, but it is an embedded/downsampled old copy and does not include the supporting supervised/LLM/hybrid benchmark. | Keep the original Figure 4 slot if desired, but update the caption and add/cite the supporting benchmark `revision/figures/message_selection_gain.png`. |

## Downloads Search

I searched `/Users/yiqun/Downloads` for recent files with names matching
Figure 1 / Fig 1 / LLM message / smoking / cessation / PME. I did **not** find a
newer manually created LLM-smoking Figure 1 outside the repo. The most relevant
Figure 1 files found are already in:

- `figures/llm-message-paper-figure1.svg`
- `figures/llm-message-paper-figure1.pdf`
- `figures/llm-message-paper-figure1.png`

## Word Draft Embedded Media Mapping

The current Word draft embeds:

- `word/media/image1.png`: old Figure 1
- `word/media/image2.png` through `image6.png`: old Figure 2 panels
- `word/media/image7.png` through `image9.png`: old Figure 3 panels
- `word/media/image10.png`: original top-K Figure 4 panel
- `word/media/image11.png` through `image14.png`: Supplementary Figure 5 panels
