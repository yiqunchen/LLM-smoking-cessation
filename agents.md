# Figure And Revision Preferences

- Follow the repo's existing publication palette unless there is a strong reason not to.
- Reuse these semantic colors consistently:
  `GPT-4o-mini #0173B2`, `GPT-5 #DE8F05`, `DeepSeek-R1 #029E73`, `Grok-4-Fast #CC78BC`, `Gemini-2.5-Pro #CA9161`, `Logistic Regression #E02020`, `Random Forest #7F7F7F`.
- Keep figure and axes backgrounds white.
- Use bold axis labels, titles, and usually bold tick labels for publication or revision figures.
- Prefer neutral gray reference and grid styling similar to `../distributional-ppi`: `#4D4D4D`, dashed reference lines, light grid alpha.
- If a figure introduces non-model categories, draw from the same restrained, colorblind-safe family and avoid ad hoc bright colors when a repo color can be reused.
- When style is ambiguous, inspect:
  `analysis-script/create_publication_figures.py`
  `analysis-script/create_comprehensive_figures.py`
  `../distributional-ppi/src/distributional_ppi/plotting.py`
- Never fabricate preview rows or placeholder revision results.
- Do not generate or ship mock revision data. If a result is missing, say it is missing and give the rerun reference instead of fabricating preview values.
