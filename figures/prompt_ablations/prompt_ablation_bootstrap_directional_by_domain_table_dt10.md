# Prompt ablations: domain-specific directional performance

Directional ratings use the established three-bin definition: low (1–2), neutral (3), and high (4–5). Intervals use the same participant-clustered percentile bootstrap.

| Domain | Model | Configuration | Directional accuracy (95% CI) | Directional macro-F1 (95% CI) |
| --- | --- | --- | --- | --- |
| Content | GPT-4o-mini | History + ratings only | 0.619 [0.581, 0.660] | 0.316 [0.293, 0.339] |
| Content | GPT-4o-mini | History text only | 0.643 [0.603, 0.682] | 0.295 [0.275, 0.317] |
| Content | GPT-4o-mini | PP + history (no CBT/ACT) | 0.627 [0.590, 0.666] | 0.343 [0.322, 0.367] |
| Content | GPT-4o-mini | PP + history + CBT/ACT | 0.611 [0.573, 0.650] | 0.330 [0.309, 0.352] |
| Content | GPT-5 | History + ratings only | 0.654 [0.618, 0.689] | 0.501 [0.453, 0.547] |
| Content | GPT-5 | History text only | 0.631 [0.591, 0.673] | 0.299 [0.277, 0.322] |
| Content | GPT-5 | PP + history (no CBT/ACT) | 0.648 [0.613, 0.684] | 0.486 [0.435, 0.533] |
| Content | GPT-5 | PP + history + CBT/ACT | 0.669 [0.632, 0.708] | 0.486 [0.435, 0.536] |
| Content | Gemini-2.5-Pro | History + ratings only | 0.648 [0.611, 0.684] | 0.477 [0.430, 0.522] |
| Content | Gemini-2.5-Pro | History text only | 0.618 [0.579, 0.654] | 0.318 [0.293, 0.343] |
| Content | Gemini-2.5-Pro | PP + history (no CBT/ACT) | 0.679 [0.642, 0.716] | 0.462 [0.414, 0.512] |
| Content | Gemini-2.5-Pro | PP + history + CBT/ACT | 0.685 [0.646, 0.722] | 0.462 [0.410, 0.513] |
| Content | Grok-4.3 | History + ratings only | 0.693 [0.654, 0.729] | 0.526 [0.467, 0.579] |
| Content | Grok-4.3 | History text only | 0.616 [0.576, 0.654] | 0.311 [0.286, 0.339] |
| Content | Grok-4.3 | PP + history (no CBT/ACT) | 0.719 [0.682, 0.754] | 0.524 [0.469, 0.573] |
| Content | Grok-4.3 | PP + history + CBT/ACT | 0.709 [0.671, 0.745] | 0.525 [0.472, 0.574] |
| Coping | GPT-4o-mini | History + ratings only | 0.386 [0.354, 0.420] | 0.299 [0.272, 0.327] |
| Coping | GPT-4o-mini | History text only | 0.305 [0.276, 0.336] | 0.252 [0.226, 0.279] |
| Coping | GPT-4o-mini | PP + history (no CBT/ACT) | 0.391 [0.361, 0.423] | 0.330 [0.299, 0.363] |
| Coping | GPT-4o-mini | PP + history + CBT/ACT | 0.354 [0.324, 0.387] | 0.331 [0.298, 0.364] |
| Coping | GPT-5 | History + ratings only | 0.573 [0.536, 0.611] | 0.494 [0.458, 0.531] |
| Coping | GPT-5 | History text only | 0.404 [0.371, 0.440] | 0.312 [0.284, 0.342] |
| Coping | GPT-5 | PP + history (no CBT/ACT) | 0.565 [0.526, 0.605] | 0.476 [0.436, 0.514] |
| Coping | GPT-5 | PP + history + CBT/ACT | 0.622 [0.585, 0.661] | 0.533 [0.494, 0.571] |
| Coping | Gemini-2.5-Pro | History + ratings only | 0.594 [0.556, 0.632] | 0.485 [0.444, 0.526] |
| Coping | Gemini-2.5-Pro | History text only | 0.468 [0.434, 0.503] | 0.339 [0.309, 0.371] |
| Coping | Gemini-2.5-Pro | PP + history (no CBT/ACT) | 0.631 [0.596, 0.667] | 0.499 [0.460, 0.539] |
| Coping | Gemini-2.5-Pro | PP + history + CBT/ACT | 0.629 [0.591, 0.667] | 0.485 [0.440, 0.526] |
| Coping | Grok-4.3 | History + ratings only | 0.641 [0.603, 0.680] | 0.551 [0.509, 0.592] |
| Coping | Grok-4.3 | History text only | 0.391 [0.359, 0.423] | 0.315 [0.287, 0.344] |
| Coping | Grok-4.3 | PP + history (no CBT/ACT) | 0.640 [0.600, 0.680] | 0.536 [0.493, 0.577] |
| Coping | Grok-4.3 | PP + history + CBT/ACT | 0.645 [0.606, 0.684] | 0.537 [0.492, 0.576] |
| Quitting | GPT-4o-mini | History + ratings only | 0.450 [0.417, 0.486] | 0.362 [0.329, 0.395] |
| Quitting | GPT-4o-mini | History text only | 0.341 [0.311, 0.372] | 0.302 [0.272, 0.332] |
| Quitting | GPT-4o-mini | PP + history (no CBT/ACT) | 0.487 [0.450, 0.524] | 0.415 [0.375, 0.452] |
| Quitting | GPT-4o-mini | PP + history + CBT/ACT | 0.462 [0.424, 0.498] | 0.418 [0.377, 0.456] |
| Quitting | GPT-5 | History + ratings only | 0.619 [0.582, 0.656] | 0.542 [0.503, 0.581] |
| Quitting | GPT-5 | History text only | 0.314 [0.285, 0.346] | 0.289 [0.260, 0.320] |
| Quitting | GPT-5 | PP + history (no CBT/ACT) | 0.630 [0.593, 0.667] | 0.548 [0.509, 0.587] |
| Quitting | GPT-5 | PP + history + CBT/ACT | 0.664 [0.627, 0.702] | 0.578 [0.539, 0.616] |
| Quitting | Gemini-2.5-Pro | History + ratings only | 0.650 [0.615, 0.686] | 0.541 [0.503, 0.579] |
| Quitting | Gemini-2.5-Pro | History text only | 0.504 [0.464, 0.543] | 0.318 [0.288, 0.350] |
| Quitting | Gemini-2.5-Pro | PP + history (no CBT/ACT) | 0.654 [0.616, 0.691] | 0.513 [0.472, 0.555] |
| Quitting | Gemini-2.5-Pro | PP + history + CBT/ACT | 0.665 [0.629, 0.701] | 0.529 [0.487, 0.572] |
| Quitting | Grok-4.3 | History + ratings only | 0.682 [0.644, 0.719] | 0.595 [0.554, 0.633] |
| Quitting | Grok-4.3 | History text only | 0.391 [0.359, 0.423] | 0.322 [0.292, 0.351] |
| Quitting | Grok-4.3 | PP + history (no CBT/ACT) | 0.660 [0.622, 0.699] | 0.552 [0.509, 0.592] |
| Quitting | Grok-4.3 | PP + history + CBT/ACT | 0.674 [0.636, 0.713] | 0.568 [0.525, 0.609] |
