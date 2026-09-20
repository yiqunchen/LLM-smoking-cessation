# Paired differences for all condition pairs

Each row contrasts two prompt conditions on the same participant-bootstrap replicates. Verdict: 'comparison better' if the 95% CI lies above zero, 'reference better' if below zero, otherwise 'comparable'. No multiplicity correction is applied.

| Model | Domain | Metric | Reference | Comparison | Comparison - reference (95% CI) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-4o-mini | Content | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.001 [-0.018, 0.020] | comparable |
| GPT-4o-mini | Content | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.002 [-0.017, 0.014] | comparable |
| GPT-4o-mini | Content | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.015 [-0.059, 0.032] | comparable |
| GPT-4o-mini | Content | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.016 [-0.003, 0.036] | comparable |
| GPT-4o-mini | Content | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.013 [-0.005, 0.033] | comparable |
| GPT-4o-mini | Coping | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.001 [-0.026, 0.027] | comparable |
| GPT-4o-mini | Coping | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.025 [-0.050, -0.001] | reference better |
| GPT-4o-mini | Coping | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.023 [-0.060, 0.016] | comparable |
| GPT-4o-mini | Coping | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.037 [0.004, 0.068] | comparison better |
| GPT-4o-mini | Coping | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.001 [-0.031, 0.029] | comparable |
| GPT-4o-mini | Quitting | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.003 [-0.022, 0.031] | comparable |
| GPT-4o-mini | Quitting | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.022 [-0.047, 0.004] | comparable |
| GPT-4o-mini | Quitting | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.013 [-0.055, 0.030] | comparable |
| GPT-4o-mini | Quitting | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.024 [-0.006, 0.058] | comparable |
| GPT-4o-mini | Quitting | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.003 [-0.037, 0.032] | comparable |
| GPT-4o-mini | Content | accuracy | PP + history + CBT/ACT | History + ratings only | -0.002 [-0.023, 0.018] | comparable |
| GPT-4o-mini | Content | macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.008 [-0.025, 0.008] | comparable |
| GPT-4o-mini | Content | qwk | PP + history + CBT/ACT | History + ratings only | -0.054 [-0.109, 0.001] | comparable |
| GPT-4o-mini | Content | directional_accuracy | PP + history + CBT/ACT | History + ratings only | 0.008 [-0.014, 0.029] | comparable |
| GPT-4o-mini | Content | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.014 [-0.035, 0.006] | comparable |
| GPT-4o-mini | Coping | accuracy | PP + history + CBT/ACT | History + ratings only | -0.006 [-0.039, 0.028] | comparable |
| GPT-4o-mini | Coping | macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.032 [-0.059, -0.006] | reference better |
| GPT-4o-mini | Coping | qwk | PP + history + CBT/ACT | History + ratings only | -0.113 [-0.159, -0.066] | reference better |
| GPT-4o-mini | Coping | directional_accuracy | PP + history + CBT/ACT | History + ratings only | 0.032 [-0.008, 0.073] | comparable |
| GPT-4o-mini | Coping | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.033 [-0.070, 0.004] | comparable |
| GPT-4o-mini | Quitting | accuracy | PP + history + CBT/ACT | History + ratings only | -0.019 [-0.050, 0.014] | comparable |
| GPT-4o-mini | Quitting | macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.046 [-0.076, -0.015] | reference better |
| GPT-4o-mini | Quitting | qwk | PP + history + CBT/ACT | History + ratings only | -0.124 [-0.176, -0.075] | reference better |
| GPT-4o-mini | Quitting | directional_accuracy | PP + history + CBT/ACT | History + ratings only | -0.012 [-0.049, 0.027] | comparable |
| GPT-4o-mini | Quitting | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.056 [-0.091, -0.018] | reference better |
| GPT-4o-mini | Content | accuracy | PP + history + CBT/ACT | History text only | -0.017 [-0.038, 0.006] | comparable |
| GPT-4o-mini | Content | macro_f1 | PP + history + CBT/ACT | History text only | -0.051 [-0.069, -0.032] | reference better |
| GPT-4o-mini | Content | qwk | PP + history + CBT/ACT | History text only | -0.132 [-0.184, -0.079] | reference better |
| GPT-4o-mini | Content | directional_accuracy | PP + history + CBT/ACT | History text only | 0.031 [0.009, 0.053] | comparison better |
| GPT-4o-mini | Content | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.035 [-0.057, -0.012] | reference better |
| GPT-4o-mini | Coping | accuracy | PP + history + CBT/ACT | History text only | -0.041 [-0.071, -0.012] | reference better |
| GPT-4o-mini | Coping | macro_f1 | PP + history + CBT/ACT | History text only | -0.056 [-0.084, -0.029] | reference better |
| GPT-4o-mini | Coping | qwk | PP + history + CBT/ACT | History text only | -0.133 [-0.182, -0.083] | reference better |
| GPT-4o-mini | Coping | directional_accuracy | PP + history + CBT/ACT | History text only | -0.049 [-0.086, -0.012] | reference better |
| GPT-4o-mini | Coping | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.080 [-0.118, -0.041] | reference better |
| GPT-4o-mini | Quitting | accuracy | PP + history + CBT/ACT | History text only | -0.069 [-0.106, -0.032] | reference better |
| GPT-4o-mini | Quitting | macro_f1 | PP + history + CBT/ACT | History text only | -0.076 [-0.109, -0.043] | reference better |
| GPT-4o-mini | Quitting | qwk | PP + history + CBT/ACT | History text only | -0.198 [-0.263, -0.130] | reference better |
| GPT-4o-mini | Quitting | directional_accuracy | PP + history + CBT/ACT | History text only | -0.121 [-0.166, -0.076] | reference better |
| GPT-4o-mini | Quitting | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.115 [-0.162, -0.065] | reference better |
| GPT-4o-mini | Content | accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.003 [-0.023, 0.017] | comparable |
| GPT-4o-mini | Content | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.007 [-0.024, 0.010] | comparable |
| GPT-4o-mini | Content | qwk | PP + history (no CBT/ACT) | History + ratings only | -0.040 [-0.085, 0.003] | comparable |
| GPT-4o-mini | Content | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.008 [-0.027, 0.010] | comparable |
| GPT-4o-mini | Content | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.027 [-0.048, -0.008] | reference better |
| GPT-4o-mini | Coping | accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.007 [-0.028, 0.013] | comparable |
| GPT-4o-mini | Coping | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.007 [-0.019, 0.005] | comparable |
| GPT-4o-mini | Coping | qwk | PP + history (no CBT/ACT) | History + ratings only | -0.090 [-0.126, -0.055] | reference better |
| GPT-4o-mini | Coping | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.004 [-0.030, 0.022] | comparable |
| GPT-4o-mini | Coping | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.031 [-0.057, -0.006] | reference better |
| GPT-4o-mini | Quitting | accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.022 [-0.043, 0.000] | comparable |
| GPT-4o-mini | Quitting | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.023 [-0.042, -0.006] | reference better |
| GPT-4o-mini | Quitting | qwk | PP + history (no CBT/ACT) | History + ratings only | -0.111 [-0.149, -0.071] | reference better |
| GPT-4o-mini | Quitting | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.037 [-0.062, -0.011] | reference better |
| GPT-4o-mini | Quitting | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.053 [-0.081, -0.025] | reference better |
| GPT-4o-mini | Content | accuracy | PP + history (no CBT/ACT) | History text only | -0.018 [-0.039, 0.002] | comparable |
| GPT-4o-mini | Content | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.049 [-0.066, -0.032] | reference better |
| GPT-4o-mini | Content | qwk | PP + history (no CBT/ACT) | History text only | -0.118 [-0.162, -0.074] | reference better |
| GPT-4o-mini | Content | directional_accuracy | PP + history (no CBT/ACT) | History text only | 0.016 [-0.006, 0.036] | comparable |
| GPT-4o-mini | Content | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.048 [-0.070, -0.026] | reference better |
| GPT-4o-mini | Coping | accuracy | PP + history (no CBT/ACT) | History text only | -0.042 [-0.067, -0.018] | reference better |
| GPT-4o-mini | Coping | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.031 [-0.050, -0.012] | reference better |
| GPT-4o-mini | Coping | qwk | PP + history (no CBT/ACT) | History text only | -0.109 [-0.156, -0.062] | reference better |
| GPT-4o-mini | Coping | directional_accuracy | PP + history (no CBT/ACT) | History text only | -0.086 [-0.117, -0.054] | reference better |
| GPT-4o-mini | Coping | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.078 [-0.112, -0.045] | reference better |
| GPT-4o-mini | Quitting | accuracy | PP + history (no CBT/ACT) | History text only | -0.072 [-0.103, -0.042] | reference better |
| GPT-4o-mini | Quitting | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.054 [-0.079, -0.028] | reference better |
| GPT-4o-mini | Quitting | qwk | PP + history (no CBT/ACT) | History text only | -0.184 [-0.242, -0.125] | reference better |
| GPT-4o-mini | Quitting | directional_accuracy | PP + history (no CBT/ACT) | History text only | -0.146 [-0.185, -0.106] | reference better |
| GPT-4o-mini | Quitting | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.113 [-0.155, -0.068] | reference better |
| GPT-4o-mini | Content | accuracy | History + ratings only | History text only | -0.014 [-0.035, 0.006] | comparable |
| GPT-4o-mini | Content | macro_f1 | History + ratings only | History text only | -0.042 [-0.059, -0.026] | reference better |
| GPT-4o-mini | Content | qwk | History + ratings only | History text only | -0.078 [-0.125, -0.034] | reference better |
| GPT-4o-mini | Content | directional_accuracy | History + ratings only | History text only | 0.023 [0.006, 0.042] | comparison better |
| GPT-4o-mini | Content | directional_macro_f1 | History + ratings only | History text only | -0.021 [-0.039, 0.001] | comparable |
| GPT-4o-mini | Coping | accuracy | History + ratings only | History text only | -0.036 [-0.060, -0.011] | reference better |
| GPT-4o-mini | Coping | macro_f1 | History + ratings only | History text only | -0.024 [-0.041, -0.004] | reference better |
| GPT-4o-mini | Coping | qwk | History + ratings only | History text only | -0.020 [-0.057, 0.019] | comparable |
| GPT-4o-mini | Coping | directional_accuracy | History + ratings only | History text only | -0.081 [-0.111, -0.051] | reference better |
| GPT-4o-mini | Coping | directional_macro_f1 | History + ratings only | History text only | -0.047 [-0.072, -0.021] | reference better |
| GPT-4o-mini | Quitting | accuracy | History + ratings only | History text only | -0.050 [-0.078, -0.021] | reference better |
| GPT-4o-mini | Quitting | macro_f1 | History + ratings only | History text only | -0.031 [-0.051, -0.008] | reference better |
| GPT-4o-mini | Quitting | qwk | History + ratings only | History text only | -0.073 [-0.118, -0.027] | reference better |
| GPT-4o-mini | Quitting | directional_accuracy | History + ratings only | History text only | -0.109 [-0.145, -0.073] | reference better |
| GPT-4o-mini | Quitting | directional_macro_f1 | History + ratings only | History text only | -0.060 [-0.094, -0.023] | reference better |
| GPT-5 | Content | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.019 [-0.046, 0.008] | comparable |
| GPT-5 | Content | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.011 [-0.033, 0.013] | comparable |
| GPT-5 | Content | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.032 [-0.070, 0.003] | comparable |
| GPT-5 | Content | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.021 [-0.044, 0.001] | comparable |
| GPT-5 | Content | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.000 [-0.026, 0.027] | comparable |
| GPT-5 | Coping | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.038 [-0.062, -0.013] | reference better |
| GPT-5 | Coping | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.026 [-0.053, 0.002] | comparable |
| GPT-5 | Coping | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.055 [-0.090, -0.022] | reference better |
| GPT-5 | Coping | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.058 [-0.082, -0.034] | reference better |
| GPT-5 | Coping | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.057 [-0.083, -0.031] | reference better |
| GPT-5 | Quitting | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.017 [-0.046, 0.010] | comparable |
| GPT-5 | Quitting | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.017 [-0.045, 0.011] | comparable |
| GPT-5 | Quitting | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.044 [-0.077, -0.009] | reference better |
| GPT-5 | Quitting | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.033 [-0.059, -0.010] | reference better |
| GPT-5 | Quitting | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.030 [-0.059, -0.002] | reference better |
| GPT-5 | Content | accuracy | PP + history + CBT/ACT | History + ratings only | -0.010 [-0.037, 0.017] | comparable |
| GPT-5 | Content | macro_f1 | PP + history + CBT/ACT | History + ratings only | 0.012 [-0.022, 0.053] | comparable |
| GPT-5 | Content | qwk | PP + history + CBT/ACT | History + ratings only | -0.016 [-0.056, 0.024] | comparable |
| GPT-5 | Content | directional_accuracy | PP + history + CBT/ACT | History + ratings only | -0.016 [-0.038, 0.006] | comparable |
| GPT-5 | Content | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | 0.014 [-0.019, 0.053] | comparable |
| GPT-5 | Coping | accuracy | PP + history + CBT/ACT | History + ratings only | -0.024 [-0.052, 0.001] | comparable |
| GPT-5 | Coping | macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.008 [-0.039, 0.022] | comparable |
| GPT-5 | Coping | qwk | PP + history + CBT/ACT | History + ratings only | -0.024 [-0.059, 0.011] | comparable |
| GPT-5 | Coping | directional_accuracy | PP + history + CBT/ACT | History + ratings only | -0.049 [-0.075, -0.026] | reference better |
| GPT-5 | Coping | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.039 [-0.067, -0.011] | reference better |
| GPT-5 | Quitting | accuracy | PP + history + CBT/ACT | History + ratings only | -0.041 [-0.066, -0.017] | reference better |
| GPT-5 | Quitting | macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.035 [-0.062, -0.006] | reference better |
| GPT-5 | Quitting | qwk | PP + history + CBT/ACT | History + ratings only | -0.051 [-0.082, -0.019] | reference better |
| GPT-5 | Quitting | directional_accuracy | PP + history + CBT/ACT | History + ratings only | -0.045 [-0.069, -0.020] | reference better |
| GPT-5 | Quitting | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.036 [-0.065, -0.008] | reference better |
| GPT-5 | Content | accuracy | PP + history + CBT/ACT | History text only | -0.125 [-0.165, -0.085] | reference better |
| GPT-5 | Content | macro_f1 | PP + history + CBT/ACT | History text only | -0.153 [-0.203, -0.108] | reference better |
| GPT-5 | Content | qwk | PP + history + CBT/ACT | History text only | -0.382 [-0.457, -0.302] | reference better |
| GPT-5 | Content | directional_accuracy | PP + history + CBT/ACT | History text only | -0.038 [-0.074, -0.003] | reference better |
| GPT-5 | Content | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.187 [-0.239, -0.132] | reference better |
| GPT-5 | Coping | accuracy | PP + history + CBT/ACT | History text only | -0.154 [-0.198, -0.108] | reference better |
| GPT-5 | Coping | macro_f1 | PP + history + CBT/ACT | History text only | -0.215 [-0.258, -0.170] | reference better |
| GPT-5 | Coping | qwk | PP + history + CBT/ACT | History text only | -0.496 [-0.570, -0.416] | reference better |
| GPT-5 | Coping | directional_accuracy | PP + history + CBT/ACT | History text only | -0.218 [-0.262, -0.176] | reference better |
| GPT-5 | Coping | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.221 [-0.268, -0.173] | reference better |
| GPT-5 | Quitting | accuracy | PP + history + CBT/ACT | History text only | -0.207 [-0.255, -0.159] | reference better |
| GPT-5 | Quitting | macro_f1 | PP + history + CBT/ACT | History text only | -0.246 [-0.292, -0.198] | reference better |
| GPT-5 | Quitting | qwk | PP + history + CBT/ACT | History text only | -0.525 [-0.593, -0.451] | reference better |
| GPT-5 | Quitting | directional_accuracy | PP + history + CBT/ACT | History text only | -0.350 [-0.399, -0.299] | reference better |
| GPT-5 | Quitting | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.289 [-0.337, -0.235] | reference better |
| GPT-5 | Content | accuracy | PP + history (no CBT/ACT) | History + ratings only | 0.009 [-0.015, 0.032] | comparable |
| GPT-5 | Content | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.023 [-0.007, 0.056] | comparable |
| GPT-5 | Content | qwk | PP + history (no CBT/ACT) | History + ratings only | 0.016 [-0.020, 0.054] | comparable |
| GPT-5 | Content | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | 0.006 [-0.018, 0.027] | comparable |
| GPT-5 | Content | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.015 [-0.019, 0.052] | comparable |
| GPT-5 | Coping | accuracy | PP + history (no CBT/ACT) | History + ratings only | 0.013 [-0.010, 0.037] | comparable |
| GPT-5 | Coping | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.018 [-0.010, 0.045] | comparable |
| GPT-5 | Coping | qwk | PP + history (no CBT/ACT) | History + ratings only | 0.031 [0.000, 0.063] | comparison better |
| GPT-5 | Coping | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | 0.009 [-0.013, 0.030] | comparable |
| GPT-5 | Coping | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.018 [-0.007, 0.044] | comparable |
| GPT-5 | Quitting | accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.024 [-0.046, -0.002] | reference better |
| GPT-5 | Quitting | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.018 [-0.040, 0.005] | comparable |
| GPT-5 | Quitting | qwk | PP + history (no CBT/ACT) | History + ratings only | -0.007 [-0.036, 0.024] | comparable |
| GPT-5 | Quitting | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.011 [-0.033, 0.010] | comparable |
| GPT-5 | Quitting | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.006 [-0.031, 0.018] | comparable |
| GPT-5 | Content | accuracy | PP + history (no CBT/ACT) | History text only | -0.106 [-0.149, -0.065] | reference better |
| GPT-5 | Content | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.142 [-0.182, -0.100] | reference better |
| GPT-5 | Content | qwk | PP + history (no CBT/ACT) | History text only | -0.350 [-0.430, -0.264] | reference better |
| GPT-5 | Content | directional_accuracy | PP + history (no CBT/ACT) | History text only | -0.017 [-0.055, 0.018] | comparable |
| GPT-5 | Content | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.187 [-0.240, -0.132] | reference better |
| GPT-5 | Coping | accuracy | PP + history (no CBT/ACT) | History text only | -0.116 [-0.161, -0.072] | reference better |
| GPT-5 | Coping | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.189 [-0.233, -0.143] | reference better |
| GPT-5 | Coping | qwk | PP + history (no CBT/ACT) | History text only | -0.441 [-0.514, -0.362] | reference better |
| GPT-5 | Coping | directional_accuracy | PP + history (no CBT/ACT) | History text only | -0.160 [-0.203, -0.118] | reference better |
| GPT-5 | Coping | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.164 [-0.208, -0.119] | reference better |
| GPT-5 | Quitting | accuracy | PP + history (no CBT/ACT) | History text only | -0.190 [-0.237, -0.144] | reference better |
| GPT-5 | Quitting | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.229 [-0.275, -0.180] | reference better |
| GPT-5 | Quitting | qwk | PP + history (no CBT/ACT) | History text only | -0.481 [-0.554, -0.407] | reference better |
| GPT-5 | Quitting | directional_accuracy | PP + history (no CBT/ACT) | History text only | -0.316 [-0.362, -0.268] | reference better |
| GPT-5 | Quitting | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.259 [-0.305, -0.208] | reference better |
| GPT-5 | Content | accuracy | History + ratings only | History text only | -0.115 [-0.156, -0.076] | reference better |
| GPT-5 | Content | macro_f1 | History + ratings only | History text only | -0.165 [-0.216, -0.116] | reference better |
| GPT-5 | Content | qwk | History + ratings only | History text only | -0.366 [-0.443, -0.282] | reference better |
| GPT-5 | Content | directional_accuracy | History + ratings only | History text only | -0.022 [-0.058, 0.013] | comparable |
| GPT-5 | Content | directional_macro_f1 | History + ratings only | History text only | -0.202 [-0.252, -0.149] | reference better |
| GPT-5 | Coping | accuracy | History + ratings only | History text only | -0.129 [-0.171, -0.087] | reference better |
| GPT-5 | Coping | macro_f1 | History + ratings only | History text only | -0.207 [-0.246, -0.166] | reference better |
| GPT-5 | Coping | qwk | History + ratings only | History text only | -0.472 [-0.545, -0.390] | reference better |
| GPT-5 | Coping | directional_accuracy | History + ratings only | History text only | -0.169 [-0.210, -0.128] | reference better |
| GPT-5 | Coping | directional_macro_f1 | History + ratings only | History text only | -0.182 [-0.224, -0.139] | reference better |
| GPT-5 | Quitting | accuracy | History + ratings only | History text only | -0.166 [-0.211, -0.118] | reference better |
| GPT-5 | Quitting | macro_f1 | History + ratings only | History text only | -0.211 [-0.255, -0.163] | reference better |
| GPT-5 | Quitting | qwk | History + ratings only | History text only | -0.474 [-0.542, -0.397] | reference better |
| GPT-5 | Quitting | directional_accuracy | History + ratings only | History text only | -0.305 [-0.352, -0.255] | reference better |
| GPT-5 | Quitting | directional_macro_f1 | History + ratings only | History text only | -0.253 [-0.300, -0.202] | reference better |
| Grok-4.3 | Content | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.002 [-0.025, 0.020] | comparable |
| Grok-4.3 | Content | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.010 [-0.043, 0.017] | comparable |
| Grok-4.3 | Content | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.008 [-0.039, 0.021] | comparable |
| Grok-4.3 | Content | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.010 [-0.008, 0.028] | comparable |
| Grok-4.3 | Content | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.001 [-0.024, 0.021] | comparable |
| Grok-4.3 | Coping | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.017 [-0.045, 0.011] | comparable |
| Grok-4.3 | Coping | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.010 [-0.039, 0.017] | comparable |
| Grok-4.3 | Coping | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.004 [-0.039, 0.029] | comparable |
| Grok-4.3 | Coping | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.004 [-0.028, 0.019] | comparable |
| Grok-4.3 | Coping | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.001 [-0.030, 0.029] | comparable |
| Grok-4.3 | Quitting | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.016 [-0.040, 0.008] | comparable |
| Grok-4.3 | Quitting | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.010 [-0.038, 0.016] | comparable |
| Grok-4.3 | Quitting | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.010 [-0.045, 0.025] | comparable |
| Grok-4.3 | Quitting | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.013 [-0.036, 0.009] | comparable |
| Grok-4.3 | Quitting | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.017 [-0.046, 0.011] | comparable |
| Grok-4.3 | Content | accuracy | PP + history + CBT/ACT | History + ratings only | -0.013 [-0.038, 0.012] | comparable |
| Grok-4.3 | Content | macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.030 [-0.070, 0.008] | comparable |
| Grok-4.3 | Content | qwk | PP + history + CBT/ACT | History + ratings only | -0.031 [-0.065, 0.002] | comparable |
| Grok-4.3 | Content | directional_accuracy | PP + history + CBT/ACT | History + ratings only | -0.017 [-0.039, 0.004] | comparable |
| Grok-4.3 | Content | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | 0.002 [-0.031, 0.033] | comparable |
| Grok-4.3 | Coping | accuracy | PP + history + CBT/ACT | History + ratings only | -0.021 [-0.049, 0.007] | comparable |
| Grok-4.3 | Coping | macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.011 [-0.037, 0.014] | comparable |
| Grok-4.3 | Coping | qwk | PP + history + CBT/ACT | History + ratings only | -0.026 [-0.062, 0.011] | comparable |
| Grok-4.3 | Coping | directional_accuracy | PP + history + CBT/ACT | History + ratings only | -0.003 [-0.028, 0.022] | comparable |
| Grok-4.3 | Coping | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | 0.015 [-0.018, 0.047] | comparable |
| Grok-4.3 | Quitting | accuracy | PP + history + CBT/ACT | History + ratings only | 0.002 [-0.021, 0.025] | comparable |
| Grok-4.3 | Quitting | macro_f1 | PP + history + CBT/ACT | History + ratings only | 0.016 [-0.011, 0.041] | comparable |
| Grok-4.3 | Quitting | qwk | PP + history + CBT/ACT | History + ratings only | -0.016 [-0.048, 0.017] | comparable |
| Grok-4.3 | Quitting | directional_accuracy | PP + history + CBT/ACT | History + ratings only | 0.008 [-0.013, 0.028] | comparable |
| Grok-4.3 | Quitting | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | 0.026 [-0.002, 0.053] | comparable |
| Grok-4.3 | Content | accuracy | PP + history + CBT/ACT | History text only | -0.166 [-0.213, -0.119] | reference better |
| Grok-4.3 | Content | macro_f1 | PP + history + CBT/ACT | History text only | -0.213 [-0.265, -0.162] | reference better |
| Grok-4.3 | Content | qwk | PP + history + CBT/ACT | History text only | -0.410 [-0.497, -0.319] | reference better |
| Grok-4.3 | Content | directional_accuracy | PP + history + CBT/ACT | History text only | -0.094 [-0.129, -0.056] | reference better |
| Grok-4.3 | Content | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.214 [-0.269, -0.154] | reference better |
| Grok-4.3 | Coping | accuracy | PP + history + CBT/ACT | History text only | -0.197 [-0.245, -0.153] | reference better |
| Grok-4.3 | Coping | macro_f1 | PP + history + CBT/ACT | History text only | -0.241 [-0.285, -0.193] | reference better |
| Grok-4.3 | Coping | qwk | PP + history + CBT/ACT | History text only | -0.526 [-0.608, -0.438] | reference better |
| Grok-4.3 | Coping | directional_accuracy | PP + history + CBT/ACT | History text only | -0.254 [-0.299, -0.209] | reference better |
| Grok-4.3 | Coping | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.221 [-0.267, -0.172] | reference better |
| Grok-4.3 | Quitting | accuracy | PP + history + CBT/ACT | History text only | -0.223 [-0.270, -0.173] | reference better |
| Grok-4.3 | Quitting | macro_f1 | PP + history + CBT/ACT | History text only | -0.265 [-0.310, -0.219] | reference better |
| Grok-4.3 | Quitting | qwk | PP + history + CBT/ACT | History text only | -0.531 [-0.609, -0.451] | reference better |
| Grok-4.3 | Quitting | directional_accuracy | PP + history + CBT/ACT | History text only | -0.283 [-0.329, -0.237] | reference better |
| Grok-4.3 | Quitting | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.247 [-0.297, -0.195] | reference better |
| Grok-4.3 | Content | accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.011 [-0.036, 0.013] | comparable |
| Grok-4.3 | Content | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.020 [-0.051, 0.012] | comparable |
| Grok-4.3 | Content | qwk | PP + history (no CBT/ACT) | History + ratings only | -0.023 [-0.056, 0.009] | comparable |
| Grok-4.3 | Content | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.027 [-0.047, -0.007] | reference better |
| Grok-4.3 | Content | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.003 [-0.022, 0.029] | comparable |
| Grok-4.3 | Coping | accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.004 [-0.027, 0.018] | comparable |
| Grok-4.3 | Coping | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.001 [-0.026, 0.026] | comparable |
| Grok-4.3 | Coping | qwk | PP + history (no CBT/ACT) | History + ratings only | -0.022 [-0.058, 0.014] | comparable |
| Grok-4.3 | Coping | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | 0.001 [-0.021, 0.023] | comparable |
| Grok-4.3 | Coping | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.015 [-0.015, 0.045] | comparable |
| Grok-4.3 | Quitting | accuracy | PP + history (no CBT/ACT) | History + ratings only | 0.018 [-0.004, 0.039] | comparable |
| Grok-4.3 | Quitting | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.026 [0.001, 0.052] | comparison better |
| Grok-4.3 | Quitting | qwk | PP + history (no CBT/ACT) | History + ratings only | -0.006 [-0.040, 0.026] | comparable |
| Grok-4.3 | Quitting | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | 0.021 [0.000, 0.041] | comparable |
| Grok-4.3 | Quitting | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.043 [0.014, 0.070] | comparison better |
| Grok-4.3 | Content | accuracy | PP + history (no CBT/ACT) | History text only | -0.164 [-0.211, -0.117] | reference better |
| Grok-4.3 | Content | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.203 [-0.256, -0.151] | reference better |
| Grok-4.3 | Content | qwk | PP + history (no CBT/ACT) | History text only | -0.402 [-0.484, -0.314] | reference better |
| Grok-4.3 | Content | directional_accuracy | PP + history (no CBT/ACT) | History text only | -0.104 [-0.136, -0.070] | reference better |
| Grok-4.3 | Content | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.213 [-0.267, -0.152] | reference better |
| Grok-4.3 | Coping | accuracy | PP + history (no CBT/ACT) | History text only | -0.180 [-0.226, -0.136] | reference better |
| Grok-4.3 | Coping | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.231 [-0.276, -0.183] | reference better |
| Grok-4.3 | Coping | qwk | PP + history (no CBT/ACT) | History text only | -0.522 [-0.597, -0.442] | reference better |
| Grok-4.3 | Coping | directional_accuracy | PP + history (no CBT/ACT) | History text only | -0.249 [-0.292, -0.208] | reference better |
| Grok-4.3 | Coping | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.221 [-0.265, -0.175] | reference better |
| Grok-4.3 | Quitting | accuracy | PP + history (no CBT/ACT) | History text only | -0.207 [-0.254, -0.159] | reference better |
| Grok-4.3 | Quitting | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.256 [-0.303, -0.207] | reference better |
| Grok-4.3 | Quitting | qwk | PP + history (no CBT/ACT) | History text only | -0.521 [-0.594, -0.445] | reference better |
| Grok-4.3 | Quitting | directional_accuracy | PP + history (no CBT/ACT) | History text only | -0.269 [-0.315, -0.227] | reference better |
| Grok-4.3 | Quitting | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.230 [-0.281, -0.179] | reference better |
| Grok-4.3 | Content | accuracy | History + ratings only | History text only | -0.153 [-0.200, -0.105] | reference better |
| Grok-4.3 | Content | macro_f1 | History + ratings only | History text only | -0.183 [-0.223, -0.144] | reference better |
| Grok-4.3 | Content | qwk | History + ratings only | History text only | -0.379 [-0.465, -0.284] | reference better |
| Grok-4.3 | Content | directional_accuracy | History + ratings only | History text only | -0.077 [-0.114, -0.039] | reference better |
| Grok-4.3 | Content | directional_macro_f1 | History + ratings only | History text only | -0.215 [-0.272, -0.153] | reference better |
| Grok-4.3 | Coping | accuracy | History + ratings only | History text only | -0.176 [-0.222, -0.130] | reference better |
| Grok-4.3 | Coping | macro_f1 | History + ratings only | History text only | -0.230 [-0.274, -0.183] | reference better |
| Grok-4.3 | Coping | qwk | History + ratings only | History text only | -0.500 [-0.578, -0.416] | reference better |
| Grok-4.3 | Coping | directional_accuracy | History + ratings only | History text only | -0.251 [-0.294, -0.208] | reference better |
| Grok-4.3 | Coping | directional_macro_f1 | History + ratings only | History text only | -0.236 [-0.282, -0.188] | reference better |
| Grok-4.3 | Quitting | accuracy | History + ratings only | History text only | -0.225 [-0.272, -0.177] | reference better |
| Grok-4.3 | Quitting | macro_f1 | History + ratings only | History text only | -0.281 [-0.328, -0.233] | reference better |
| Grok-4.3 | Quitting | qwk | History + ratings only | History text only | -0.515 [-0.591, -0.436] | reference better |
| Grok-4.3 | Quitting | directional_accuracy | History + ratings only | History text only | -0.291 [-0.336, -0.247] | reference better |
| Grok-4.3 | Quitting | directional_macro_f1 | History + ratings only | History text only | -0.273 [-0.322, -0.224] | reference better |
| Gemini-2.5-Pro | Content | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.016 [-0.011, 0.044] | comparable |
| Gemini-2.5-Pro | Content | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.012 [-0.015, 0.040] | comparable |
| Gemini-2.5-Pro | Content | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.001 [-0.040, 0.043] | comparable |
| Gemini-2.5-Pro | Content | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.006 [-0.028, 0.018] | comparable |
| Gemini-2.5-Pro | Content | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.000 [-0.035, 0.039] | comparable |
| Gemini-2.5-Pro | Coping | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.011 [-0.015, 0.038] | comparable |
| Gemini-2.5-Pro | Coping | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.013 [-0.016, 0.045] | comparable |
| Gemini-2.5-Pro | Coping | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.008 [-0.049, 0.033] | comparable |
| Gemini-2.5-Pro | Coping | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.002 [-0.022, 0.027] | comparable |
| Gemini-2.5-Pro | Coping | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | 0.014 [-0.019, 0.051] | comparable |
| Gemini-2.5-Pro | Quitting | accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.020 [-0.045, 0.004] | comparable |
| Gemini-2.5-Pro | Quitting | macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.022 [-0.050, 0.007] | comparable |
| Gemini-2.5-Pro | Quitting | qwk | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.023 [-0.066, 0.018] | comparable |
| Gemini-2.5-Pro | Quitting | directional_accuracy | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.011 [-0.034, 0.011] | comparable |
| Gemini-2.5-Pro | Quitting | directional_macro_f1 | PP + history + CBT/ACT | PP + history (no CBT/ACT) | -0.017 [-0.050, 0.019] | comparable |
| Gemini-2.5-Pro | Content | accuracy | PP + history + CBT/ACT | History + ratings only | 0.009 [-0.019, 0.038] | comparable |
| Gemini-2.5-Pro | Content | macro_f1 | PP + history + CBT/ACT | History + ratings only | 0.042 [0.005, 0.082] | comparison better |
| Gemini-2.5-Pro | Content | qwk | PP + history + CBT/ACT | History + ratings only | -0.008 [-0.067, 0.050] | comparable |
| Gemini-2.5-Pro | Content | directional_accuracy | PP + history + CBT/ACT | History + ratings only | -0.037 [-0.062, -0.009] | reference better |
| Gemini-2.5-Pro | Content | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | 0.015 [-0.028, 0.060] | comparable |
| Gemini-2.5-Pro | Coping | accuracy | PP + history + CBT/ACT | History + ratings only | -0.022 [-0.051, 0.007] | comparable |
| Gemini-2.5-Pro | Coping | macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.013 [-0.044, 0.019] | comparable |
| Gemini-2.5-Pro | Coping | qwk | PP + history + CBT/ACT | History + ratings only | -0.020 [-0.069, 0.031] | comparable |
| Gemini-2.5-Pro | Coping | directional_accuracy | PP + history + CBT/ACT | History + ratings only | -0.036 [-0.065, -0.008] | reference better |
| Gemini-2.5-Pro | Coping | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | -0.000 [-0.039, 0.039] | comparable |
| Gemini-2.5-Pro | Quitting | accuracy | PP + history + CBT/ACT | History + ratings only | -0.003 [-0.031, 0.026] | comparable |
| Gemini-2.5-Pro | Quitting | macro_f1 | PP + history + CBT/ACT | History + ratings only | 0.000 [-0.031, 0.037] | comparable |
| Gemini-2.5-Pro | Quitting | qwk | PP + history + CBT/ACT | History + ratings only | -0.000 [-0.047, 0.046] | comparable |
| Gemini-2.5-Pro | Quitting | directional_accuracy | PP + history + CBT/ACT | History + ratings only | -0.014 [-0.040, 0.012] | comparable |
| Gemini-2.5-Pro | Quitting | directional_macro_f1 | PP + history + CBT/ACT | History + ratings only | 0.011 [-0.024, 0.048] | comparable |
| Gemini-2.5-Pro | Content | accuracy | PP + history + CBT/ACT | History text only | -0.120 [-0.169, -0.073] | reference better |
| Gemini-2.5-Pro | Content | macro_f1 | PP + history + CBT/ACT | History text only | -0.128 [-0.175, -0.082] | reference better |
| Gemini-2.5-Pro | Content | qwk | PP + history + CBT/ACT | History text only | -0.308 [-0.396, -0.217] | reference better |
| Gemini-2.5-Pro | Content | directional_accuracy | PP + history + CBT/ACT | History text only | -0.067 [-0.100, -0.033] | reference better |
| Gemini-2.5-Pro | Content | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.145 [-0.198, -0.090] | reference better |
| Gemini-2.5-Pro | Coping | accuracy | PP + history + CBT/ACT | History text only | -0.143 [-0.192, -0.095] | reference better |
| Gemini-2.5-Pro | Coping | macro_f1 | PP + history + CBT/ACT | History text only | -0.206 [-0.254, -0.156] | reference better |
| Gemini-2.5-Pro | Coping | qwk | PP + history + CBT/ACT | History text only | -0.436 [-0.521, -0.347] | reference better |
| Gemini-2.5-Pro | Coping | directional_accuracy | PP + history + CBT/ACT | History text only | -0.161 [-0.201, -0.121] | reference better |
| Gemini-2.5-Pro | Coping | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.146 [-0.201, -0.090] | reference better |
| Gemini-2.5-Pro | Quitting | accuracy | PP + history + CBT/ACT | History text only | -0.193 [-0.242, -0.142] | reference better |
| Gemini-2.5-Pro | Quitting | macro_f1 | PP + history + CBT/ACT | History text only | -0.263 [-0.307, -0.215] | reference better |
| Gemini-2.5-Pro | Quitting | qwk | PP + history + CBT/ACT | History text only | -0.458 [-0.543, -0.371] | reference better |
| Gemini-2.5-Pro | Quitting | directional_accuracy | PP + history + CBT/ACT | History text only | -0.160 [-0.199, -0.124] | reference better |
| Gemini-2.5-Pro | Quitting | directional_macro_f1 | PP + history + CBT/ACT | History text only | -0.211 [-0.263, -0.159] | reference better |
| Gemini-2.5-Pro | Content | accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.007 [-0.036, 0.025] | comparable |
| Gemini-2.5-Pro | Content | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.031 [-0.006, 0.069] | comparable |
| Gemini-2.5-Pro | Content | qwk | PP + history (no CBT/ACT) | History + ratings only | -0.007 [-0.064, 0.049] | comparable |
| Gemini-2.5-Pro | Content | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.031 [-0.058, -0.006] | reference better |
| Gemini-2.5-Pro | Content | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.015 [-0.023, 0.052] | comparable |
| Gemini-2.5-Pro | Coping | accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.033 [-0.063, -0.004] | reference better |
| Gemini-2.5-Pro | Coping | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.026 [-0.058, 0.004] | comparable |
| Gemini-2.5-Pro | Coping | qwk | PP + history (no CBT/ACT) | History + ratings only | -0.012 [-0.056, 0.033] | comparable |
| Gemini-2.5-Pro | Coping | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.038 [-0.065, -0.012] | reference better |
| Gemini-2.5-Pro | Coping | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | -0.015 [-0.049, 0.018] | comparable |
| Gemini-2.5-Pro | Quitting | accuracy | PP + history (no CBT/ACT) | History + ratings only | 0.017 [-0.009, 0.046] | comparable |
| Gemini-2.5-Pro | Quitting | macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.022 [-0.004, 0.051] | comparable |
| Gemini-2.5-Pro | Quitting | qwk | PP + history (no CBT/ACT) | History + ratings only | 0.023 [-0.020, 0.064] | comparable |
| Gemini-2.5-Pro | Quitting | directional_accuracy | PP + history (no CBT/ACT) | History + ratings only | -0.003 [-0.029, 0.022] | comparable |
| Gemini-2.5-Pro | Quitting | directional_macro_f1 | PP + history (no CBT/ACT) | History + ratings only | 0.028 [-0.011, 0.065] | comparable |
| Gemini-2.5-Pro | Content | accuracy | PP + history (no CBT/ACT) | History text only | -0.136 [-0.180, -0.092] | reference better |
| Gemini-2.5-Pro | Content | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.140 [-0.181, -0.098] | reference better |
| Gemini-2.5-Pro | Content | qwk | PP + history (no CBT/ACT) | History text only | -0.307 [-0.389, -0.223] | reference better |
| Gemini-2.5-Pro | Content | directional_accuracy | PP + history (no CBT/ACT) | History text only | -0.061 [-0.092, -0.030] | reference better |
| Gemini-2.5-Pro | Content | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.145 [-0.197, -0.091] | reference better |
| Gemini-2.5-Pro | Coping | accuracy | PP + history (no CBT/ACT) | History text only | -0.154 [-0.203, -0.104] | reference better |
| Gemini-2.5-Pro | Coping | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.219 [-0.266, -0.169] | reference better |
| Gemini-2.5-Pro | Coping | qwk | PP + history (no CBT/ACT) | History text only | -0.428 [-0.510, -0.341] | reference better |
| Gemini-2.5-Pro | Coping | directional_accuracy | PP + history (no CBT/ACT) | History text only | -0.164 [-0.204, -0.123] | reference better |
| Gemini-2.5-Pro | Coping | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.160 [-0.212, -0.106] | reference better |
| Gemini-2.5-Pro | Quitting | accuracy | PP + history (no CBT/ACT) | History text only | -0.173 [-0.220, -0.126] | reference better |
| Gemini-2.5-Pro | Quitting | macro_f1 | PP + history (no CBT/ACT) | History text only | -0.241 [-0.286, -0.195] | reference better |
| Gemini-2.5-Pro | Quitting | qwk | PP + history (no CBT/ACT) | History text only | -0.435 [-0.512, -0.354] | reference better |
| Gemini-2.5-Pro | Quitting | directional_accuracy | PP + history (no CBT/ACT) | History text only | -0.149 [-0.187, -0.114] | reference better |
| Gemini-2.5-Pro | Quitting | directional_macro_f1 | PP + history (no CBT/ACT) | History text only | -0.194 [-0.244, -0.144] | reference better |
| Gemini-2.5-Pro | Content | accuracy | History + ratings only | History text only | -0.129 [-0.177, -0.083] | reference better |
| Gemini-2.5-Pro | Content | macro_f1 | History + ratings only | History text only | -0.170 [-0.222, -0.120] | reference better |
| Gemini-2.5-Pro | Content | qwk | History + ratings only | History text only | -0.300 [-0.381, -0.212] | reference better |
| Gemini-2.5-Pro | Content | directional_accuracy | History + ratings only | History text only | -0.030 [-0.065, 0.003] | comparable |
| Gemini-2.5-Pro | Content | directional_macro_f1 | History + ratings only | History text only | -0.160 [-0.209, -0.110] | reference better |
| Gemini-2.5-Pro | Coping | accuracy | History + ratings only | History text only | -0.120 [-0.169, -0.072] | reference better |
| Gemini-2.5-Pro | Coping | macro_f1 | History + ratings only | History text only | -0.192 [-0.237, -0.148] | reference better |
| Gemini-2.5-Pro | Coping | qwk | History + ratings only | History text only | -0.416 [-0.494, -0.331] | reference better |
| Gemini-2.5-Pro | Coping | directional_accuracy | History + ratings only | History text only | -0.126 [-0.167, -0.086] | reference better |
| Gemini-2.5-Pro | Coping | directional_macro_f1 | History + ratings only | History text only | -0.145 [-0.195, -0.096] | reference better |
| Gemini-2.5-Pro | Quitting | accuracy | History + ratings only | History text only | -0.189 [-0.241, -0.141] | reference better |
| Gemini-2.5-Pro | Quitting | macro_f1 | History + ratings only | History text only | -0.263 [-0.307, -0.218] | reference better |
| Gemini-2.5-Pro | Quitting | qwk | History + ratings only | History text only | -0.458 [-0.532, -0.377] | reference better |
| Gemini-2.5-Pro | Quitting | directional_accuracy | History + ratings only | History text only | -0.146 [-0.186, -0.108] | reference better |
| Gemini-2.5-Pro | Quitting | directional_macro_f1 | History + ratings only | History text only | -0.223 [-0.267, -0.177] | reference better |
