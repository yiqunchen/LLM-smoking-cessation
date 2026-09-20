# Prompt ablations: domain-specific clustered-bootstrap confidence intervals

Participant-clustered percentile bootstrap (2,000 replicates by default); each model/configuration uses the shared canonical dt10-k7 test set (898 messages from 301 participants). Metrics are reported separately for Content, Coping, and Quitting; no cross-domain mean is calculated.

| Domain | Model | Configuration | Accuracy (95% CI) | Macro-F1 (95% CI) | QWK (95% CI) |
| --- | --- | --- | --- | --- | --- |
| Content | GPT-4o-mini | History + ratings only | 0.341 [0.308, 0.375] | 0.162 [0.144, 0.180] | 0.085 [0.027, 0.143] |
| Content | GPT-4o-mini | History text only | 0.326 [0.292, 0.362] | 0.120 [0.106, 0.134] | 0.007 [-0.029, 0.044] |
| Content | GPT-4o-mini | PP + history (no CBT/ACT) | 0.344 [0.310, 0.380] | 0.168 [0.150, 0.188] | 0.125 [0.075, 0.176] |
| Content | GPT-4o-mini | PP + history + CBT/ACT | 0.343 [0.308, 0.378] | 0.170 [0.152, 0.189] | 0.139 [0.084, 0.192] |
| Content | GPT-5 | History + ratings only | 0.447 [0.412, 0.482] | 0.347 [0.300, 0.396] | 0.372 [0.290, 0.444] |
| Content | GPT-5 | History text only | 0.332 [0.301, 0.363] | 0.182 [0.164, 0.200] | 0.006 [-0.044, 0.059] |
| Content | GPT-5 | PP + history (no CBT/ACT) | 0.438 [0.399, 0.475] | 0.324 [0.284, 0.363] | 0.355 [0.277, 0.429] |
| Content | GPT-5 | PP + history + CBT/ACT | 0.457 [0.419, 0.495] | 0.335 [0.290, 0.380] | 0.388 [0.310, 0.459] |
| Content | Gemini-2.5-Pro | History + ratings only | 0.450 [0.412, 0.488] | 0.353 [0.305, 0.399] | 0.330 [0.251, 0.408] |
| Content | Gemini-2.5-Pro | History text only | 0.321 [0.288, 0.354] | 0.182 [0.160, 0.204] | 0.030 [-0.031, 0.091] |
| Content | Gemini-2.5-Pro | PP + history (no CBT/ACT) | 0.457 [0.419, 0.494] | 0.322 [0.282, 0.361] | 0.337 [0.261, 0.413] |
| Content | Gemini-2.5-Pro | PP + history + CBT/ACT | 0.441 [0.402, 0.479] | 0.310 [0.266, 0.355] | 0.338 [0.265, 0.409] |
| Content | Grok-4.3 | History + ratings only | 0.479 [0.439, 0.517] | 0.342 [0.306, 0.378] | 0.401 [0.317, 0.475] |
| Content | Grok-4.3 | History text only | 0.326 [0.294, 0.360] | 0.159 [0.139, 0.181] | 0.022 [-0.027, 0.070] |
| Content | Grok-4.3 | PP + history (no CBT/ACT) | 0.490 [0.452, 0.528] | 0.362 [0.316, 0.410] | 0.424 [0.344, 0.491] |
| Content | Grok-4.3 | PP + history + CBT/ACT | 0.492 [0.453, 0.532] | 0.372 [0.325, 0.420] | 0.432 [0.352, 0.502] |
| Coping | GPT-4o-mini | History + ratings only | 0.255 [0.225, 0.286] | 0.154 [0.135, 0.174] | 0.051 [0.003, 0.097] |
| Coping | GPT-4o-mini | History text only | 0.219 [0.192, 0.248] | 0.131 [0.114, 0.149] | 0.031 [-0.006, 0.069] |
| Coping | GPT-4o-mini | PP + history (no CBT/ACT) | 0.262 [0.232, 0.291] | 0.161 [0.143, 0.180] | 0.141 [0.096, 0.187] |
| Coping | GPT-4o-mini | PP + history + CBT/ACT | 0.261 [0.230, 0.292] | 0.186 [0.163, 0.212] | 0.164 [0.118, 0.209] |
| Coping | GPT-5 | History + ratings only | 0.373 [0.339, 0.408] | 0.351 [0.312, 0.388] | 0.465 [0.396, 0.531] |
| Coping | GPT-5 | History text only | 0.244 [0.214, 0.273] | 0.144 [0.127, 0.161] | -0.007 [-0.058, 0.046] |
| Coping | GPT-5 | PP + history (no CBT/ACT) | 0.360 [0.324, 0.399] | 0.333 [0.290, 0.376] | 0.434 [0.360, 0.503] |
| Coping | GPT-5 | PP + history + CBT/ACT | 0.398 [0.359, 0.436] | 0.359 [0.316, 0.401] | 0.489 [0.423, 0.552] |
| Coping | Gemini-2.5-Pro | History + ratings only | 0.382 [0.347, 0.419] | 0.345 [0.305, 0.381] | 0.409 [0.329, 0.483] |
| Coping | Gemini-2.5-Pro | History text only | 0.262 [0.231, 0.296] | 0.152 [0.134, 0.172] | -0.007 [-0.066, 0.054] |
| Coping | Gemini-2.5-Pro | PP + history (no CBT/ACT) | 0.415 [0.379, 0.453] | 0.371 [0.328, 0.412] | 0.421 [0.345, 0.495] |
| Coping | Gemini-2.5-Pro | PP + history + CBT/ACT | 0.404 [0.369, 0.440] | 0.358 [0.315, 0.398] | 0.429 [0.352, 0.502] |
| Coping | Grok-4.3 | History + ratings only | 0.408 [0.371, 0.445] | 0.375 [0.332, 0.415] | 0.484 [0.411, 0.552] |
| Coping | Grok-4.3 | History text only | 0.232 [0.203, 0.261] | 0.145 [0.128, 0.164] | -0.016 [-0.070, 0.041] |
| Coping | Grok-4.3 | PP + history (no CBT/ACT) | 0.412 [0.375, 0.450] | 0.377 [0.331, 0.420] | 0.506 [0.434, 0.571] |
| Coping | Grok-4.3 | PP + history + CBT/ACT | 0.429 [0.390, 0.467] | 0.386 [0.342, 0.428] | 0.510 [0.437, 0.579] |
| Quitting | GPT-4o-mini | History + ratings only | 0.275 [0.244, 0.306] | 0.178 [0.156, 0.200] | 0.124 [0.071, 0.173] |
| Quitting | GPT-4o-mini | History text only | 0.225 [0.198, 0.254] | 0.148 [0.129, 0.168] | 0.051 [0.003, 0.100] |
| Quitting | GPT-4o-mini | PP + history (no CBT/ACT) | 0.297 [0.267, 0.329] | 0.202 [0.177, 0.224] | 0.235 [0.174, 0.291] |
| Quitting | GPT-4o-mini | PP + history + CBT/ACT | 0.294 [0.261, 0.327] | 0.224 [0.195, 0.252] | 0.248 [0.188, 0.305] |
| Quitting | GPT-5 | History + ratings only | 0.385 [0.350, 0.422] | 0.368 [0.327, 0.406] | 0.483 [0.417, 0.544] |
| Quitting | GPT-5 | History text only | 0.219 [0.192, 0.249] | 0.158 [0.137, 0.180] | 0.009 [-0.034, 0.058] |
| Quitting | GPT-5 | PP + history (no CBT/ACT) | 0.410 [0.374, 0.449] | 0.387 [0.343, 0.427] | 0.491 [0.425, 0.554] |
| Quitting | GPT-5 | PP + history + CBT/ACT | 0.427 [0.393, 0.464] | 0.403 [0.361, 0.443] | 0.534 [0.472, 0.591] |
| Quitting | Gemini-2.5-Pro | History + ratings only | 0.432 [0.396, 0.470] | 0.390 [0.350, 0.429] | 0.479 [0.409, 0.540] |
| Quitting | Gemini-2.5-Pro | History text only | 0.243 [0.211, 0.275] | 0.127 [0.109, 0.145] | 0.021 [-0.034, 0.079] |
| Quitting | Gemini-2.5-Pro | PP + history (no CBT/ACT) | 0.415 [0.381, 0.454] | 0.367 [0.326, 0.408] | 0.456 [0.384, 0.526] |
| Quitting | Gemini-2.5-Pro | PP + history + CBT/ACT | 0.435 [0.398, 0.474] | 0.389 [0.346, 0.431] | 0.479 [0.408, 0.547] |
| Quitting | Grok-4.3 | History + ratings only | 0.450 [0.412, 0.487] | 0.426 [0.381, 0.468] | 0.532 [0.463, 0.597] |
| Quitting | Grok-4.3 | History text only | 0.225 [0.196, 0.255] | 0.145 [0.126, 0.164] | 0.017 [-0.035, 0.067] |
| Quitting | Grok-4.3 | PP + history (no CBT/ACT) | 0.432 [0.396, 0.469] | 0.401 [0.355, 0.442] | 0.538 [0.471, 0.602] |
| Quitting | Grok-4.3 | PP + history + CBT/ACT | 0.448 [0.412, 0.486] | 0.410 [0.369, 0.450] | 0.548 [0.480, 0.610] |
