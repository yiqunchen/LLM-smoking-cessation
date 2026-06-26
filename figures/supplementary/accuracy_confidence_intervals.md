# Accuracy Confidence Intervals

Individual-level bootstrap accuracy confidence intervals with `2000` resamples.

All rows use the cleaned canonical PP 70/30 source; known train/test duplicate items are removed from every method.

| Model | Method | Domain | N | Split | Accuracy | Accuracy_CI_Lower | Accuracy_CI_Upper | Accuracy_SE | Bootstrap_N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-4o-mini | Zero-shot (all) | Content | 306 | digital_twin_7030_cleaned | 0.386 | 0.333 | 0.438 | 0.028 | 2000 |
| GPT-5 | Zero-shot (all) | Content | 306 | digital_twin_7030_cleaned | 0.402 | 0.350 | 0.458 | 0.028 | 2000 |
| DeepSeek-R1 | Zero-shot (all) | Content | 306 | digital_twin_7030_cleaned | 0.301 | 0.248 | 0.350 | 0.026 | 2000 |
| Grok-4-Fast | Zero-shot (all) | Content | 306 | digital_twin_7030_cleaned | 0.399 | 0.346 | 0.454 | 0.028 | 2000 |
| Gemini-2.5-Pro | Zero-shot (all) | Content | 306 | digital_twin_7030_cleaned | 0.392 | 0.337 | 0.445 | 0.028 | 2000 |
| GPT-4o-mini | Few-shot (all) | Content | 306 | digital_twin_7030_cleaned | 0.369 | 0.314 | 0.425 | 0.028 | 2000 |
| GPT-5 | Few-shot (all) | Content | 306 | digital_twin_7030_cleaned | 0.402 | 0.346 | 0.454 | 0.028 | 2000 |
| DeepSeek-R1 | Few-shot (all) | Content | 306 | digital_twin_7030_cleaned | 0.314 | 0.261 | 0.363 | 0.027 | 2000 |
| Grok-4-Fast | Few-shot (all) | Content | 306 | digital_twin_7030_cleaned | 0.399 | 0.346 | 0.451 | 0.027 | 2000 |
| Gemini-2.5-Pro | Few-shot (all) | Content | 306 | digital_twin_7030_cleaned | 0.366 | 0.314 | 0.422 | 0.028 | 2000 |
| GPT-4o-mini | PP | Content | 306 | digital_twin_7030_cleaned | 0.418 | 0.366 | 0.474 | 0.027 | 2000 |
| GPT-5 | PP | Content | 306 | digital_twin_7030_cleaned | 0.458 | 0.402 | 0.510 | 0.028 | 2000 |
| DeepSeek-R1 | PP | Content | 306 | digital_twin_7030_cleaned | 0.435 | 0.382 | 0.490 | 0.029 | 2000 |
| Grok-4-Fast | PP | Content | 306 | digital_twin_7030_cleaned | 0.487 | 0.428 | 0.546 | 0.029 | 2000 |
| Gemini-2.5-Pro | PP | Content | 306 | digital_twin_7030_cleaned | 0.428 | 0.373 | 0.484 | 0.029 | 2000 |
| GPT-4o-mini | Hybrid RF+PP | Content | 306 | digital_twin_7030_cleaned | 0.428 | 0.373 | 0.484 | 0.028 | 2000 |
| GPT-5 | Hybrid RF+PP | Content | 306 | digital_twin_7030_cleaned | 0.474 | 0.418 | 0.529 | 0.028 | 2000 |
| DeepSeek-R1 | Hybrid RF+PP | Content | 302 | digital_twin_7030_cleaned | 0.334 | 0.281 | 0.387 | 0.027 | 2000 |
| Grok-4-Fast | Hybrid RF+PP | Content | 306 | digital_twin_7030_cleaned | 0.474 | 0.418 | 0.529 | 0.029 | 2000 |
| Gemini-2.5-Pro | Hybrid RF+PP | Content | 306 | digital_twin_7030_cleaned | 0.418 | 0.366 | 0.474 | 0.028 | 2000 |
| GPT-4o-mini | Zero-shot (all) | Coping | 307 | digital_twin_7030_cleaned | 0.254 | 0.208 | 0.306 | 0.025 | 2000 |
| GPT-5 | Zero-shot (all) | Coping | 307 | digital_twin_7030_cleaned | 0.241 | 0.195 | 0.290 | 0.024 | 2000 |
| DeepSeek-R1 | Zero-shot (all) | Coping | 307 | digital_twin_7030_cleaned | 0.205 | 0.163 | 0.251 | 0.023 | 2000 |
| Grok-4-Fast | Zero-shot (all) | Coping | 307 | digital_twin_7030_cleaned | 0.313 | 0.261 | 0.365 | 0.027 | 2000 |
| Gemini-2.5-Pro | Zero-shot (all) | Coping | 307 | digital_twin_7030_cleaned | 0.290 | 0.241 | 0.342 | 0.026 | 2000 |
| GPT-4o-mini | Few-shot (all) | Coping | 307 | digital_twin_7030_cleaned | 0.264 | 0.215 | 0.313 | 0.025 | 2000 |
| GPT-5 | Few-shot (all) | Coping | 307 | digital_twin_7030_cleaned | 0.241 | 0.195 | 0.290 | 0.024 | 2000 |
| DeepSeek-R1 | Few-shot (all) | Coping | 307 | digital_twin_7030_cleaned | 0.218 | 0.173 | 0.267 | 0.024 | 2000 |
| Grok-4-Fast | Few-shot (all) | Coping | 307 | digital_twin_7030_cleaned | 0.322 | 0.274 | 0.371 | 0.026 | 2000 |
| Gemini-2.5-Pro | Few-shot (all) | Coping | 307 | digital_twin_7030_cleaned | 0.241 | 0.195 | 0.287 | 0.024 | 2000 |
| GPT-4o-mini | PP | Coping | 307 | digital_twin_7030_cleaned | 0.319 | 0.267 | 0.371 | 0.027 | 2000 |
| GPT-5 | PP | Coping | 307 | digital_twin_7030_cleaned | 0.371 | 0.319 | 0.427 | 0.027 | 2000 |
| DeepSeek-R1 | PP | Coping | 307 | digital_twin_7030_cleaned | 0.388 | 0.332 | 0.440 | 0.027 | 2000 |
| Grok-4-Fast | PP | Coping | 307 | digital_twin_7030_cleaned | 0.446 | 0.388 | 0.502 | 0.029 | 2000 |
| Gemini-2.5-Pro | PP | Coping | 307 | digital_twin_7030_cleaned | 0.433 | 0.375 | 0.489 | 0.029 | 2000 |
| GPT-4o-mini | Hybrid RF+PP | Coping | 307 | digital_twin_7030_cleaned | 0.352 | 0.296 | 0.407 | 0.028 | 2000 |
| GPT-5 | Hybrid RF+PP | Coping | 307 | digital_twin_7030_cleaned | 0.427 | 0.371 | 0.482 | 0.028 | 2000 |
| DeepSeek-R1 | Hybrid RF+PP | Coping | 303 | digital_twin_7030_cleaned | 0.376 | 0.320 | 0.429 | 0.028 | 2000 |
| Grok-4-Fast | Hybrid RF+PP | Coping | 307 | digital_twin_7030_cleaned | 0.420 | 0.365 | 0.476 | 0.029 | 2000 |
| Gemini-2.5-Pro | Hybrid RF+PP | Coping | 307 | digital_twin_7030_cleaned | 0.404 | 0.352 | 0.459 | 0.028 | 2000 |
| GPT-4o-mini | Zero-shot (all) | Quitting | 307 | digital_twin_7030_cleaned | 0.309 | 0.261 | 0.362 | 0.027 | 2000 |
| GPT-5 | Zero-shot (all) | Quitting | 307 | digital_twin_7030_cleaned | 0.143 | 0.104 | 0.182 | 0.020 | 2000 |
| DeepSeek-R1 | Zero-shot (all) | Quitting | 307 | digital_twin_7030_cleaned | 0.137 | 0.101 | 0.176 | 0.020 | 2000 |
| Grok-4-Fast | Zero-shot (all) | Quitting | 307 | digital_twin_7030_cleaned | 0.283 | 0.235 | 0.336 | 0.026 | 2000 |
| Gemini-2.5-Pro | Zero-shot (all) | Quitting | 307 | digital_twin_7030_cleaned | 0.300 | 0.251 | 0.352 | 0.026 | 2000 |
| GPT-4o-mini | Few-shot (all) | Quitting | 307 | digital_twin_7030_cleaned | 0.274 | 0.225 | 0.322 | 0.025 | 2000 |
| GPT-5 | Few-shot (all) | Quitting | 307 | digital_twin_7030_cleaned | 0.212 | 0.169 | 0.254 | 0.023 | 2000 |
| DeepSeek-R1 | Few-shot (all) | Quitting | 307 | digital_twin_7030_cleaned | 0.287 | 0.238 | 0.339 | 0.026 | 2000 |
| Grok-4-Fast | Few-shot (all) | Quitting | 307 | digital_twin_7030_cleaned | 0.355 | 0.303 | 0.414 | 0.028 | 2000 |
| Gemini-2.5-Pro | Few-shot (all) | Quitting | 307 | digital_twin_7030_cleaned | 0.309 | 0.261 | 0.362 | 0.026 | 2000 |
| GPT-4o-mini | PP | Quitting | 307 | digital_twin_7030_cleaned | 0.326 | 0.277 | 0.381 | 0.027 | 2000 |
| GPT-5 | PP | Quitting | 307 | digital_twin_7030_cleaned | 0.407 | 0.352 | 0.463 | 0.028 | 2000 |
| DeepSeek-R1 | PP | Quitting | 307 | digital_twin_7030_cleaned | 0.401 | 0.345 | 0.453 | 0.027 | 2000 |
| Grok-4-Fast | PP | Quitting | 307 | digital_twin_7030_cleaned | 0.440 | 0.384 | 0.495 | 0.029 | 2000 |
| Gemini-2.5-Pro | PP | Quitting | 307 | digital_twin_7030_cleaned | 0.450 | 0.394 | 0.505 | 0.028 | 2000 |
| GPT-4o-mini | Hybrid RF+PP | Quitting | 307 | digital_twin_7030_cleaned | 0.388 | 0.336 | 0.443 | 0.028 | 2000 |
| GPT-5 | Hybrid RF+PP | Quitting | 307 | digital_twin_7030_cleaned | 0.446 | 0.394 | 0.505 | 0.028 | 2000 |
| DeepSeek-R1 | Hybrid RF+PP | Quitting | 303 | digital_twin_7030_cleaned | 0.413 | 0.360 | 0.469 | 0.028 | 2000 |
| Grok-4-Fast | Hybrid RF+PP | Quitting | 307 | digital_twin_7030_cleaned | 0.492 | 0.440 | 0.547 | 0.028 | 2000 |
| Gemini-2.5-Pro | Hybrid RF+PP | Quitting | 307 | digital_twin_7030_cleaned | 0.436 | 0.384 | 0.489 | 0.028 | 2000 |

Artifacts:

- [accuracy_confidence_intervals.csv](accuracy_confidence_intervals.csv)
- [accuracy_confidence_intervals.png](accuracy_confidence_intervals.png)
- [accuracy_confidence_intervals.pdf](accuracy_confidence_intervals.pdf)
