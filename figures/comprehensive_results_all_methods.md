# Comprehensive Results: All Models × All Methods

| Model          | Model_ID                  | Method               | Category     | Domain   |   Accuracy |    Acc±1 |       Kappa |   Kendall_Tau |   Spearman_Rho |   N |
|:---------------|:--------------------------|:---------------------|:-------------|:---------|-----------:|---------:|------------:|--------------:|---------------:|----:|
| GPT-4o-mini    | gpt-4o-mini               | Zero-shot (all)      | Generic LLM  | Content  |   0.364964 | 0.916058 |  0.00396941 |   0.0275174   |     0          | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Zero-shot (all)      | Generic LLM  | Coping   |   0.284672 | 0.737226 |  0.0168424  |   0.0205264   |    -0.122436   | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Zero-shot (all)      | Generic LLM  | Quitting |   0.251825 | 0.762774 | -0.0214027  |   0.0322749   |    -0.106958   | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Zero-shot (select)   | Generic LLM  | Content  |   0.386861 | 0.883212 |  0.0481783  |   0.0721251   |     0.0979367  | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Zero-shot (select)   | Generic LLM  | Coping   |   0.284672 | 0.80292  | -0.00934088 |   0.0925332   |    -0.068315   | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Zero-shot (select)   | Generic LLM  | Quitting |   0.284672 | 0.740876 |  0.021696   |   0.0480426   |    -0.0731837  | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Few-shot (all)       | Generic LLM  | Content  |   0.350365 | 0.894161 |  0.00566769 |   0.0851307   |     0.0881436  | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Few-shot (all)       | Generic LLM  | Coping   |   0.255474 | 0.737226 | -0.00851617 |   0.0530038   |    -0.0212189  | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Few-shot (all)       | Generic LLM  | Quitting |   0.291971 | 0.729927 |  0.056899   |   0.104523    |    -0.00934706 | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Few-shot (select)    | Generic LLM  | Content  |   0.328467 | 0.879562 | -0.0113744  |   0.0667378   |     0.115012   | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Few-shot (select)    | Generic LLM  | Coping   |   0.30292  | 0.770073 |  0.0521435  |   0.140642    |     0.116001   | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Few-shot (select)    | Generic LLM  | Quitting |   0.328467 | 0.70073  |  0.0922905  |   0.130861    |    -0.0703788  | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Continuous (NL)      | Generic LLM  | Content  |   0.339416 | 0.835766 |  0.0244699  |   0.0616484   |    -0.0433547  | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Continuous (NL)      | Generic LLM  | Coping   |   0.291971 | 0.770073 |  0.0119703  |   0.111434    |     0.0362037  | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Continuous (NL)      | Generic LLM  | Quitting |   0.273723 | 0.744526 |  0.041318   |   0.0935182   |    -0.160451   | 274 |
| GPT-4o-mini    | gpt-4o-mini               | Digital Twin (70/30) | Digital Twin | Content  |   0.416149 | 0.909938 |  0.114259   |   0.292552    |    -1          | 322 |
| GPT-4o-mini    | gpt-4o-mini               | Digital Twin (70/30) | Digital Twin | Coping   |   0.328173 | 0.789474 |  0.108677   |   0.295419    |     0          | 323 |
| GPT-4o-mini    | gpt-4o-mini               | Digital Twin (70/30) | Digital Twin | Quitting |   0.325077 | 0.814241 |  0.0954473  |   0.295329    |    -1          | 323 |
| GPT-5          | gpt-5                     | Zero-shot (all)      | Generic LLM  | Content  |   0.375912 | 0.864964 |  0.0410757  |  -0.0206016   |     0.121132   | 274 |
| GPT-5          | gpt-5                     | Zero-shot (all)      | Generic LLM  | Coping   |   0.273723 | 0.726277 |  0.0176734  |   0.0137877   |    -0.123539   | 274 |
| GPT-5          | gpt-5                     | Zero-shot (all)      | Generic LLM  | Quitting |   0.19708  | 0.540146 |  0.0313665  |   0.0381038   |     0.0674388  | 274 |
| GPT-5          | gpt-5                     | Zero-shot (select)   | Generic LLM  | Content  |   0.390511 | 0.857664 |  0.0672293  |  -0.00524408  |    -0.0485531  | 274 |
| GPT-5          | gpt-5                     | Zero-shot (select)   | Generic LLM  | Coping   |   0.291971 | 0.740876 |  0.0381094  |   0.0360928   |    -0.0805422  | 274 |
| GPT-5          | gpt-5                     | Zero-shot (select)   | Generic LLM  | Quitting |   0.20438  | 0.536496 |  0.0287796  |   0.035482    |    -0.0408242  | 274 |
| GPT-5          | gpt-5                     | Few-shot (all)       | Generic LLM  | Content  |   0.354015 | 0.79927  |  0.0518104  |   0.0502296   |     0.0133349  | 274 |
| GPT-5          | gpt-5                     | Few-shot (all)       | Generic LLM  | Coping   |   0.19708  | 0.675182 | -0.0486761  |   0.0304401   |    -0.0830867  | 274 |
| GPT-5          | gpt-5                     | Few-shot (all)       | Generic LLM  | Quitting |   0.222628 | 0.620438 |  0.0251553  |   0.0387813   |    -0.0741335  | 274 |
| GPT-5          | gpt-5                     | Few-shot (select)    | Generic LLM  | Content  |   0.343066 | 0.784672 |  0.0499123  |   0.0248751   |     0.036728   | 274 |
| GPT-5          | gpt-5                     | Few-shot (select)    | Generic LLM  | Coping   |   0.244526 | 0.635036 |  0.0231308  |  -0.0116152   |    -0.150452   | 274 |
| GPT-5          | gpt-5                     | Few-shot (select)    | Generic LLM  | Quitting |   0.218978 | 0.587591 |  0.0322495  |   0.0782197   |    -0.0499779  | 274 |
| GPT-5          | gpt-5                     | Continuous (NL)      | Generic LLM  | Content  |   0.368613 | 0.883212 |  0.0336181  |   0.028306    |     0.121035   | 274 |
| GPT-5          | gpt-5                     | Continuous (NL)      | Generic LLM  | Coping   |   0.251825 | 0.693431 | -0.00371681 |  -0.0355166   |    -0.206101   | 274 |
| GPT-5          | gpt-5                     | Continuous (NL)      | Generic LLM  | Quitting |   0.178832 | 0.50365  |  0.0116866  |   0.0587284   |    -0.102566   | 274 |
| GPT-5          | gpt-5                     | Digital Twin (70/30) | Digital Twin | Content  |   0.453416 | 0.872671 |  0.208315   |   0.329834    |     0.142857   | 322 |
| GPT-5          | gpt-5                     | Digital Twin (70/30) | Digital Twin | Coping   |   0.377709 | 0.823529 |  0.165192   |   0.331202    |     0.2        | 323 |
| GPT-5          | gpt-5                     | Digital Twin (70/30) | Digital Twin | Quitting |   0.405573 | 0.845201 |  0.210118   |   0.431014    |     0.5        | 323 |
| DeepSeek-R1    | deepseek_deepseek-r1-0528 | Zero-shot (all)      | Generic LLM  | Content  |   0.335766 | 0.883212 |  0.00568261 |   0.143762    |     0.170992   | 274 |
| DeepSeek-R1    | deepseek_deepseek-r1-0528 | Zero-shot (all)      | Generic LLM  | Coping   |   0.244526 | 0.675182 |  0.0169509  |   0.0775774   |    -0.0667212  | 274 |
| DeepSeek-R1    | deepseek_deepseek-r1-0528 | Zero-shot (all)      | Generic LLM  | Quitting |   0.160584 | 0.478102 |  0.0248205  |   0.0219247   |    -0.0625534  | 274 |
| DeepSeek-R1    | deepseek_deepseek-r1-0528 | Zero-shot (select)   | Generic LLM  | Content  |   0.368613 | 0.864964 |  0.0389279  |   0.0531824   |     0.0494629  | 274 |
| DeepSeek-R1    | deepseek_deepseek-r1-0528 | Zero-shot (select)   | Generic LLM  | Coping   |   0.284672 | 0.773723 |  0.0112492  |   0.0706855   |     0.0812068  | 274 |
| DeepSeek-R1    | deepseek_deepseek-r1-0528 | Zero-shot (select)   | Generic LLM  | Quitting |   0.240876 | 0.60219  |  0.0543572  |   0.1027      |    -0.0113718  | 274 |
| DeepSeek-R1    | deepseek_deepseek-r1-0528 | Continuous (NL)      | Generic LLM  | Content  |   0.350365 | 0.824818 |  0.0423531  |   0.0467713   |     0.147616   | 274 |
| DeepSeek-R1    | deepseek_deepseek-r1-0528 | Continuous (NL)      | Generic LLM  | Coping   |   0.284672 | 0.726277 |  0.0369759  |   0.0577955   |     0.0211752  | 274 |
| DeepSeek-R1    | deepseek_deepseek-r1-0528 | Continuous (NL)      | Generic LLM  | Quitting |   0.211679 | 0.521898 |  0.0422836  |   0.0476887   |    -0.0122089  | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Zero-shot (all)      | Generic LLM  | Content  |   0.368613 | 0.832117 |  0.0300984  |   0.0086487   |    -0.0091263  | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Zero-shot (all)      | Generic LLM  | Coping   |   0.321168 | 0.784672 |  0.0334364  |   0.0268234   |    -0.070235   | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Zero-shot (all)      | Generic LLM  | Quitting |   0.281022 | 0.70073  |  0.0530175  |   0.0610244   |    -0.0390902  | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Zero-shot (select)   | Generic LLM  | Content  |   0.361314 | 0.824818 |  0.026218   |   0.111315    |     0.00276251 | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Zero-shot (select)   | Generic LLM  | Coping   |   0.306569 | 0.788321 |  0.0170496  |   0.0676127   |     0.155017   | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Zero-shot (select)   | Generic LLM  | Quitting |   0.288321 | 0.715328 |  0.0491529  |   0.127313    |     0.0293345  | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Few-shot (all)       | Generic LLM  | Content  |   0.313869 | 0.781022 | -0.0191315  |   0.0558246   |    -0.150871   | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Few-shot (all)       | Generic LLM  | Coping   |   0.291971 | 0.733577 |  0.012961   |  -0.000749139 |    -0.143436   | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Few-shot (all)       | Generic LLM  | Quitting |   0.317518 | 0.733577 |  0.0681628  |   0.110546    |     0.127576   | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Few-shot (select)    | Generic LLM  | Content  |   0.324818 | 0.79562  |  0.0284433  |   0.115369    |    -0.0194744  | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Few-shot (select)    | Generic LLM  | Coping   |   0.284672 | 0.748175 |  0.0289837  |   0.112511    |    -0.0654736  | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Few-shot (select)    | Generic LLM  | Quitting |   0.313869 | 0.722628 |  0.0812749  |   0.160608    |    -0.0165659  | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Continuous (NL)      | Generic LLM  | Content  |   0.343066 | 0.894161 | -0.0167395  |   0.044913    |     0.0662164  | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Continuous (NL)      | Generic LLM  | Coping   |   0.313869 | 0.824818 |  0.0136147  |   0.137537    |     0.0607466  | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Continuous (NL)      | Generic LLM  | Quitting |   0.259124 | 0.784672 |  0.0104256  |   0.166734    |     0.065838   | 274 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Digital Twin (70/30) | Digital Twin | Content  |   0.481366 | 0.875776 |  0.229952   |   0.340953    |     0.2        | 322 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Digital Twin (70/30) | Digital Twin | Coping   |   0.436533 | 0.839009 |  0.226459   |   0.368743    |     0.333333   | 323 |
| Grok-4-Fast    | x-ai_grok-4-fast          | Digital Twin (70/30) | Digital Twin | Quitting |   0.436533 | 0.851393 |  0.231535   |   0.421138    |     0.333333   | 323 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Zero-shot (all)      | Generic LLM  | Content  |   0.306569 | 0.791971 | -0.0372998  |  -0.00978184  |     0.0255663  | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Zero-shot (all)      | Generic LLM  | Coping   |   0.343066 | 0.766423 |  0.0661744  |   0.0300085   |    -0.0755814  | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Zero-shot (all)      | Generic LLM  | Quitting |   0.313869 | 0.770073 |  0.050855   |   0.0457341   |     0.0556721  | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Zero-shot (select)   | Generic LLM  | Content  |   0.379562 | 0.80292  |  0.0698696  |   0.103643    |     0.163332   | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Zero-shot (select)   | Generic LLM  | Coping   |   0.332117 | 0.781022 |  0.0503409  |   0.0635874   |     0.0512821  | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Zero-shot (select)   | Generic LLM  | Quitting |   0.284672 | 0.777372 |  0.0221057  |   0.0825953   |     0.092484   | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Few-shot (all)       | Generic LLM  | Content  |   0.324818 | 0.762774 |  0.0175405  |   0.0947055   |     0.0323038  | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Few-shot (all)       | Generic LLM  | Coping   |   0.288321 | 0.69708  |  0.0332211  |   0.0571381   |    -0.00372152 | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Few-shot (all)       | Generic LLM  | Quitting |   0.288321 | 0.70438  |  0.0294454  |   0.10091     |     0.0809184  | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Few-shot (select)    | Generic LLM  | Content  |   0.375912 | 0.777372 |  0.107101   |   0.0816883   |     0.00628314 | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Few-shot (select)    | Generic LLM  | Coping   |   0.266423 | 0.660584 |  0.0264795  |   0.0282234   |    -0.0304942  | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Few-shot (select)    | Generic LLM  | Quitting |   0.291971 | 0.70438  |  0.0546347  |   0.107083    |    -0.0145406  | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Continuous (NL)      | Generic LLM  | Content  |   0.310219 | 0.751825 | -0.0182469  |   0.0238604   |     0.0289476  | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Continuous (NL)      | Generic LLM  | Coping   |   0.310219 | 0.748175 |  0.0200768  |   0.00185483  |    -0.0116279  | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Continuous (NL)      | Generic LLM  | Quitting |   0.255474 | 0.781022 | -0.0101017  |   0.0377112   |     0.0455389  | 274 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Digital Twin (70/30) | Digital Twin | Content  |   0.42236  | 0.829193 |  0.162619   |   0.256417    |     0.333333   | 322 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Digital Twin (70/30) | Digital Twin | Coping   |   0.433437 | 0.820433 |  0.233456   |   0.344308    |     0.714286   | 323 |
| Gemini-2.5-Pro | gemini-2.5-pro            | Digital Twin (70/30) | Digital Twin | Quitting |   0.448916 | 0.80805  |  0.256174   |   0.398502    |     1          | 323 |