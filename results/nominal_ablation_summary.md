# Ablation — nominal location on top of raw coordinates (TabPFN-3.5, zero-shot)

Difference = variant − plain table; negative is better. CI95 by block bootstrap (2,000 resamples of the spatial blocks); Wilcoxon paired over blocks.

| Leg | Input | RMSE_ln | MAPE | R²_ln | Moran I | ΔRMSE_ln [CI95] | ΔMAPE p.p. [CI95] | Wilcoxon p | blocks better |
|---|---|---|---|---|---|---|---|---|---|
| 1 spatial CV (Level A) | plain table | 0.3938 | 32.2% | 0.346 | 0.348 | — | — | — | — |
| 1 spatial CV (Level A) | + bairro | 0.3934 | 32.3% | 0.347 | 0.351 | -0.0004 [-0.0037; +0.0036] | +0.10 [-0.20; +0.33] | 1.000 | 4/10 |
| 1 spatial CV (Level A) | + cep | 0.3989 | 32.2% | 0.329 | 0.367 | +0.0051 [-0.0106; +0.0259] | +0.05 [-1.16; +1.80] | 0.846 | 5/10 |
| 1 spatial CV (Level A) | + bairro + cep | 0.3912 | 32.4% | 0.355 | 0.358 | -0.0027 [-0.0110; +0.0101] | +0.24 [-0.76; +1.94] | 0.375 | 6/10 |
| 2 out-of-time (Level A 24k -> 2026) | plain table | 0.2750 | 20.4% | 0.622 | 0.122 | — | — | — | — |
| 2 out-of-time (Level A 24k -> 2026) | + bairro | 0.2768 | 20.5% | 0.617 | 0.122 | +0.0019 [+0.0014; +0.0025] | +0.12 [+0.07; +0.18] | 0.002 | 0/10 |
| 2 out-of-time (Level A 24k -> 2026) | + cep | 0.2762 | 20.5% | 0.618 | 0.128 | +0.0012 [+0.0003; +0.0021] | +0.13 [+0.07; +0.19] | 0.105 | 2/10 |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep | 0.2766 | 20.7% | 0.617 | 0.130 | +0.0017 [+0.0004; +0.0027] | +0.26 [+0.18; +0.34] | 0.084 | 2/10 |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep (categorical) | 0.2766 | 20.7% | 0.617 | 0.130 | +0.0017 [+0.0004; +0.0027] | +0.26 [+0.18; +0.34] | 0.084 | 2/10 |
| 2 out-of-time (full 82k -> 2026) | plain table | 0.2649 | 19.5% | 0.649 | 0.092 | — | — | — | — |
| 2 out-of-time (full 82k -> 2026) | + bairro | 0.2647 | 19.5% | 0.649 | 0.093 | -0.0001 [-0.0006; +0.0003] | +0.01 [-0.05; +0.05] | 0.557 | 6/10 |
| 2 out-of-time (full 82k -> 2026) | + cep | 0.2636 | 19.4% | 0.652 | 0.090 | -0.0012 [-0.0018; -0.0006] | -0.04 [-0.08; -0.01] | 0.010 | 8/10 |
| 2 out-of-time (full 82k -> 2026) | + bairro + cep | 0.2629 | 19.5% | 0.654 | 0.088 | -0.0020 [-0.0031; -0.0009] | -0.03 [-0.14; +0.06] | 0.014 | 8/10 |

## Mechanism check (out-of-time leg)

The same difference, split by whether the 2026 row's value occurs in the training base.

| Leg | Input | Subset of 2026 | n (share) | RMSE plain | RMSE variant | ΔRMSE_ln [CI95] |
|---|---|---|---|---|---|---|
| 2 out-of-time (Level A 24k -> 2026) | + bairro | bairro seen in training | 24,369 (51%) | 0.2667 | 0.2682 | +0.0015 [+0.0006; +0.0023] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro | bairro not seen | 5,510 (12%) | 0.2843 | 0.2887 | +0.0044 [+0.0027; +0.0062] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro | bairro empty | 17,931 (38%) | 0.2830 | 0.2846 | +0.0016 [+0.0006; +0.0028] |
| 2 out-of-time (Level A 24k -> 2026) | + cep | cep seen in training | 37,113 (78%) | 0.2630 | 0.2639 | +0.0010 [+0.0001; +0.0016] |
| 2 out-of-time (Level A 24k -> 2026) | + cep | cep not seen | 10,697 (22%) | 0.3130 | 0.3149 | +0.0019 [-0.0003; +0.0042] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep | bairro seen in training | 24,369 (51%) | 0.2667 | 0.2678 | +0.0011 [-0.0007; +0.0026] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep | bairro not seen | 5,510 (12%) | 0.2843 | 0.2877 | +0.0034 [+0.0018; +0.0045] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep | bairro empty | 17,931 (38%) | 0.2830 | 0.2849 | +0.0019 [+0.0000; +0.0029] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep | cep seen in training | 37,113 (78%) | 0.2630 | 0.2642 | +0.0012 [-0.0001; +0.0023] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep | cep not seen | 10,697 (22%) | 0.3130 | 0.3161 | +0.0031 [+0.0011; +0.0050] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep (categorical) | bairro seen in training | 24,369 (51%) | 0.2667 | 0.2678 | +0.0011 [-0.0007; +0.0026] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep (categorical) | bairro not seen | 5,510 (12%) | 0.2843 | 0.2877 | +0.0034 [+0.0018; +0.0045] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep (categorical) | bairro empty | 17,931 (38%) | 0.2830 | 0.2849 | +0.0019 [+0.0000; +0.0029] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep (categorical) | cep seen in training | 37,113 (78%) | 0.2630 | 0.2642 | +0.0012 [-0.0001; +0.0023] |
| 2 out-of-time (Level A 24k -> 2026) | + bairro + cep (categorical) | cep not seen | 10,697 (22%) | 0.3130 | 0.3161 | +0.0031 [+0.0011; +0.0050] |
| 2 out-of-time (full 82k -> 2026) | + bairro | bairro seen in training | 27,210 (57%) | 0.2568 | 0.2558 | -0.0011 [-0.0016; -0.0006] |
| 2 out-of-time (full 82k -> 2026) | + bairro | bairro not seen | 2,669 (6%) | 0.2932 | 0.2922 | -0.0009 [-0.0026; +0.0005] |
| 2 out-of-time (full 82k -> 2026) | + bairro | bairro empty | 17,931 (38%) | 0.2723 | 0.2736 | +0.0013 [+0.0005; +0.0023] |
| 2 out-of-time (full 82k -> 2026) | + cep | cep seen in training | 43,284 (91%) | 0.2560 | 0.2547 | -0.0013 [-0.0019; -0.0007] |
| 2 out-of-time (full 82k -> 2026) | + cep | cep not seen | 4,526 (9%) | 0.3380 | 0.3374 | -0.0006 [-0.0023; +0.0011] |
| 2 out-of-time (full 82k -> 2026) | + bairro + cep | bairro seen in training | 27,210 (57%) | 0.2568 | 0.2549 | -0.0019 [-0.0040; +0.0002] |
| 2 out-of-time (full 82k -> 2026) | + bairro + cep | bairro not seen | 2,669 (6%) | 0.2932 | 0.2884 | -0.0047 [-0.0129; +0.0030] |
| 2 out-of-time (full 82k -> 2026) | + bairro + cep | bairro empty | 17,931 (38%) | 0.2723 | 0.2707 | -0.0016 [-0.0041; +0.0010] |
| 2 out-of-time (full 82k -> 2026) | + bairro + cep | cep seen in training | 43,284 (91%) | 0.2560 | 0.2541 | -0.0019 [-0.0033; -0.0008] |
| 2 out-of-time (full 82k -> 2026) | + bairro + cep | cep not seen | 4,526 (9%) | 0.3380 | 0.3357 | -0.0023 [-0.0066; +0.0017] |
