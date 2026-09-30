# A name in place of the coordinates — TabPFN-3.5 zero-shot against every model

TabPFN-3.5 rows marked *no lat/lon* drop latitude and longitude from the plain table and carry the named descriptor instead; rows marked *no station distance* also drop the distance to the nearest station, which is computed from the coordinates, so that the postal code is the only spatial information left. The baselines are untouched: SAR keeps its weights matrix, XGBoost its k-NN lag, rotated coordinates and station distance. Everything else is unchanged (same rows, folds, seed). Differences are RMSE_ln of the row minus RMSE_ln of the reference, negative is better, CI95 by block bootstrap (2,000 resamples of the spatial blocks). *Signal recovered* places the row between the floor of its own family (0 %) and the plain table (100 %) on the RMSE_ln scale.

## Leg 1 — leave-one-block-out CV, Level A (24k)

| Model | RMSE_ln | MAPE | R²_ln | Moran I | signal recovered | Δ vs TabPFN plain [CI95] | Δ vs XGBoost+lag [CI95] | Δ vs SAR [CI95] |
|---|---|---|---|---|---|---|---|---|
| OLS hedonic | 0.4705 | 39.4% | 0.066 | 0.390 | — | — | — | — |
| SAR lag | 0.4476 | 38.4% | 0.155 | 0.363 | — | — | — | — |
| XGBoost + rotated coords + k-NN-8 lag | 0.3846 | 32.3% | 0.376 | 0.309 | — | — | — | — |
| **TabPFN-3.5, plain table (lat/lon)** | 0.3938 | 32.2% | 0.346 | 0.348 | 100 % | — | — | — |
| TabPFN-3.5, no lat/lon: no location (floor) | 0.4341 | 35.6% | 0.205 | 0.393 | 0 % | +0.0403 [+0.0125; +0.0735] (2/10) | +0.0495 [+0.0126; +0.0768] (1/10) | -0.0135 [-0.0696; +0.0262] (8/10) |
| TabPFN-3.5, no lat/lon: postal code | 0.4453 | 37.6% | 0.164 | 0.477 | -28 % | +0.0515 [+0.0005; +0.0988] (2/10) | +0.0607 [-0.0007; +0.1050] (4/10) | -0.0023 [-0.0774; +0.0518] (8/10) |
| TabPFN-3.5, no lat/lon: district field | 0.4289 | 35.0% | 0.224 | 0.372 | 13 % | +0.0351 [+0.0076; +0.0705] (3/10) | +0.0442 [+0.0093; +0.0718] (2/10) | -0.0187 [-0.0732; +0.0195] (7/10) |
| TabPFN-3.5, no lat/lon: district + postal code | 0.4458 | 37.8% | 0.162 | 0.482 | -29 % | +0.0520 [-0.0010; +0.0975] (3/10) | +0.0612 [-0.0024; +0.1076] (4/10) | -0.0018 [-0.0805; +0.0558] (8/10) |
| TabPFN-3.5, no lat/lon: postal code as a number | 0.4268 | 35.6% | 0.232 | 0.432 | 18 % | +0.0330 [-0.0091; +0.0810] (4/10) | +0.0421 [-0.0080; +0.0813] (4/10) | -0.0208 [-0.0858; +0.0282] (8/10) |
| TabPFN-3.5, no lat/lon, no station distance: nothing spatial (strict floor) | 0.4403 | 36.6% | 0.182 | 0.372 | 0 % | +0.0465 [+0.0234; +0.0744] (1/10) | +0.0556 [+0.0252; +0.0788] (1/10) | -0.0073 [-0.0574; +0.0290] (7/10) |
| TabPFN-3.5, no lat/lon, no station distance: postal code only | 0.4568 | 38.6% | 0.120 | 0.502 | -35 % | +0.0630 [+0.0033; +0.1173] (2/10) | +0.0721 [+0.0026; +0.1228] (4/10) | +0.0092 [-0.0735; +0.0688] (8/10) |
| TabPFN-3.5, no lat/lon, no station distance: postal code as a number only | 0.4399 | 36.9% | 0.184 | 0.450 | 1 % | +0.0460 [-0.0009; +0.1015] (2/10) | +0.0552 [+0.0071; +0.0973] (4/10) | -0.0078 [-0.0660; +0.0377] (7/10) |

(n/10) = spatial blocks in which the row has the lower RMSE.

## Leg 2 — fit on Level A (24k), predict 2026

| Model | RMSE_ln | MAPE | R²_ln | Moran I | signal recovered | Δ vs TabPFN plain [CI95] | Δ vs XGBoost+lag [CI95] | Δ vs SAR [CI95] |
|---|---|---|---|---|---|---|---|---|
| OLS hedonic | 0.4036 | 32.4% | 0.185 | 0.408 | — | — | — | — |
| SAR lag | 0.3459 | 27.0% | 0.401 | 0.224 | — | — | — | — |
| XGBoost + rotated coords + k-NN-8 lag | 0.2884 | 21.8% | 0.584 | 0.140 | — | — | — | — |
| **TabPFN-3.5, plain table (lat/lon)** | 0.2750 | 20.4% | 0.622 | 0.122 | 100 % | — | — | — |
| TabPFN-3.5, no lat/lon: no location (floor) | 0.3212 | 23.8% | 0.484 | 0.224 | 0 % | +0.0463 [+0.0301; +0.0620] (0/10) | +0.0328 [+0.0182; +0.0475] (0/10) | -0.0246 [-0.0353; -0.0152] (9/10) |
| TabPFN-3.5, no lat/lon: postal code | 0.2767 | 20.8% | 0.617 | 0.125 | 96 % | +0.0017 [-0.0002; +0.0031] (3/10) | -0.0117 [-0.0135; -0.0102] (10/10) | -0.0692 [-0.0762; -0.0615] (10/10) |
| TabPFN-3.5, no lat/lon: district field | 0.3136 | 23.3% | 0.508 | 0.194 | 16 % | +0.0387 [+0.0244; +0.0525] (0/10) | +0.0252 [+0.0128; +0.0375] (0/10) | -0.0322 [-0.0414; -0.0251] (10/10) |
| TabPFN-3.5, no lat/lon: district + postal code | 0.2780 | 20.8% | 0.613 | 0.124 | 93 % | +0.0030 [+0.0014; +0.0042] (2/10) | -0.0104 [-0.0123; -0.0085] (10/10) | -0.0679 [-0.0751; -0.0598] (10/10) |
| TabPFN-3.5, no lat/lon: postal code as a number | 0.2745 | 20.3% | 0.623 | 0.124 | 101 % | -0.0005 [-0.0018; +0.0007] (8/10) | -0.0140 [-0.0158; -0.0120] (10/10) | -0.0714 [-0.0786; -0.0639] (10/10) |
| TabPFN-3.5, no lat/lon, no station distance: nothing spatial (strict floor) | 0.3515 | 26.7% | 0.382 | 0.297 | 0 % | +0.0766 [+0.0502; +0.1005] (0/10) | +0.0631 [+0.0381; +0.0856] (0/10) | +0.0056 [-0.0146; +0.0227] (7/10) |
| TabPFN-3.5, no lat/lon, no station distance: postal code only | 0.2771 | 20.7% | 0.616 | 0.129 | 97 % | +0.0021 [+0.0006; +0.0032] (2/10) | -0.0114 [-0.0132; -0.0096] (10/10) | -0.0688 [-0.0760; -0.0614] (10/10) |
| TabPFN-3.5, no lat/lon, no station distance: postal code as a number only | 0.2756 | 20.5% | 0.620 | 0.125 | 99 % | +0.0006 [-0.0004; +0.0014] (4/10) | -0.0128 [-0.0145; -0.0109] (10/10) | -0.0703 [-0.0775; -0.0627] (10/10) |

(n/10) = spatial blocks in which the row has the lower RMSE.

Split of the 2026 rows by whether the value occurs in the training base:

| Descriptor | Subset of 2026 | n (share) | RMSE floor | RMSE descriptor | RMSE plain table | signal recovered |
|---|---|---|---|---|---|---|
| postal code (no lat/lon) | cep seen in training | 37,113 (78%) | 0.3013 | 0.2646 | 0.2630 | 96 % |
| postal code (no lat/lon) | cep not seen | 10,697 (22%) | 0.3824 | 0.3150 | 0.3130 | 97 % |
| district field (no lat/lon) | bairro seen in training | 24,369 (51%) | 0.3106 | 0.2887 | 0.2667 | 50 % |
| district field (no lat/lon) | bairro not seen | 5,510 (12%) | 0.3264 | 0.3291 | 0.2843 | -7 % |
| district field (no lat/lon) | bairro empty | 17,931 (38%) | 0.3335 | 0.3402 | 0.2830 | -13 % |
| district + postal code (no lat/lon) | cep seen in training | 37,113 (78%) | 0.3013 | 0.2658 | 0.2630 | 93 % |
| district + postal code (no lat/lon) | cep not seen | 10,697 (22%) | 0.3824 | 0.3166 | 0.3130 | 95 % |
| postal code as a number (no lat/lon) | cep seen in training | 37,113 (78%) | 0.3013 | 0.2623 | 0.2630 | 102 % |
| postal code as a number (no lat/lon) | cep not seen | 10,697 (22%) | 0.3824 | 0.3129 | 0.3130 | 100 % |
| postal code only (no lat/lon, no station distance) | cep seen in training | 37,113 (78%) | 0.3359 | 0.2648 | 0.2630 | 98 % |
| postal code only (no lat/lon, no station distance) | cep not seen | 10,697 (22%) | 0.4010 | 0.3160 | 0.3130 | 97 % |
| postal code as a number only (no lat/lon, no station distance) | cep seen in training | 37,113 (78%) | 0.3359 | 0.2635 | 0.2630 | 99 % |
| postal code as a number only (no lat/lon, no station distance) | cep not seen | 10,697 (22%) | 0.4010 | 0.3137 | 0.3130 | 99 % |

## Leg 2 — fit on the full 2025 base (82k), predict 2026

| Model | RMSE_ln | MAPE | R²_ln | Moran I | signal recovered | Δ vs TabPFN plain [CI95] | Δ vs XGBoost+lag [CI95] | Δ vs SAR [CI95] |
|---|---|---|---|---|---|---|---|---|
| OLS hedonic | 0.4015 | 32.4% | 0.194 | 0.397 | — | — | — | — |
| SAR lag | 0.3256 | 25.0% | 0.470 | 0.134 | — | — | — | — |
| XGBoost + rotated coords + k-NN-8 lag | 0.2940 | 21.5% | 0.568 | 0.107 | — | — | — | — |
| **TabPFN-3.5, plain table (lat/lon)** | 0.2649 | 19.5% | 0.649 | 0.092 | 100 % | — | — | — |
| TabPFN-3.5, no lat/lon: no location (floor) | 0.2955 | 21.4% | 0.563 | 0.143 | 0 % | +0.0306 [+0.0212; +0.0391] (0/10) | +0.0015 [-0.0089; +0.0105] (6/10) | -0.0301 [-0.0394; -0.0235] (10/10) |
| TabPFN-3.5, no lat/lon: postal code | 0.2643 | 19.5% | 0.650 | 0.089 | 102 % | -0.0005 [-0.0020; +0.0008] (6/10) | -0.0297 [-0.0337; -0.0261] (10/10) | -0.0613 [-0.0668; -0.0557] (10/10) |
| TabPFN-3.5, no lat/lon: district field | 0.2896 | 21.1% | 0.580 | 0.124 | 19 % | +0.0247 [+0.0161; +0.0327] (0/10) | -0.0044 [-0.0136; +0.0038] (7/10) | -0.0360 [-0.0446; -0.0299] (10/10) |
| TabPFN-3.5, no lat/lon: district + postal code | 0.2621 | 19.5% | 0.656 | 0.087 | 109 % | -0.0027 [-0.0046; -0.0012] (8/10) | -0.0318 [-0.0361; -0.0281] (10/10) | -0.0635 [-0.0692; -0.0581] (10/10) |
| TabPFN-3.5, no lat/lon: postal code as a number | 0.2632 | 19.4% | 0.653 | 0.088 | 105 % | -0.0016 [-0.0024; -0.0008] (8/10) | -0.0307 [-0.0341; -0.0276] (10/10) | -0.0624 [-0.0677; -0.0568] (10/10) |
| TabPFN-3.5, no lat/lon, no station distance: nothing spatial (strict floor) | 0.3272 | 24.4% | 0.464 | 0.226 | 0 % | +0.0623 [+0.0419; +0.0806] (0/10) | +0.0332 [+0.0125; +0.0519] (1/10) | +0.0016 [-0.0171; +0.0168] (7/10) |
| TabPFN-3.5, no lat/lon, no station distance: postal code only | 0.2684 | 19.8% | 0.639 | 0.091 | 94 % | +0.0036 [+0.0027; +0.0045] (0/10) | -0.0255 [-0.0292; -0.0222] (10/10) | -0.0572 [-0.0626; -0.0514] (10/10) |
| TabPFN-3.5, no lat/lon, no station distance: postal code as a number only | 0.2660 | 19.6% | 0.646 | 0.092 | 98 % | +0.0012 [+0.0003; +0.0020] (2/10) | -0.0279 [-0.0315; -0.0248] (10/10) | -0.0595 [-0.0649; -0.0541] (10/10) |

(n/10) = spatial blocks in which the row has the lower RMSE.

Split of the 2026 rows by whether the value occurs in the training base:

| Descriptor | Subset of 2026 | n (share) | RMSE floor | RMSE descriptor | RMSE plain table | signal recovered |
|---|---|---|---|---|---|---|
| postal code (no lat/lon) | cep seen in training | 43,284 (91%) | 0.2815 | 0.2555 | 0.2560 | 102 % |
| postal code (no lat/lon) | cep not seen | 4,526 (9%) | 0.4055 | 0.3374 | 0.3380 | 101 % |
| district field (no lat/lon) | bairro seen in training | 27,210 (57%) | 0.2828 | 0.2673 | 0.2568 | 60 % |
| district field (no lat/lon) | bairro not seen | 2,669 (6%) | 0.3225 | 0.3286 | 0.2932 | -21 % |
| district field (no lat/lon) | bairro empty | 17,931 (38%) | 0.3098 | 0.3148 | 0.2723 | -13 % |
| district + postal code (no lat/lon) | cep seen in training | 43,284 (91%) | 0.2815 | 0.2533 | 0.2560 | 111 % |
| district + postal code (no lat/lon) | cep not seen | 4,526 (9%) | 0.4055 | 0.3348 | 0.3380 | 105 % |
| postal code as a number (no lat/lon) | cep seen in training | 43,284 (91%) | 0.2815 | 0.2545 | 0.2560 | 106 % |
| postal code as a number (no lat/lon) | cep not seen | 4,526 (9%) | 0.4055 | 0.3358 | 0.3380 | 103 % |
| postal code only (no lat/lon, no station distance) | cep seen in training | 43,284 (91%) | 0.3158 | 0.2596 | 0.2560 | 94 % |
| postal code only (no lat/lon, no station distance) | cep not seen | 4,526 (9%) | 0.4202 | 0.3413 | 0.3380 | 96 % |
| postal code as a number only (no lat/lon, no station distance) | cep seen in training | 43,284 (91%) | 0.3158 | 0.2570 | 0.2560 | 98 % |
| postal code as a number only (no lat/lon, no station distance) | cep not seen | 4,526 (9%) | 0.4202 | 0.3403 | 0.3380 | 97 % |

