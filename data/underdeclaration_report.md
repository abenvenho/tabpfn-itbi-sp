# Under-declaration diagnostics — ITBI-SP base (v3)

The ITBI transaction price is **self-declared by the taxpayer**. In São Paulo
the tax base is max(declared price, VVR): declaring below the VVR saves no
tax, so the economic incentive is to declare **exactly the VVR** when the
true price is higher — visible as bunching of the declared/VVR ratio at 1.
Financed deals go through the lender's underwriting (bank appraisal, loan
contract) and are less exposed; a loan larger than the declared price
(LTV > 1) would be a direct red flag.

Since pipeline v3, cash deals pegged to the VVR (ratio in [0.99, 1.01],
not financed) are excluded upstream, so this report shows the RESIDUAL
picture after that exclusion — the bunching that motivated the rule is
documented in `data/filter_report.md`. Reminder: the gross-error ratio
filter [0.3, 5] is applied upstream as well.


## 2025 (training) — declared/VVR ratio by property type

| type | n | VVR>0 | P25 | median | P75 | ratio<0.7 | 0.7<=ratio<0.95 | pegged to 1 (+/-5%) | exactly 1 (+/-1%) | ratio>1.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| apartment | 60067 | 99.3% | 1.07 | 1.31 | 1.61 | 3.3% | 11.6% | 7.9% | 0.7% | 77.1% |
| house | 15940 | 97.8% | 0.95 | 1.25 | 1.66 | 8.0% | 17.1% | 9.4% | 0.8% | 65.6% |
| commercial | 6180 | 91.4% | 0.70 | 0.92 | 1.26 | 24.9% | 29.1% | 8.0% | 0.2% | 37.9% |
| TOTAL | 82187 | 98.4% | 1.02 | 1.28 | 1.60 | 5.7% | 13.9% | 8.2% | 0.7% | 72.2% |

## 2026 (test) — declared/VVR ratio by property type

| type | n | VVR>0 | P25 | median | P75 | ratio<0.7 | 0.7<=ratio<0.95 | pegged to 1 (+/-5%) | exactly 1 (+/-1%) | ratio>1.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| apartment | 35354 | 99.4% | 1.05 | 1.29 | 1.58 | 3.4% | 12.6% | 8.6% | 0.8% | 75.4% |
| house | 9429 | 97.2% | 0.91 | 1.20 | 1.58 | 9.8% | 19.0% | 9.0% | 0.8% | 62.2% |
| commercial | 3027 | 90.0% | 0.66 | 0.90 | 1.25 | 29.4% | 25.4% | 8.9% | 0.3% | 36.3% |
| TOTAL | 47810 | 98.4% | 1.01 | 1.26 | 1.56 | 6.1% | 14.6% | 8.7% | 0.8% | 70.6% |

## Ratio bunching around 1 — 2025 base (2% bins)

| bin | n | % of base |
|---|---|---|
| [0.80, 0.82) | 895 | 1.09% |
| [0.82, 0.84) | 859 | 1.05% |
| [0.84, 0.86) | 1000 | 1.22% |
| [0.86, 0.88) | 1007 | 1.23% |
| [0.88, 0.90) | 1055 | 1.28% |
| [0.90, 0.92) | 1081 | 1.32% |
| [0.92, 0.94) | 1234 | 1.50% |
| [0.94, 0.96) | 1252 | 1.52% |
| [0.96, 0.98) | 1391 | 1.69% |
| [0.98, 1.00) | 1000 | 1.22% |
| [1.00, 1.02) | 1181 | 1.44% |
| [1.02, 1.04) | 1613 | 1.96% |
| [1.04, 1.06) | 1527 | 1.86% |
| [1.06, 1.08) | 1469 | 1.79% |
| [1.08, 1.10) | 1439 | 1.75% |
| [1.10, 1.12) | 1555 | 1.89% |
| [1.12, 1.14) | 1564 | 1.90% |
| [1.14, 1.16) | 1564 | 1.90% |
| [1.16, 1.18) | 1655 | 2.01% |
| [1.18, 1.20) | 1541 | 1.87% |

## 2025 (training) — financed vs cash

| type | % financed | median ratio (financed) | median ratio (cash) | pegged +/-5% (fin.) | pegged +/-5% (cash) | ratio<1 (fin.) | ratio<1 (cash) | median R$/m2 (fin.) | median R$/m2 (cash) |
|---|---|---|---|---|---|---|---|---|---|
| apartment | 35.7% | 1.34 | 1.29 | 8.1% | 7.7% | 14.3% | 20.9% | 4,746 | 5,143 |
| house | 37.1% | 1.40 | 1.15 | 8.2% | 10.1% | 18.3% | 36.1% | 4,640 | 4,352 |
| commercial | 6.4% | 0.93 | 0.92 | 12.5% | 7.7% | 58.7% | 57.8% | 4,416 | 4,324 |
| TOTAL | 33.8% | 1.34 | 1.24 | 8.2% | 8.2% | 15.8% | 27.3% | 4,712 | 4,916 |

## 2026 (test) — financed vs cash

| type | % financed | median ratio (financed) | median ratio (cash) | pegged +/-5% (fin.) | pegged +/-5% (cash) | ratio<1 (fin.) | ratio<1 (cash) | median R$/m2 (fin.) | median R$/m2 (cash) |
|---|---|---|---|---|---|---|---|---|---|
| apartment | 44.2% | 1.32 | 1.26 | 8.6% | 8.6% | 15.5% | 23.2% | 4,938 | 5,272 |
| house | 44.5% | 1.32 | 1.08 | 8.4% | 9.4% | 22.4% | 41.8% | 4,778 | 4,471 |
| commercial | 7.9% | 0.90 | 0.90 | 6.6% | 9.1% | 56.4% | 58.8% | 4,412 | 4,500 |
| TOTAL | 41.9% | 1.32 | 1.21 | 8.5% | 8.8% | 17.4% | 30.0% | 4,902 | 5,039 |

## Loan-to-declared-price (LTV) — 2025 base

| type | n financed w/ amount | median LTV | LTV>0.9 | LTV>1 (loan above declared price) | LTV>1.2 |
|---|---|---|---|---|---|
| apartment | 21446 | 0.69 | 5.0% | 0 (0.0%) | 0 (0.0%) |
| house | 5915 | 0.67 | 4.1% | 0 (0.0%) | 0 (0.0%) |
| commercial | 393 | 0.70 | 24.7% | 0 (0.0%) | 0 (0.0%) |
| TOTAL | 27754 | 0.68 | 5.0% | 0 (0.0%) | 0 (0.0%) |

## Loan-to-declared-price (LTV) — 2026 base

| type | n financed w/ amount | median LTV | LTV>0.9 | LTV>1 (loan above declared price) | LTV>1.2 |
|---|---|---|---|---|---|
| apartment | 15609 | 0.73 | 4.8% | 0 (0.0%) | 0 (0.0%) |
| house | 4200 | 0.71 | 2.7% | 0 (0.0%) | 0 (0.0%) |
| commercial | 240 | 0.71 | 20.4% | 0 (0.0%) | 0 (0.0%) |
| TOTAL | 20049 | 0.72 | 4.6% | 0 (0.0%) | 0 (0.0%) |

## Combined residual signals — 2025 base

| signal | n (2025) | % of base |
|---|---|---|
| ratio pegged to 1 (+/-5%) | 6612 | 8.0% |
| ratio < 0.95 | 15893 | 19.3% |
| LTV > 1 (financed only) | 0 | 0.0% |
| pegged AND cash | 4348 | 5.3% |
| pegged AND financed | 2264 | 2.8% |
