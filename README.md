# Location, location, location

*TabPFN-3.5 with the plain table vs models that see space explicitly, on 82,187 real transaction
prices from São Paulo, tested one year ahead on 47,810 more.*

Prior Labs TabPFN-3.5 Hackathon entry · **Live app: [tabpfn-itbi-sp.streamlit.app](https://tabpfn-itbi-sp.streamlit.app)** · **Video: [youtu.be/S1wQqxKR-hI](https://youtu.be/S1wQqxKR-hI)** · [app manual](app/MANUAL.md) · Apache-2.0

---

Ask any appraiser what sets the value of a property and you get the old line: three things,
*location, location, location*. Spatial econometrics took it seriously and built machinery around
it: a weights matrix over the neighbours, a spatial lag, a ρ to estimate. Gradient boosting needs the
same help, hand-fed as neighbourhood features. It works. It is also a lot of engineering that has to be
redone for every city and every dataset.

This repository asks a blunt question. What if you give a tabular foundation model latitude and
longitude as two ordinary numeric columns, nothing else about space, and let it figure location out on
its own? Does it match the specialists that were built for exactly this?

## What TabPFN-3.5 can and cannot do with location

Five models, two tests, every comparison paired over the same ten spatial blocks. The verdicts for
TabPFN-3.5:

| Setting | TabPFN-3.5 gets location from | vs XGBoost + spatial lag | vs geographically weighted XGBoost | vs SAR lag | Moran's I of its residuals |
|---|---|---|---|---|---|
| Same city, one year ahead | latitude and longitude | **win** · 10/10 | **win** · 9/10 | **win** · 10/10 | 0.092 · below every specialist |
| Same city, one year ahead | the postal code alone | **win** · 10/10 | **win** · 9/10 | **win** · 10/10 | 0.091 · below every specialist |
| District never seen | latitude and longitude | tie · 7/10 | tie · 6–7/10 | **win** · 8/10 | 0.342–0.348 · above both XGBoost |
| District never seen | coordinates + unit and building text | tie · 7/10 | **win** · 9/10 | **win** · 9/10 | 0.331 · above both XGBoost |
| District never seen | the postal code alone | **loss** · 4/10 | tie · 4/10 | tie · 8/10 | 0.502 · highest of any model |

*Win or loss: the 95 % block-bootstrap interval of the pooled RMSE difference (ln) excludes zero,
against each seed of the opponent (TabPFN-3.5 enters with seed 42; with the plain table in the unseen
district, that is the weakest of its three seeds); tie: the interval includes zero. n/10: spatial blocks in which
TabPFN-3.5 has the lower RMSE. One year ahead: fitted on the 82,187 transactions of 2025, scored on
the 47,810 of 2026 (fitted on 24,000 rows instead, it wins 10 of 10 blocks against every opponent with
either input). District never seen: leave-one-block-out CV on 24,000 transactions of 2025. Postal code
alone: latitude, longitude and the distance to the nearest station, which is computed from them, are
removed. Moran's I of the specialists: 0.103–0.166 one year ahead, 0.309–0.363 in the unseen
district; ranges are over seeds.*

**1. Inside a market it has seen, TabPFN-3.5 models location by itself, and better than the
specialists.** Given latitude and longitude as two plain numeric columns, it priced the 2026
transactions with lower error than the three models built to capture spatial dependence (RMSE 0.265
in ln against 0.283–0.326; MAPE 19.5 % against 20.9–25.0 %) and left less spatial autocorrelation in
its errors than any of them. No weights matrix, no spatial lag, no neighbourhood features; one fit,
15 seconds.

**2. Inside that market, it needs a location identifier, not coordinates.** With the raw postal code
(CEP) in place of latitude and longitude, a text column with 15,938 values and no geometry, the error
rises by only 0.004 in ln (0.268 against 0.265, though in every block) and every one-year-ahead
verdict holds, residual autocorrelation included. Added on top of the coordinates, the postal code and
the district name change the error by less than 0.006 in ln: both describe the same location.

**3. In a district it has never seen, it ties the best specialists on error and loses on residual
structure.** It ties both XGBoost models on error and beats SAR, but its residuals keep more spatial
autocorrelation than either XGBoost's (0.342–0.348 against 0.309–0.326). By the criteria written
before the runs, this is a partial refutation of the bet. Against XGBoost + lag, the refutation does
not appear on the financed transactions alone, whose prices carry a bank appraisal: there TabPFN-3.5
has the lower error in 8 or 9 of 10 blocks, depending on the XGBoost seed, and less residual
autocorrelation (0.279 against 0.327–0.340). The geographically weighted XGBoost was not re-run on
that subset.

**4. To price a district it has never seen, it needs coordinates.** With the postal code alone, that
test fails: 97–100 % of the codes in a held-out block are new to the model, and the residual
autocorrelation is the highest of any model (0.502). A new code among known ones is priced almost as
well as a known one (96–97 % of the location signal one year ahead); a whole district of new codes
is not.

**5. More information and more compute lower the error in an unseen district; neither removes the
residual gap.** The unit and building text typed on the tax form (`AP 201 E 3VGS`,
`EDIFICIO THE PARK`) lowers the error there by 0.007–0.010 in ln against each seed of the plain
table; thinking mode by 0.002–0.005, and grouping its internal validation by spatial blocks adds
nothing. In every case the residual autocorrelation stays above both XGBoost models (0.331 at best,
against 0.309–0.326).

**Limitation.** Every model under-predicts 2026, a rising market, and the non-linear ones more
(bias +0.04 to +0.10 in ln against +0.02 to +0.03 for OLS and SAR): they hold the last observed price
level. This concerns the trend, not location; appraisal practice would apply a price index.

**In practice.** Inside a market with a sales history, TabPFN-3.5 prices location from the
coordinates, or from the postal code already on the tax form, with no spatial engineering, and does
it better than the specialists tested here. For a region with no sales in the training base,
geocode, and expect XGBoost with a spatial lag to match its error and leave less spatial structure in
the residuals.

The evidence follows: the bet and how it could lose, the data, the protocol and the two main tests.
The [further tests](#further-tests) are summarised after them and written up in full in
[`docs/further_tests.md`](docs/further_tests.md).

## The bet, written so it can lose

> A tabular foundation model that receives only the property attributes and the coordinates as two
> plain numeric columns — no weights matrix, no spatial lag, no engineered neighbourhood features —
> predicts spatially correlated prices as well as, or better than, specialised models that encode
> spatial dependence explicitly.

The comparison is rigged against the bet. The baselines keep every spatial advantage: the SAR lag
model has its k-NN weights matrix and ρ; XGBoost gets a fold-internal k-NN-8 spatial lag, rotated
coordinate axes and nested Optuna tuning; a geographically weighted XGBoost fits its own local models
across the city, with the bandwidth chosen by nested cross-validation. TabPFN-3.5 gets the plain table (built and lot area, age,
finish grade, distance to the nearest station, month, property type, latitude, longitude), zero-shot,
plus a thinking-mode variant. If it matches or beats the specialists, nobody can blame a weak baseline.

It loses if any of these happens: worse pooled RMSE or MAPE; losses in most spatial blocks (paired
Wilcoxon, block bootstrap); or, the sharpest test, **more spatial autocorrelation left in its
out-of-fold residuals** (Moran's I) than the explicitly spatial models leave in theirs.

## Why real prices from São Paulo

Most mass-appraisal benchmarks use listings. Here the target is what buyers declared they paid on
the ITBI form (*Imposto sobre Transmissão de Bens Imóveis*, the municipal transfer tax), month by
month, for the whole city: 82,187 cleaned transactions in 2025 for training, 47,810 from January to
July 2026 for testing. Three property types (apartment, house, commercial), physical attributes from
the IPTU property-tax register, coordinates from the fiscal-block centroids of the GeoSampa cadastre.
This is the larger, noisier and harder version of the question asked in the author's award-winning
study on Belo Horizonte listings ([earlier work](#earlier-work-by-the-author)).

Declared prices come with a known trap: the tax base is max(declared price, assessed reference value),
so under-declared cash deals bunch exactly at the assessed value. The cleaning pipeline detects and
removes that bunching for cash purchases (financed deals are bank-appraised); the whole filter chain
with per-step counts is in [`data/filter_report.md`](data/filter_report.md) and
[`data/underdeclaration_report.md`](data/underdeclaration_report.md). All data is public
([`DATA_NOTICE.md`](DATA_NOTICE.md)).

<details>
<summary>Glossary of Brazilian terms used in the data</summary>

| Term | Meaning |
|---|---|
| ITBI (*Imposto sobre Transmissão de Bens Imóveis*) | Municipal real-estate transfer tax; its paid forms ("guias") record declared transaction prices |
| IPTU (*Imposto Predial e Territorial Urbano*) | Annual property tax; source of the physical attributes (areas, use, standard, completion year) |
| SQL (*Setor, Quadra, Lote*) | 11-digit cadastral key: sector (3) + block (3) + lot (4) + check digit (1) |
| VVR (*Valor Venal de Referência*) | Municipal reference assessed value; the ITBI tax base is max(declared price, VVR) |
| ACC (*Ano de Construção Corrigido*) | Corrected construction-completion year from the IPTU register |
| Quadra fiscal | Fiscal block, the city-block polygon of the cadastre (GeoSampa layer) |
| Guia | The individual tax form; one paid form = one transaction record |
| SFH / MCMV | Housing-finance system / federal affordable-housing programme (financing categories) |
| NBR 14653-2 | Brazilian valuation standard for urban properties |
| GeoSampa | Open geoportal of the São Paulo City Hall |

</details>

## Earlier work by the author

This entry is the third step of a line of work on tabular foundation models in property valuation.

1. **Transformers na avaliação de imóveis residenciais: comparação entre regressão linear múltipla,
   FT-Transformer e TabPFN v2 com dados de ITBI de São Paulo** (*Transformers in residential property
   valuation: a comparison of multiple linear regression, FT-Transformer and TabPFN v2 on São Paulo
   ITBI data*). I Fórum Nacional IGEL & SOBREA, August 2026
   ([igelsobrea.com.br](https://igelsobrea.com.br)). On transaction prices from the same São Paulo
   transfer tax, TabPFN v2 outperformed both the FT-Transformer and the multiple linear regression
   (OLS) model.
2. **Non-linear methods versus spatial regression in apartment valuation: a comparative study in
   Belo Horizonte** (*Métodos não lineares versus regressão espacial na avaliação de apartamentos: um
   estudo comparativo em Belo Horizonte*). III CEAD, Conferência de Engenharia de Avaliações e
   Diagnóstica, September 2026 ([PDF](paper/M%C3%89TODOS_N%C3%83O_LINEARES_VERSUS_REGRESS%C3%83O_ESPACIAL_NA_AVALIA%C3%87%C3%83O_DE_APARTAMENTOS_UM_ESTUDO_COMPARATIVO_EM_BELO_HORIZONTE_final_.pdf)). On 5,005 apartment listings, TabPFN v3 was
   compared with linear and non-linear spatial econometric models (among them OLS, SAR, SEM and
   PS-SAR) and outperformed every model in the comparison. The paper received the conference's award
   for the best work in appraisal engineering.

This repository extends the Belo Horizonte study to a larger, noisier and more complex base: 130,000
declared transaction prices instead of 5,005 listings, three property types instead of apartments
only, tax under-declaration to detect and remove, a one-year-ahead test on top of the spatial block
CV, and TabPFN-3.5 against spatial specialists that include a geographically weighted XGBoost.

## The protocol

Two legs, same metrics, same tests, everything in `src/protocol.py`.

**Leg 1, spatial block CV.** A stratified 24,000-row sample of 2025 ("Level A", so the slow
baselines can run) is split into 10 K-means blocks on the coordinates; each block is held out in turn.
This asks: can the model price a part of the city it has never seen? It is the harder test for a model
that leans on location.

**Leg 2, out-of-time.** Fit once on 2025 (Level A, and then the full 82k base), predict every 2026
transaction. Same city, one to seven months later: the spatial-lag models get neighbours from the
same streets, so this is the most favourable setting for explicit spatial modelling.

Metrics on ln of the unit price (R$/m²): RMSE, MAPE, R². Residual Moran's I on a k-NN-8 matrix.
Paired tests over the ten blocks (Wilcoxon) and a block bootstrap for the pooled difference. Three
seeds where a model is stochastic. Leakage checks: zero identical transactions between 2026 and any
2025 base; 0.6 % of 2026 rows share a cadastral key with Level A.

The five models:

| Model | Sees space through |
|---|---|
| OLS hedonic | nothing: hedonic attributes only, among them the distance to the nearest station |
| SAR lag (`spreg` GM_Lag) | k-NN-8 weights matrix and ρ |
| XGBoost + spatial features | fold-internal k-NN-8 lag (leave-one-out for training rows), UTM coordinates + 30°/45°/60° rotations, nested Optuna tuning |
| Geographically weighted XGBoost | local XGBoost models, each fitted on the k nearest rows with bi-square distance weights, blended with a global model; k and the blend chosen by nested block CV |
| **TabPFN-3.5, plain table** | latitude and longitude as two numeric columns; zero-shot or thinking mode |

Geographically weighted XGBoost (G-XGBoost, Grekousis 2025) is one of the methods of the reference
study. As packaged (`geoxgboost`) it calibrates one local model, with its own grid search, at every
training row over a dense distance matrix: the pilot (`src/pilot_gxgb_timing.py`) measured about
0.5 h per bandwidth candidate per fold at 20k rows and 2.9 GB of matrix. `src/model_gxgb.py` keeps
the estimator — local XGBoost models with adaptive bi-square weights, blended with a global XGBoost
by a weight α — and calibrates the local models at 1,000 regression points (K-means centres of the
training coordinates), as geographically weighted regression allows; a location is predicted by
its three nearest local models. Like the original, the local models see the hedonic attributes and
no coordinates: space enters only through the weighting. The bandwidth k (50 to 3,200 neighbours
for 24,000 training rows, scaled to the training size; the floor was lowered from 400 to 50 after the
first searches kept choosing the smallest value) and α are chosen by an inner block CV inside each
outer fold. On Level A the inner search still chose the smallest value in 8 of 10 folds; the 2026
error barely moves between 50 and 400 neighbours (see Leg 2). One fit takes about a minute;
tuning, ten.

## Results

### Leg 1 — leave-one-block-out CV, Level A (24,000 rows of 2025)

![The ten K-means blocks, and the per-block 2026 RMSE difference TabPFN − XGBoost+lag](results/figures/fig1_blocks_map.png)

*Figure 1. (a) The ten blocks, each held out in turn; (b) the 2026 out-of-time RMSE difference
TabPFN − XGBoost+lag per block, negative everywhere.*

| Model | RMSE (ln) | MAPE | R² (ln) | Moran's I of residuals |
|---|---|---|---|---|
| Median by property type (floor) | 0.497 | 42.1 % | −0.04 | 0.446 |
| OLS hedonic | 0.470 | 39.4 % | 0.066 | 0.390 |
| SAR lag, GM_Lag (ρ = 0.91) | 0.448 | 38.4 % | 0.155 | 0.363 |
| XGBoost + rotated coordinates, no lag (ablation) | 0.397 | 33.3 % | 0.334 | 0.355 |
| **XGBoost + rotated coordinates + k-NN-8 lag** | **0.385** (seeds 0.385–0.387) | 32.3 % | **0.376** | **0.309** |
| Geographically weighted XGBoost (k ≈ 45, α = 0.5) | 0.410 (seeds 0.407–0.410) | 34.5 % | 0.291 | 0.324 |
| **TabPFN-3.5, plain table, zero-shot** | 0.394 (seeds 0.391–0.394) | 32.2 % | 0.346 | 0.348 |
| TabPFN-3.5, plain table, thinking (medium) | 0.389 | **31.7 %** | 0.362 | 0.342 |
| TabPFN-3.5, plain table, thinking (high) | 0.388 | **31.7 %** | 0.366 | 0.339 |

What the paired tests say (`results/compare_*.json`): TabPFN zero-shot beats SAR in 8 of 10 blocks
(ΔRMSE −0.054, bootstrap 95 % CI [−0.100, −0.014], Wilcoxon p = 0.010). Against the fully spatial
XGBoost it is a statistical tie: ΔRMSE +0.009 [−0.018, +0.034], 7 blocks won, one clear loss in the
largest block. Against the geographically weighted XGBoost it is also a tie, leaning its way:
ΔRMSE −0.013 to −0.016 across that model's three seeds, intervals across zero, 6 or 7 blocks won.
Thinking mode helps a little (−0.005 [−0.008, −0.001] vs zero-shot seed 42, about −0.002 vs
seeds 43 and 44) and gives the lowest MAPE in the table; high effort costs 13× more per fold for nothing beyond noise.

On the residual criterion the bet takes a hit. TabPFN's Moran's I (0.348) sits below SAR (0.363) but
above XGBoost with the explicit lag (0.309) and above the geographically weighted XGBoost
(0.319–0.326), whose local models are fitted on the nearest 40 to 176 rows. Inside XGBoost, the lag alone is worth ΔRMSE −0.013 and
cuts Moran's I from 0.355 to 0.309: that is the price of *not* modelling space explicitly when the
test is a part of the city nobody has seen. Partial refutation, on the record.

Errors are large in absolute terms (R² 0.35–0.38 against 0.54 in the listings study); declared
prices, three property types and centroid coordinates all add noise, and every model leaves plenty
of spatial structure behind.

### Leg 2 — fit on 2025, predict the 47,810 transactions of 2026

| Trained on Level A (24,000 rows) | RMSE (ln) | MAPE | R² (ln) | Bias (ln) | Moran's I |
|---|---|---|---|---|---|
| OLS hedonic | 0.404 | 32.4 % | 0.185 | +0.033 | 0.408 |
| SAR lag | 0.346 | 27.0 % | 0.401 | +0.032 | 0.224 |
| XGBoost + rotated coordinates + k-NN-8 lag | 0.288 | 21.8 % | 0.584 | +0.047 | 0.140 |
| Geographically weighted XGBoost (k = 50, α = 0.5) | 0.314 | 24.0 % | 0.508 | +0.045 to +0.048 | 0.187 |
| **TabPFN-3.5, plain table, zero-shot** (fit in 9 s) | **0.275** | **20.4 %** | **0.622** | +0.055 | **0.122** |
| TabPFN-3.5, plain table, thinking (medium) | 0.275 | 20.5 % | 0.621 | +0.055 | 0.121 |

| Trained on the full 2025 base (82,187 rows) | RMSE (ln) | MAPE | R² (ln) | Bias (ln) | Moran's I |
|---|---|---|---|---|---|
| OLS hedonic | 0.402 | 32.4 % | 0.194 | +0.023 | 0.397 |
| SAR lag | 0.326 | 25.0 % | 0.470 | +0.031 | 0.134 |
| XGBoost + rotated coordinates + k-NN-8 lag | 0.287 (seeds 0.283–0.294) | 21.1 % | 0.59 | +0.07 to +0.10 | 0.104 |
| Geographically weighted XGBoost (k = 342, α = 0.5) | 0.299 (seeds 0.298–0.301) | 22.1 % | 0.55 | +0.06 to +0.08 | 0.164 |
| **TabPFN-3.5, plain table, zero-shot** (fit in 15 s) | **0.265** | **19.5 %** | **0.649** | +0.058 | **0.092** |

![Predicted vs observed ln unit price on the 2026 transactions](results/figures/fig3_pred_vs_obs_2026.png)

*Figure 2. Predicted vs observed ln(R$/m²) on 2026, both models fitted on the full 2025 base. The
cloud sits above the identity line: every model under-predicts a rising market.*

Paired over the ten blocks (`results/oot_paired_comparisons_*.csv`): on Level A, TabPFN vs
XGBoost+lag ΔRMSE −0.013 [−0.016, −0.011], ΔMAPE −1.4 pp, 10 of 10 blocks, Wilcoxon p = 0.002, and
less residual autocorrelation (0.122 vs 0.140). On the full base the gap widens to −0.029
[−0.032, −0.026], ΔMAPE −2.0 pp, again 10 of 10, with the lowest Moran's I of any model. Going from 24k
to 82k rows improves TabPFN by −0.010 and SAR by −0.020; XGBoost does not improve (+0.006) because its
trend bias doubles and eats the variance gain. Thinking mode adds nothing in this leg.

The geographically weighted XGBoost lands between SAR and XGBoost+lag at both scales. TabPFN beats it
by −0.038 to −0.039 across its three seeds in 10 of 10 blocks on Level A (every interval within
[−0.049, −0.028]) and by −0.033 to −0.036 on the full base, where it wins 9 of 10 blocks (p = 0.004): the one loss is the smallest block,
2,425 transactions, by 0.001–0.002. Its residuals keep more spatial autocorrelation (0.187 and 0.164)
than XGBoost+lag's, as if the local models, which see no coordinates, averaged away the fine
structure that the explicit lag and the plain coordinates keep. The bandwidth hardly matters here:
with 400, 200, 100 or 50 neighbours, as the search grid was widened, its 2026 RMSE on Level A stays
between 0.308 and 0.314 (`results/gxgb_bandwidth_sensitivity.json`).

Errors are flat across blocks (0.25–0.32) and across horizons of one to seven months (0.26–0.29): no
subset is carrying the result.

![RMSE per block, both legs](results/figures/fig2_per_block_rmse.png)

*Figure 3. RMSE per block. Left: block CV on Level A (tie with XGBoost+lag). Right: 2026, full base
(10 of 10).*

![Residual Moran's I, both legs](results/figures/fig5_moran.png)

*Figure 4. Residual Moran's I. Left: the partial refutation in the block CV. Right: one year ahead,
inside the city, the plain-table model leaves the least spatial structure of the models shown.*

![Mean residual by month ahead](results/figures/fig4_months_ahead_bias.png)

*Figure 5. Mean residual by month after the training window, full 2025 base.*

**The limitation worth stating.** Every model under-predicts 2026 (prices rose), and the
non-linear learners more so (bias +0.04 to +0.10 in ln vs +0.02 to +0.03 for the linear models):
trees and TabPFN hold the last observed level of the month index instead of extrapolating a trend.
Appraisal practice would apply an index. It is reported here, not corrected.

## Further tests

Six checks on the main result, each with its conclusion. The full write-up, with every table, the
mechanism checks and the expectations written before the runs, is in
[`docs/further_tests.md`](docs/further_tests.md).

### The postal code in place of the coordinates

Latitude, longitude and the distance to the nearest station, computed from them, are removed; the
raw postal code goes in as a string. The CEP (*Código de Endereçamento Postal*) is Brazil's ZIP code:
eight digits, in São Paulo usually one street or a stretch of one, 15,938 distinct values in the 2025
base, and nothing in the column says which code lies next to which. The baselines keep every spatial
input. Zero-shot, seed 42 (`scripts/50_nominal_location_ablation.sh replace`,
`results/postal_code_vs_baselines.md`).

| RMSE (ln) · Moran's I | TabPFN-3.5, postal code alone | TabPFN-3.5, latitude and longitude | XGBoost + lag | Geographically weighted XGBoost | SAR lag |
|---|---|---|---|---|---|
| One year ahead, fit on the full base | 0.268 · 0.091 | 0.265 · 0.092 | 0.283–0.294 · 0.103–0.107 | 0.298–0.301 · 0.163–0.166 | 0.326 · 0.134 |
| District never seen | 0.457 · 0.502 | 0.391–0.394 · 0.342–0.348 | 0.385–0.387 · 0.309–0.318 | 0.407–0.410 · 0.319–0.326 | 0.448 · 0.363 |

*Ranges are over three seeds.*

Expected before the runs (`src/model_tabpfn.py`): the postal code would recover less than half of
the way from no location to the coordinates and lose to XGBoost + lag in both tests. Found: one year
ahead it recovered 94–97 % and won every block against XGBoost + lag and SAR, and 9 or 10 of 10
against the geographically weighted XGBoost. In the unseen district it fell below its own
no-location floor (0.457 against 0.440). Codes absent from the training base (22 % of the 2026
transactions, against the 24k fit) are priced almost as well as seen ones (97 % against 98 % of the
signal), so the model is not looking prices up by code. Why it works is the one question left open
([What is not here](#what-is-not-here-and-why)).

**Conclusion:** inside the market it has seen, a raw postal code does the work of the coordinates;
on new ground, only coordinates extrapolate.

### Financed deals only

Financed transactions carry a bank appraisal and are the cleanest prices in the base. Both tests were
re-run on them alone: 7,211 of the 24,000 Level A rows for the block CV, fitted once on the same rows
and scored on the 20,052 financed transactions of 2026 (`scripts/40_financed_robustness.sh`).

| Model | CV RMSE (ln) | CV MAPE | CV Moran's I | 2026 RMSE (ln) | 2026 MAPE | 2026 Moran's I |
|---|---|---|---|---|---|---|
| OLS hedonic | 0.354 | 28.5 % | 0.343 | 0.343 | 26.5 % | 0.491 |
| SAR lag | 0.344 | 28.1 % | 0.342 | 0.286 | 22.0 % | 0.297 |
| XGBoost + rotated coordinates + k-NN-8 lag | 0.286 | 23.0 % | 0.327 | 0.221 | 16.7 % | 0.160 |
| **TabPFN-3.5, plain table, zero-shot** | **0.269** | **21.1 %** | **0.279** | **0.207** | **15.3 %** | **0.143** |

**Conclusion:** errors fall for every model, and in the block CV TabPFN-3.5 moves ahead of XGBoost +
lag: ΔRMSE −0.017 [−0.032, −0.003] against its seed 42 (8 of 10 blocks, p = 0.049; 9 of 10 against
seeds 43 and 44), with less residual autocorrelation (0.279 against 0.327–0.340). One year ahead the
ranking is unchanged. Against XGBoost + lag, the partial refutation does not appear on these prices;
the geographically weighted XGBoost was not re-run on them.

### The unit and building text

The unit complement (`AP 201 E 3VGS`: apartment 201 with three parking spaces) and the *Referência*
field (`EDIFICIO THE PARK`) added as raw strings next to the coordinates; nothing parsed
(`scripts/70_text_and_grouped_thinking.sh`).

| | plain table | + unit and building text | ΔRMSE (ln) [95 % CI] · blocks won |
|---|---|---|---|
| Leg 1, block CV on Level A | 0.3938 · MAPE 32.2 % · Moran's I 0.348 | 0.3835 · 30.8 % · 0.331 | −0.0103 [−0.0161, −0.0031] · 8/10 |
| Leg 2, 24k → 2026 | 0.2750 · 20.4 % · 0.122 | 0.2750 · 20.6 % · 0.122 | +0.0000 [−0.0011, +0.0010] · 6/10 |
| Leg 2, 82k → 2026 | 0.2649 · 19.5 % · 0.092 | 0.2610 · 19.2 % · 0.086 | −0.0039 [−0.0049, −0.0031] · 10/10 |

Expected before the runs: a small gain at most, one year ahead, where the same buildings sell again.
Found: the largest gain is in the district never seen, against each of the three seeds of the plain
table (−0.007 to −0.010); there TabPFN-3.5 ties XGBoost + lag and beats the geographically weighted
XGBoost in 9 of 10 blocks. The gain sits in forms that mention parking (−0.034), commercial units
(−0.023) or a named building (−0.017), and complements the model has never seen gain as much as those
it has (−0.014 against −0.013): the gain does not depend on having seen the exact string.

**Conclusion:** in an unseen district the text lowers the error and turns the tie with the
geographically weighted XGBoost into a win; against XGBoost + lag it stays a tie, and the residual
gap remains (Moran's I 0.331 against 0.309–0.326 for the two XGBoost models).

### Thinking mode

Thinking (medium) lowers the block-CV error by 0.005 against zero-shot seed 42 (−0.005 [−0.008,
−0.001]) and by 0.002 against seeds 43 and 44, and adds nothing one year ahead; high effort costs 13
times more per fold for no further gain. With the spatial blocks as `group_col`, its internal
validation holds out whole districts, as the block CV does. In that test the change against thinking
without groups is +0.0038 [−0.0055, +0.0149], 3 of 10 blocks, with Moran's I 0.350 against 0.342. One
year ahead it gains 0.0014, but there the group label also tells the model which part of the city a
row belongs to.

**Conclusion:** extra compute buys a small gain in an unseen district; splitting the internal
validation by districts adds nothing to it, and neither touches the residual gap.

### The address on top of the coordinates

The free-text district field (`bairro`) and the postal code added to latitude and longitude, as
filed (`scripts/50_nominal_location_ablation.sh`, `results/nominal_ablation_summary.md`). In every
test and combination the RMSE moves by less than 0.006 in ln and the MAPE by at most 0.3 points; one
year ahead, Moran's I moves by at most 0.008 and the margin over XGBoost + lag stays where it was.

**Conclusion:** the coordinates already carry what the address carries.

### SHAP: what the model looks at

Permutation SHAP on 50 stratified 2026 transactions, with the zero-shot model reloaded from its cache
(no refit) and every API call cached (`results/shap/`). Longitude (mean |SHAP| 0.184 in ln) and
latitude (0.143) are the two strongest inputs, ahead of property type (0.127), built area (0.112) and
age (0.082). The directions are those an appraiser expects: older buildings and longer walks to a
station lower the unit value, a higher finish grade raises it, larger built areas lower the price per
square metre.

![Mean |SHAP| per feature, by property type](results/figures/fig6_shap_importance.png)

![One valuation explained](results/figures/fig8_shap_waterfall.png)

*Figures 6 and 7. Mean |SHAP| per feature, by property type; one valuation from base value to
estimate, as an appraiser would read it.*

**Conclusion:** inside the model, the spatial signal comes from the two raw coordinates, its two
strongest inputs.

## The app

Live at [tabpfn-itbi-sp.streamlit.app](https://tabpfn-itbi-sp.streamlit.app), or `streamlit run app/app.py` locally; a narrated
five-minute walkthrough is on [YouTube](https://youtu.be/S1wQqxKR-hI). Four tabs on the versioned results: the claim and
scoreboard, a map of where each model fails on the 2026 transactions, an inspector for any single
transaction (every model's estimate, the k-NN-8 neighbourhood the baselines saw, SHAP for the 50 explained properties), and,
with a `TABPFN_TOKEN`, a live TabPFN-3.5 estimate for a property you describe. Install, deploy and
troubleshoot in [`app/MANUAL.md`](app/MANUAL.md).

## Reproducing

```bash
pip install -r requirements.txt

python pipeline/01_fiscal_block_centroids.py        # block centroids from the GeoSampa GPKG
python pipeline/02_itbi_cleaning.py                 # raw workbooks -> train / test / level_a + reports
python pipeline/03_underdeclaration_diagnostics.py
python pipeline/04_english_labels.py                # English-header copies under data/english/
python pipeline/05_reference_field.py               # the free-text Referência field (text variant only)

python -m src.protocol --data data/itbi_sp_2025_level_a.csv --smoke   # protocol on a trivial baseline, no API

python -m src.model_sar --estimator ols             # OLS
python -m src.model_sar                             # SAR lag
python -m src.model_xgb --seeds 42 43 44            # XGBoost + spatial features (~70 min on 2 vCPU)
python -m src.model_xgb --no-lag --seeds 42 43 44   # ablation
python -m src.model_gxgb --seeds 42 43 44           # geographically weighted XGBoost (~2.5 h on 2 vCPU)

export TABPFN_TOKEN=...                             # TabPFN-3.5 through the API (separate venv: requirements-tabpfn.txt)
bash scripts/10_tabpfn_level_a.sh t0                # zero-shot + paired tests
bash scripts/10_tabpfn_level_a.sh t0-think          # thinking mode

python -m src.out_of_time --model ols --train level_a   # leg 2, baselines (also: sar, xgb, gxgb; --train full)
bash scripts/20_tabpfn_out_of_time.sh t0 && bash scripts/20_tabpfn_out_of_time.sh compare

python -m src.make_figures                          # figures from the versioned results, no model re-run
bash scripts/30_tabpfn_shap.sh run && python -m src.make_figures --shap
bash scripts/40_financed_robustness.sh baselines && bash scripts/40_financed_robustness.sh tabpfn
bash scripts/50_nominal_location_ablation.sh all    # ablation: district and postal code as string columns
bash scripts/50_nominal_location_ablation.sh replace   # a name in place of the coordinates
python scripts/51_postal_code_vs_baselines.py          # postal code only vs every seed of every baseline (no API)
bash scripts/70_text_and_grouped_thinking.sh all       # text variant + thinking with blocks as groups
python scripts/72_variants_summary.py                  # every model, both legs, paired tests (no API)
python scripts/60_app_live_check.py                 # presses Estimate in tab 4 of the app, headless
```

Every TabPFN-3.5 response is cached under `results/tabpfn_cache/<label>/`, so the protocol re-runs
offline from the cached predictions. Seeds are fixed (42, plus 43/44 where a model is stochastic);
nothing is downloaded or geocoded at runtime. The TabPFN runs need their own environment
(`requirements-tabpfn.txt`: `tabpfn-client` pins pandas ≤ 2.3.3).

Clean-clone check, 2026-09-28: from a fresh `git clone` on Linux / Python 3.11, the pipeline rebuilt
the three bases byte-identical to the versioned files (82,187 / 47,810 / 24,000 rows), the smoke test
returned the documented numbers (RMSE 0.4971, Moran's I 0.446) and the OLS baseline regenerated
`results/cv_ols_2025_level_a.json` byte-identically.

## Data files

| File | Content |
|---|---|
| `data/raw/GUIAS DE ITBI PAGAS 2025 XLS.xlsx` | 230,525 ITBI forms paid in 2025 |
| `data/raw/GUIAS DE ITBI PAGAS 2026 XLS.xlsx` | 135,655 forms paid Jan–Jul 2026 |
| `data/raw/quadra_fiscal_p*.gpkg` | 64,223 fiscal-block polygons (GeoSampa WFS; not versioned, see DATA_NOTICE) |
| `data/raw/estacao_metro.geojson`, `estacao_trem.geojson` | 203 subway and rail stations |
| `data/fiscal_block_centroids.csv`, `data/fiscal_block_lookup.csv` | derived centroids (pipeline 01) |
| `data/itbi_sp_2025_train.csv.gz` | cleaned 2025 base, 82,187 rows |
| `data/itbi_sp_2026_test.csv` | cleaned 2026 base, 47,810 rows |
| `data/itbi_sp_2025_level_a.csv` | stratified 24,000-row sample |
| `data/english/*_en.csv`, `*_en.csv.gz` | English-header copies (not the originals) + `column_mapping.csv` |
| `data/shap_subsample.csv`, `data/shap_background.csv` | the 50 explained properties and the 20-row background |
| `data/itbi_sp_reference_field.csv.gz` | the free-text *Referência* field for every cleaned row (pipeline 05; text variant only) |

Column semantics: [`data/data_dictionary.md`](data/data_dictionary.md). Raw files keep their
original Portuguese names for provenance.

## Layout

```
data/       raw public files, derived datasets, data reports
pipeline/   data preparation (01–05)
src/        protocol, SAR/OLS, XGBoost, geographically weighted XGBoost, TabPFN-3.5, out-of-time, SHAP, figures
scripts/    TabPFN-3.5 API runs (outputs cached in results/)
results/    metrics, out-of-fold and hold-out predictions, paired tests, API cache, SHAP, figures
app/        Streamlit app + manual
docs/       full write-up of the further tests
paper/      reference study (Belo Horizonte, CEAD 2026); manuscript in preparation
notebooks/  side experiments
```

## What is not here, and why

* **Per-transaction comparables from inside the model.** TabPFN's classification head is an
  attention-weighted vote over training rows and can be read out exactly; its regression head is a
  linear layer over distribution buckets and cannot. A companion-classifier route was designed and
  left out of scope for the hackathon (`notebooks/`).
* **Prediction intervals and NBR 14653-2 precision grades.** Predictive quantiles are cached for the
  zero-shot fits; the coverage analysis is future work.
* **Trend correction.** See the limitation above.
* **Why postal codes never seen in training still work.** Replacing each code with a random one under
  a fixed bijection keeps the identities and destroys the geographic order; if the unseen codes then
  fall to the floor while the seen ones hold, the model is reading the order of the codes. Designed,
  not run.

## The paper

A scientific paper with the full methods, results and discussion accompanies this repository. It
will be published on an open platform (arXiv or a repository with a DOI) and linked here.

## License and citation

Code and documentation under [Apache 2.0](LICENSE). Data notices in
[`DATA_NOTICE.md`](DATA_NOTICE.md). To cite, see [`CITATION.cff`](CITATION.cff).

## Author

**Agnaldo Calvi Benvenho**, mechanical engineer (POLI-USP), MSc in Appraisal Engineering
(Universidad Politécnica de Valencia), Certification Director at IBAPE Nacional (Brazilian Institute
of Engineering Appraisals and Expert Examinations), court-appointed expert at the São Paulo State
Court.

**AI-assistance statement.** The code of this project was developed with the assistance of cloud
AI tools — Claude (Fable and Opus models) and ChatGPT (Astra) — and of the locally run models
Qwen 3.8 Next Flash and Qwen 3 Coder Next. The cloud tools also assisted with drafting,
translation, literature search, figure generation and text revision. The experimental design, the
methodological decisions, the analysis of the results and the conclusions are the author's, who
reviewed and takes responsibility for the entire content.
