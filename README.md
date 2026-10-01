# Location, location, location

*TabPFN-3.5 with the plain table vs models that see space explicitly, on 82,187 real transaction
prices from São Paulo, tested one year ahead on 47,810 more.*

Prior Labs TabPFN-3.5 Hackathon entry · **Live app: [tabpfn-itbi-sp.streamlit.app](https://tabpfn-itbi-sp.streamlit.app)** · [app manual](app/MANUAL.md) · Apache-2.0

---

Ask any appraiser what sets the value of a property and you get the old line: three things,
*location, location, location*. Spatial econometrics took it seriously and built machinery around
it: a weights matrix over the neighbours, a spatial lag, a ρ to estimate. Gradient boosting needs the
same help, hand-fed as neighbourhood features. It works. It is also a lot of engineering that has to be
redone for every city and every dataset.

This repository asks a blunt question. What if you give a tabular foundation model latitude and
longitude as two ordinary numeric columns, nothing else about space, and let it figure location out on
its own? Does it match the specialists that were built for exactly this?

**Short answer.** On declared sale prices from São Paulo, fitted once in about ten seconds:

* Predicting one year ahead (fit on 2025, predict every cleaned 2026 transaction), TabPFN-3.5 with
  nine raw columns beats XGBoost with a hand-built spatial lag in 10 of 10 spatial blocks
  (RMSE 0.265 vs 0.287 in ln, MAPE 19.5 % vs 21.0 %), beats a geographically weighted XGBoost in
  9 of 10 (0.265 vs 0.298–0.301) and the SAR lag model by a wide margin, and leaves the *least*
  spatial autocorrelation in its residuals of the five models (Moran's I 0.092).
* Predicting a part of the city it has never seen (leave-one-block-out CV), it ties both XGBoost
  models on error and beats SAR, but leaves *more* spatial structure in its residuals than either
  XGBoost. That is a partial refutation, and it is reported as one. Reading the unit and building
  text typed on the tax form closes the error gap (a tie with XGBoost+lag, a win over the
  geographically weighted XGBoost) but not the residual one.
* **Take the coordinates away and give it only the raw postal code** — the CEP, Brazil's ZIP code,
  as a text column with 15,938 distinct values and no geometry — and one year ahead it still beats
  all three spatial specialists, against every seed (10 of 10 blocks against XGBoost+lag and SAR,
  9 or 10 of 10 against the geographically weighted XGBoost), and still leaves less spatial
  autocorrelation in its residuals than any of them (full base: RMSE 0.268 vs 0.283–0.294, 0.298–0.301
  and 0.326; Moran's I 0.091 vs 0.103–0.107, 0.163–0.166 and 0.134). A categorical label is enough for the model to do
  the spatial analysis itself. For a part of the city it has never seen it is not: there only
  coordinates extrapolate. [Details](#spatial-analysis-from-a-zip-code).

Everything below is how those claims were earned.

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
An earlier study by the author on 5,005 Belo Horizonte apartment *listings* (awarded at CEAD 2026,
`paper/`) found TabPFN v3 ahead of OLS, SAR, SEM and PS-SAR; this is the harder, larger, real-price
version of that question.

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
| OLS hedonic | nothing (coordinates as covariates only) |
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
no coordinates: space enters only through the weighting. The bandwidth k (200 to 3,200 neighbours,
scaled to the training size) and α are chosen by an inner block CV inside each outer fold. One fit
takes about a minute; tuning, ten.

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
Thinking mode helps a little and consistently (−0.005 [−0.008, −0.001] vs zero-shot)
and gives the lowest MAPE in the table; high effort costs 13× more per fold for nothing beyond noise.

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
| SAR lag | 0.346 | 27.0 % | 0.401 | +0.033 | 0.224 |
| XGBoost + rotated coordinates + k-NN-8 lag | 0.288 | 21.8 % | 0.584 | +0.047 | 0.140 |
| Geographically weighted XGBoost (k = 50, α = 0.5) | 0.314 | 24.0 % | 0.508 | +0.045 to +0.048 | 0.187 |
| **TabPFN-3.5, plain table, zero-shot** (fit in 9 s) | **0.275** | **20.4 %** | **0.622** | +0.055 | **0.122** |
| TabPFN-3.5, plain table, thinking (medium) | 0.275 | 20.5 % | 0.621 | +0.055 | 0.121 |

| Trained on the full 2025 base (82,187 rows) | RMSE (ln) | MAPE | R² (ln) | Bias (ln) | Moran's I |
|---|---|---|---|---|---|
| OLS hedonic | 0.402 | 32.4 % | 0.194 | +0.023 | 0.397 |
| SAR lag | 0.326 | 25.0 % | 0.470 | +0.031 | 0.134 |
| XGBoost + rotated coordinates + k-NN-8 lag | 0.287 (seeds 0.283–0.294) | 21.0 % | 0.59 | +0.07 to +0.10 | 0.105 |
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

## Explaining an estimate — SHAP on 50 properties

An appraiser who signs a valuation has to say why. The model is served through an API, so the
explanation is model-agnostic permutation SHAP (`shap.PermutationExplainer`, independent masker,
20-row background), run on the zero-shot model reloaded from its cached record — no refit — for 50
stratified 2026 transactions it had never seen. 27,217 predicted rows in total, every call cached under
`results/shap/cache/`, so the attributions are reproducible offline. Values are in ln; `exp(SHAP)` is
the multiplier on R$/m².

![Mean |SHAP| per feature, by property type](results/figures/fig6_shap_importance.png)

![SHAP beeswarm](results/figures/fig7_shap_beeswarm.png)

![One valuation explained](results/figures/fig8_shap_waterfall.png)

*Figures 6–8. Mean |SHAP| per feature; direction of the effects; one valuation from base value to
estimate, as an appraiser would read it.*

Longitude (mean |SHAP| 0.184, a typical ×1.20 on the unit value) and latitude (0.143, ×1.15) are the
two strongest inputs, ahead of property type (0.127), built area (0.112) and age (0.082). That is the
bet seen from inside the model: the two raw coordinates carry the spatial signal the baselines have
to encode with a weights matrix. What they carry is location rather than geometry: with a
[raw postal code in their place](#spatial-analysis-from-a-zip-code) the model loses almost nothing
inside the market it has seen. Directions are the ones an appraiser expects: older buildings and
longer walks to a station pull the unit value down, higher finish grades push it up, larger built
areas lower the price per square metre.

## Robustness — financed deals only

Financed transactions carry a bank appraisal and are the cleanest prices in the base. Both legs were
re-run on that subset alone: 7,211 of the 24,000 Level A rows for the block CV, the same rows fitted
once and scored on the 20,052 financed transactions of 2026 (`scripts/40_financed_robustness.sh`,
`results/*_fin*`).

| Model | CV RMSE (ln) | CV MAPE | CV Moran's I | 2026 RMSE (ln) | 2026 MAPE | 2026 Moran's I |
|---|---|---|---|---|---|---|
| OLS hedonic | 0.354 | 28.5 % | 0.343 | 0.343 | 26.5 % | 0.491 |
| SAR lag | 0.344 | 28.1 % | 0.342 | 0.286 | 22.0 % | 0.297 |
| XGBoost + rotated coordinates + k-NN-8 lag | 0.286 | 23.0 % | 0.327 | 0.221 | 16.7 % | 0.160 |
| **TabPFN-3.5, plain table, zero-shot** | **0.269** | **21.1 %** | **0.279** | **0.207** | **15.3 %** | **0.143** |

Cleaner prices, lower errors for everyone, same ranking, and the plain-table model does better here
than in the main analysis on every criterion: block CV vs XGBoost+lag ΔRMSE −0.017 [−0.032, −0.003],
p = 0.049, 8 of 10 blocks (a tie on the full Level A); 2026 vs XGBoost −0.013 [−0.017, −0.010], 10 of
10. The partial refutation from the block CV does not reappear: on financed deals TabPFN leaves less
spatial structure than XGBoost+lag in both legs. Nothing in the headline results depends on the cash
deals; if anything, the noisier cash prices are where the explicitly spatial learner holds its own.

## Ablation — the address as well as the coordinates

The ITBI form states location twice: as the coordinates the main analysis uses, and as names — a
free-text district field (`bairro`) and the postal code (`cep`). TabPFN-3.5 takes high-cardinality
string columns as they come, and none of the runs above uses that. This ablation adds the two names
to the plain table and changes nothing else: same rows, same folds, same seed, zero-shot. It stays
outside the main analysis because it changes the information set, and the baselines are not re-run
with these columns (`scripts/50_nominal_location_ablation.sh`, `results/nominal_ablation_summary.md`).

The columns go in as filed. `cep` has 10,022 distinct values in the 24,000 rows of Level A. `bairro`
has 3,898, is empty in 39 % of the rows, and about a quarter of what is filled in is a tower or block
label typed into the wrong field. Nothing is cleaned.

Two expectations were written into `src/model_tabpfn.py` before the runs. Leg 1 should show nothing:
97–100 % of a held-out block's postal codes are absent from the training folds by construction, so
it is a negative control. Leg 2 is where names could help, and if they do, the gain should sit in
the 2026 rows whose code occurs in the training base.

ΔRMSE (ln) against the plain table, 95 % CI by block bootstrap; negative is better:

| | plain table | + bairro | + cep | + bairro + cep |
|---|---|---|---|---|
| Leg 1, block CV on Level A | 0.3938 | −0.0004 [−0.0037, +0.0036] | +0.0051 [−0.0106, +0.0259] | −0.0027 [−0.0110, +0.0101] |
| Leg 2, fit on Level A (24k) → 2026 | 0.2750 | +0.0019 [+0.0014, +0.0025] | +0.0012 [+0.0003, +0.0021] | +0.0017 [+0.0004, +0.0027] |
| Leg 2, fit on the full base (82k) → 2026 | 0.2649 | −0.0001 [−0.0006, +0.0003] | −0.0012 [−0.0018, −0.0006] | −0.0020 [−0.0031, −0.0009] |

The names add nothing that matters. Leg 1 is flat, as it had to be. In leg 2 they cost about 0.002
on 24,000 rows and return about 0.002 on 82,000, under 1 % of the RMSE in either direction. MAPE
moves by at most 0.3 points (20.4 % → 20.5–20.7 %; 19.5 % → 19.4–19.5 %), Moran's I of the residuals
by at most 0.008 (0.122 → 0.122–0.130; 0.092 → 0.088–0.093), and the margin over XGBoost+lag is
where it was (−0.012 on 24k, −0.030 on 82k). Several of those intervals exclude zero, but they
resample the test blocks under a single seed; three seeds of the plain-table model in leg 1 span
0.3907–0.3938, a range wider than any difference in the leg-2 rows. Read as a whole: no effect.

The reason is in the data. Coordinates here are fiscal-block centroids, 10,242 distinct points in
Level A against 10,022 postal codes, and nine codes in ten span less than about 320 m. The postal
code is close to a second spelling of the coordinate pair the model already has, with 2.4 rows per
code in Level A and 5.2 in the full base to learn it from. That the sign turns from cost to gain as
the rows per code double fits this reading; so does the split by seen and unseen codes on the full
base (−0.0013 [−0.0019, −0.0007] where the code occurs in training, −0.0006 [−0.0023, +0.0011] where
it does not), though the two intervals overlap and the check does not settle it.

Two side observations. Declaring the columns categorical (`categorical_features_indices`) instead of
sending plain strings returned byte-identical predictions from a separate fit: the server already
treats them as categories. And the mean under-prediction of 2026 drops by a fifth to a quarter with
both names in (bias +0.055 → +0.045 in ln on Level A, +0.058 → +0.043 on the full base) while the RMSE
stays put; we report it without an explanation.

For the bet, this is the useful negative result: two raw numeric columns already carry what the
address carries. The converse test, the postal code *instead of* the coordinates, is the next section.

## Spatial analysis from a ZIP code

*No coordinates at all: the raw postal code as a text column in their place, against the spatial
specialists with every one of their spatial inputs.*

The CEP (*Código de Endereçamento Postal*) is Brazil's ZIP code: eight digits, written `01310-100`,
assigned by the postal service, in São Paulo usually to a single street or a stretch of one. It is
on every tax form, deed and listing, and using it needs no geocoding. To a model it is a nominal
variable of very high cardinality: 10,022 distinct values in the 24,000 rows of Level A, 15,938 in
the full 2025 base (about five transactions per code there), and nothing in the column says which
code lies next to which.

The test takes the plain table, removes latitude and longitude, and puts the postal code in their
place as a string, so that it cannot be read as a number. In the strict version the distance to the
nearest station goes as well, because it is computed from the coordinates; the postal code is then
the only spatial information TabPFN-3.5 receives. The baselines are untouched: SAR keeps its k-NN-8
weights matrix and ρ; XGBoost keeps the k-NN-8 spatial lag, the rotated coordinates, the station
distance and the nested tuning; the geographically weighted XGBoost keeps its local models. Same
rows, folds and blocks, zero-shot, seed 42
(`scripts/50_nominal_location_ablation.sh replace`). One side models space with coordinates,
neighbours and a weights matrix; the other is handed a label.

**One year ahead, the label wins.** Leg 2, postal code only:

| Trained on | Model | Sees space through | RMSE (ln) | MAPE | Moran's I of residuals |
|---|---|---|---|---|---|
| Level A (24k) | SAR lag | k-NN-8 weights matrix, ρ | 0.346 | 27.0 % | 0.224 |
| | XGBoost + spatial features | k-NN-8 lag, rotated coordinates, station distance | 0.288 | 21.8 % | 0.138–0.140 |
| | Geographically weighted XGBoost | local models over the 50 nearest rows | 0.314 | 24.0 % | 0.187–0.188 |
| | **TabPFN-3.5, postal code only** | **one text column** | **0.277** | **20.7 %** | **0.129** |
| Full base (82k) | SAR lag | k-NN-8 weights matrix, ρ | 0.326 | 25.0 % | 0.134 |
| | XGBoost + spatial features | k-NN-8 lag, rotated coordinates, station distance | 0.283–0.294 | 20.9–21.5 % | 0.103–0.107 |
| | Geographically weighted XGBoost | local models over the 342 nearest rows | 0.298–0.301 | 22.1–22.2 % | 0.163–0.166 |
| | **TabPFN-3.5, postal code only** | **one text column** | **0.268** | **19.8 %** | **0.091** |

*XGBoost models: range over their three seeds.*

Paired over the ten spatial blocks: ΔRMSE in ln (TabPFN − opponent), 95 % CI by block bootstrap,
blocks won (`scripts/51_postal_code_vs_baselines.py`, `results/postal_code_vs_baselines.md`):

| TabPFN-3.5, postal code only, against | fit on Level A (24k) | fit on the full base (82k) |
|---|---|---|
| XGBoost + lag, seed 42 (the run in the main paired tests) | −0.0114 [−0.0132, −0.0096] · 10/10 | −0.0255 [−0.0292, −0.0222] · 10/10 |
| XGBoost + lag, its best seed | −0.0112 [−0.0129, −0.0092] · 10/10 | −0.0141 [−0.0170, −0.0115] · 10/10 |
| Geographically weighted XGBoost, its best seed | −0.0364 [−0.0464, −0.0267] · 10/10 | −0.0291 [−0.0385, −0.0191] · 9/10 |
| SAR lag | −0.0688 [−0.0760, −0.0614] · 10/10 | −0.0572 [−0.0626, −0.0514] · 10/10 |

Wilcoxon p = 0.002 in every row except the 9-of-10 one (p = 0.004; the lost block is the smallest,
by 0.001–0.002). MAPE falls by 1.1 to 1.8 points against XGBoost+lag, by 2.3 to 3.3 against the
geographically weighted XGBoost and by 5.3 to 6.3 against SAR.

Every refutation criterion of the bet is passed, at both training scales and against every seed, by
a model that never saw a coordinate. That includes the sharpest one: it leaves less spatial
autocorrelation in its residuals than the three models built to capture it. The Moran's I is computed
on a k-NN-8 matrix of the very coordinates the model was denied, so it is graded on a neighbourhood
structure it was never shown.

**A categorical column doing the spatial work.** Nothing was built around the code: no target
encoding, no embedding, no neighbour list, no lookup of coordinates. The string goes to the API as it
is, and the server treats it as a category (declaring it categorical returned byte-identical
predictions in the ablation above). A lookup table of prices by code could not price a code it has
never seen; this model does. On the 2026 transactions whose code never occurs in the training base,
22 % of them against Level A and 9 % against the full base, the postal-code-only model still recovers
97 % and 96 % of the way from its no-location floor to the coordinate model, about as much as on the
codes it has seen (98 % and 94 %). Why is not settled. The leading hypothesis is order: Brazilian
postal codes are assigned geographically, so neighbouring codes mostly mean neighbouring streets. The
test that would settle it has not been run (see [What is not here](#what-is-not-here-and-why)).

**Is it the model, or the code?** The postal code is information the baselines were not given, so
the objection is fair. Two results answer it. Without the code, the same model with nothing spatial
falls well behind XGBoost (0.352 on Level A, 0.327 on the full base): the code is what carries
location. And it carries no more location than the coordinate pair: inside TabPFN it comes within
0.004 of what latitude and longitude give (0.277 vs 0.275; 0.268 vs 0.265), and the ablation above
found that it adds nothing on top of them. XGBoost had the coordinate pair and, on top of it, a
spatial lag, rotated axes and the station distance. The location information is no richer on
TabPFN's side and all the spatial engineering is on the other: the difference is the model.

**Where it stops: a part of the city never seen.** In the leave-one-block-out CV (leg 1) a held-out
block's postal codes are absent from training by construction (97–100 %). There the code carries
nothing, and worse than nothing. The postal-code-only model lands below its own no-location floor
(RMSE 0.457 vs 0.440), loses to XGBoost+lag on pooled error (+0.069 to +0.072 in ln across its three
seeds, intervals just clear of zero, 4 of 10 blocks won), trails the geographically weighted XGBoost
(+0.047 to +0.050, intervals across zero, 4 of 10), ties SAR (+0.009 [−0.074, +0.069], 8 of 10
blocks) and leaves the most spatial autocorrelation of any model in the study (Moran's I 0.502,
against 0.390 for OLS). A whole block of codes the model has never seen is priced as one unknown, the
error shifts as a block, and Moran's I measures exactly that. Coordinates extrapolate into an unseen
district; a label cannot. The written expectation for this leg was that the code would fall to the
floor; it fell below it.

**Against the expectations written before the runs** (`src/model_tabpfn.py`). The postal-code-only
variant was expected to recover less than half of the way to the plain table and to lose to
XGBoost+lag in both legs. In leg 2 it recovered 97 % (24k) and 94 % (82k) and won every block. The
trigger stated in advance, "a name-based input that matches the plain table in leg 2", fired, and the
claim changes as announced: within a market the model has seen, it is not two raw coordinate columns
that carry the spatial signal but any fine-grained location identifier, a raw text code included.
For the bet this is the same claim in a stronger form: no weights matrix, no lag, no engineered
neighbourhood features and, inside the market, not even coordinates.

The other descriptors (`results/location_descriptor_summary.md`): the code with the station distance
kept does as well (0.277 / 0.264, 10 of 10 blocks against XGBoost and against SAR); the code read as
a number does as well as the string; the free-text district field recovers only 16–19 %, being empty
in 38 % of the 2026 forms; and the distance to a station alone carries about 40 % (24k) and 50 %
(82k) of the location signal in leg 2.

**Limits of the reading.** The TabPFN runs are single-seed; its seed spread in leg 1 is 0.003 in ln,
far below every leg-2 margin above. The mean under-prediction of 2026 is about that of the
coordinate model (bias +0.048 and +0.064 in ln, against +0.055 and +0.058) and larger than SAR's
(+0.03). And the result holds for valuation inside an observed market; leg 1 is where it does not.

For practice the reading is simple. Inside a market with a sales history, the model prices location
from the postal code already on the tax form, with no geocoder, no neighbour list and no weights
matrix, and leaves less spatial structure in its errors than the models that build one. For a region
with no sales in the training base, geocode.

## Text variant — the unit and the building as written

*Coordinates kept; two free-text fields of the ITBI form added exactly as they were typed.*

The form carries two pieces of text that no other run uses. The unit complement says which unit
changed hands and often what came with it: `AP 201 E 3VGS` (apartment 201 with three parking spaces),
`CJ 1904 TORRE B` (office suite 1904, tower B), `LOJA 3` (shop 3). The *Referência* field, filled on
64 % of the cleaned forms, mostly names the building or the development: `EDIFICIO THE PARK`,
`UP VILLAGE BY HELBOR`, `CJ HAB SAFIRA IV` (a social-housing complex), mixed with towers, landmarks
and registry notes. `pipeline/05_reference_field.py` recovers it from the original workbooks through
the deduplication key of the cleaning pipeline (every cleaned row matched; the cleaned bases are not
touched). Both columns go to TabPFN-3.5 as raw strings next to the plain table: nothing parsed,
nothing encoded (`--text complemento referencia`, `scripts/70_text_and_grouped_thinking.sh`).

The expectation written before the runs (`src/model_tabpfn.py`) was a small gain at most, in leg 2,
where the same buildings sell again, with leg 1 as a near-negative control: inside a building, the
floor read from the unit number barely moves the price (about 0.01 % per floor over 51,495
apartments in 9,801 buildings).

| | plain table | + unit and building text | ΔRMSE (ln) [95 % CI] · blocks won |
|---|---|---|---|
| Leg 1, block CV on Level A | 0.3938 · MAPE 32.2 % · Moran's I 0.348 | 0.3835 · 30.8 % · 0.331 | −0.0103 [−0.0161, −0.0031] · 8/10 |
| Leg 2, 24k → 2026 | 0.2750 · 20.4 % · 0.122 | 0.2750 · 20.6 % · 0.122 | +0.0000 [−0.0011, +0.0010] · 6/10 |
| Leg 2, 82k → 2026 | 0.2649 · 19.5 % · 0.092 | 0.2610 · 19.2 % · 0.086 | −0.0039 [−0.0049, −0.0031] · 10/10 |

The expectation was wrong about where. The largest gain is in leg 1, the part of the city the model
has never seen. It holds against each of the three seeds of the plain table (−0.0072 to −0.0103,
every interval clear of zero), and against XGBoost+lag it turns the pooled error from a deficit into
a tie leaning TabPFN's way (−0.001 to −0.004 across its seeds, intervals across zero, 7 of 10 blocks);
against the geographically weighted XGBoost the tie becomes a win (−0.023 to −0.026, every interval
clear of zero, 9 of 10 blocks, p = 0.027). Its MAPE, 30.8 %, is the lowest of any model in that leg.
Moran's I falls from 0.348 to 0.331, still above XGBoost+lag's 0.309 and the geographically weighted
XGBoost's 0.319–0.326: the partial refutation stands, narrower.

Where the gain comes from, read off the leg-1 predictions (no extra runs): forms whose complement
mentions parking (−0.034), commercial units (−0.023) and forms that name a building (−0.017); houses
and empty complements show none. It is not string matching. In leg 1 a held-out block's building
names are new to the model (92 % of them never occur in the training folds); complements never seen
as exact strings gain as much as those seen (−0.014 and −0.013), and among those that mention parking
the unseen ones gain more (−0.043 against −0.020). What transfers to an unseen district is what the words say (a price that
includes parking, an office suite rather than a shop, a named development), not which building they
point to. One year ahead the same words add 0.004 with 82k training rows (commercial units −0.012)
and nothing with 24k.

For the bet this is a second way the plain model reaches the specialists in leg 1 without a line of
spatial engineering: not by modelling space, but by reading the form the way an appraiser does.

## Thinking with the blocks as groups

Thinking mode spends extra fit-time compute tuning the model on internal validation splits. With
`group_col` the splits never cut a group; with the spatial block as the group, the internal
validation looks like leg 1, a whole district unseen, which is where the plain-table model is
weakest. The reference is the same thinking mode without groups (`--thinking medium --group-col`).

| | thinking (medium) | thinking, blocks as groups | ΔRMSE (ln) [95 % CI] · blocks won |
|---|---|---|---|
| Leg 1, block CV on Level A | 0.3889 · Moran's I 0.342 | 0.3927 · 0.350 | +0.0038 [−0.0055, +0.0149] · 3/10 |
| Leg 2, 24k → 2026 | 0.2752 · 0.121 | 0.2738 · 0.123 | −0.0014 [−0.0021, −0.0007] · 9/10 |

Grouping does not close the leg-1 gap: the difference is inside its interval and the residual
autocorrelation does not move. One year ahead it gains 0.0014 and cuts the mean under-prediction
from +0.055 to +0.043 in ln, but there the block also tells the model which part of the city a 2026
row belongs to (the API may use a group's own rows to predict it), so the small gain is not purely
a tuning effect. On an unseen district the limit is information, not validation design: the text
variant moves leg 1, the grouping does not.

## The app

Live at [tabpfn-itbi-sp.streamlit.app](https://tabpfn-itbi-sp.streamlit.app), or `streamlit run app/app.py` locally. Four tabs on the versioned results: the claim and scoreboard, a map
of where each model fails on the 2026 transactions, an inspector for any single transaction (four
estimates, the k-NN-8 neighbourhood the baselines saw, SHAP for the 50 explained properties), and,
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
| `data/english/*_en.csv` | English-header copies (not the originals) + `column_mapping.csv` |
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
Court. Code developed with AI assistance (Claude); every methodological decision is the author's.
