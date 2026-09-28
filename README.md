# Location, location, location

*TabPFN-3.5 with the plain table vs models that see space explicitly, on 82,187 real transaction
prices from São Paulo, tested one year ahead on 47,810 more.*

Prior Labs TabPFN-3.5 Hackathon entry · [Streamlit app](app/MANUAL.md) · Apache-2.0

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
  (RMSE 0.265 vs 0.287 in ln, MAPE 19.5 % vs 21.0 %), beats the SAR lag model by a wide margin, and
  leaves the *least* spatial autocorrelation in its residuals of the four models (Moran's I 0.092).
* Predicting a part of the city it has never seen (leave-one-block-out CV), it ties XGBoost+lag on
  error and beats SAR, but leaves *more* spatial structure in its residuals than XGBoost with the
  explicit lag. That is a partial refutation, and it is reported as one.

Everything below is how those two sentences were earned.

## The bet, written so it can lose

> A tabular foundation model that receives only the property attributes and the coordinates as two
> plain numeric columns — no weights matrix, no spatial lag, no engineered neighbourhood features —
> predicts spatially correlated prices as well as, or better than, specialised models that encode
> spatial dependence explicitly.

The comparison is rigged against the bet. The baselines keep every spatial advantage: the SAR lag
model has its k-NN weights matrix and ρ; XGBoost gets a fold-internal k-NN-8 spatial lag, rotated
coordinate axes and nested Optuna tuning. TabPFN-3.5 gets the plain table (built and lot area, age,
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

The four models:

| Model | Sees space through |
|---|---|
| OLS hedonic | nothing (coordinates as covariates only) |
| SAR lag (`spreg` GM_Lag) | k-NN-8 weights matrix and ρ |
| XGBoost + spatial features | fold-internal k-NN-8 lag (leave-one-out for training rows), UTM coordinates + 30°/45°/60° rotations, nested Optuna tuning |
| **TabPFN-3.5, plain table** | latitude and longitude as two numeric columns; zero-shot or thinking mode |

Geographically weighted XGBoost, one of the methods in the reference study, was piloted and timed
(`src/pilot_gxgb_timing.py`): one local model per training row and per bandwidth candidate over a
dense distance matrix, roughly 0.5 h per bandwidth at 20k rows and 2.9 GB of matrix per fold. Not
feasible on this data; the XGBoost above is the scalable substitute that keeps the idea.

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
| **XGBoost + rotated coordinates + k-NN-8 lag** | **0.385** (seeds 0.384–0.387) | 32.3 % | **0.376** | **0.309** |
| **TabPFN-3.5, plain table, zero-shot** | 0.394 (seeds 0.390–0.394) | 32.2 % | 0.346 | 0.348 |
| TabPFN-3.5, plain table, thinking (medium) | 0.389 | **31.7 %** | 0.362 | 0.342 |
| TabPFN-3.5, plain table, thinking (high) | 0.388 | **31.7 %** | 0.366 | 0.339 |

What the paired tests say (`results/compare_*.json`): TabPFN zero-shot beats SAR in 8 of 10 blocks
(ΔRMSE −0.054, bootstrap 95 % CI [−0.100, −0.014], Wilcoxon p = 0.010). Against the fully spatial
XGBoost it is a statistical tie: ΔRMSE +0.009 [−0.018, +0.034], 7 blocks won, one clear loss in the
largest block. Thinking mode helps a little and consistently (−0.005 [−0.008, −0.001] vs zero-shot)
and gives the lowest MAPE in the table; high effort costs 13× more per fold for nothing beyond noise.

On the residual criterion the bet takes a hit. TabPFN's Moran's I (0.348) sits below SAR (0.363) but
above XGBoost with the explicit lag (0.309). Inside XGBoost, the lag alone is worth ΔRMSE −0.013 and
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
| **TabPFN-3.5, plain table, zero-shot** (fit in 9 s) | **0.275** | **20.4 %** | **0.622** | +0.055 | **0.122** |
| TabPFN-3.5, plain table, thinking (medium) | 0.275 | 20.5 % | 0.621 | +0.055 | 0.121 |

| Trained on the full 2025 base (82,187 rows) | RMSE (ln) | MAPE | R² (ln) | Bias (ln) | Moran's I |
|---|---|---|---|---|---|
| OLS hedonic | 0.402 | 32.4 % | 0.194 | +0.023 | 0.397 |
| SAR lag | 0.326 | 25.0 % | 0.470 | +0.031 | 0.134 |
| XGBoost + rotated coordinates + k-NN-8 lag | 0.287 (seeds 0.280–0.294) | 21.0 % | 0.59 | +0.07 to +0.10 | 0.105 |
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

Errors are flat across blocks (0.25–0.32) and across horizons of one to seven months (0.26–0.29): no
subset is carrying the result.

![RMSE per block, both legs](results/figures/fig2_per_block_rmse.png)

*Figure 3. RMSE per block. Left: block CV on Level A (tie with XGBoost+lag). Right: 2026, full base
(10 of 10).*

![Residual Moran's I, both legs](results/figures/fig5_moran.png)

*Figure 4. Residual Moran's I. Left: the partial refutation in the block CV. Right: one year ahead,
inside the city, the plain-table model leaves the least spatial structure of the four.*

![Mean residual by month ahead](results/figures/fig4_months_ahead_bias.png)

*Figure 5. Mean residual by month after the training window, full 2025 base.*

**The limitation worth stating.** Every model under-predicts 2026 (prices rose), and the two
non-linear learners more so (bias +0.05 to +0.10 in ln vs +0.02 to +0.03 for the linear models):
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
to encode with a weights matrix. Directions are the ones an appraiser expects: older buildings and
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

## The app

`streamlit run app/app.py` opens four tabs on the versioned results: the claim and scoreboard, a map
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

python -m src.protocol --data data/itbi_sp_2025_level_a.csv --smoke   # protocol on a trivial baseline, no API

python -m src.model_sar --estimator ols             # OLS
python -m src.model_sar                             # SAR lag
python -m src.model_xgb --seeds 42 43 44            # XGBoost + spatial features (~70 min on 2 vCPU)
python -m src.model_xgb --no-lag --seeds 42 43 44   # ablation

export TABPFN_TOKEN=...                             # TabPFN-3.5 through the API (separate venv: requirements-tabpfn.txt)
bash scripts/10_tabpfn_level_a.sh t0                # zero-shot + paired tests
bash scripts/10_tabpfn_level_a.sh t0-think          # thinking mode

python -m src.out_of_time --model ols --train level_a   # leg 2, baselines (also: sar, xgb; --train full)
bash scripts/20_tabpfn_out_of_time.sh t0 && bash scripts/20_tabpfn_out_of_time.sh compare

python -m src.make_figures                          # figures from the versioned results, no model re-run
bash scripts/30_tabpfn_shap.sh run && python -m src.make_figures --shap
bash scripts/40_financed_robustness.sh baselines && bash scripts/40_financed_robustness.sh tabpfn
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

Column semantics: [`data/data_dictionary.md`](data/data_dictionary.md). Raw files keep their
original Portuguese names for provenance.

## Layout

```
data/       raw public files, derived datasets, data reports
pipeline/   data preparation (01–04)
src/        protocol, SAR/OLS, XGBoost, TabPFN-3.5, out-of-time, SHAP, figures
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

## License and citation

Code and documentation under [Apache 2.0](LICENSE). Data notices in
[`DATA_NOTICE.md`](DATA_NOTICE.md). To cite, see [`CITATION.cff`](CITATION.cff).

## Author

**Agnaldo Calvi Benvenho**, mechanical engineer (POLI-USP), MSc in Appraisal Engineering
(Universidad Politécnica de Valencia), Certification Director at IBAPE Nacional (Brazilian Institute
of Engineering Appraisals and Expert Examinations), court-appointed expert at the São Paulo State
Court. Code developed with AI assistance (Claude); every methodological decision is the author's.
