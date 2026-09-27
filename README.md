# Mass Appraisal of São Paulo Real Estate with TabPFN-3.5

**Prior Labs TabPFN-3.5 Hackathon entry** — spatial machine learning on
**82,187 actual transaction prices** from the São Paulo ITBI — *Imposto
sobre Transmissão de Bens Imóveis*, the Brazilian municipal tax on real
estate transfers (real-estate transfer tax) — with an out-of-time test on
47,810 transactions from 2026.

## Why this matters

Mass appraisal in Brazil (and in most of the world) still relies on linear
hedonic and spatial econometric models. A previous study by the author —
awarded at the CEAD 2026 congress — compared OLS, SAR, SEM, PS-SAR and
TabPFN v3 on 5,005 apartment *listings* in Belo Horizonte, with TabPFN
winning out-of-sample. This project raises the bar in four ways:

1. **Actual prices, not listings**: ITBI forms record what buyers declared
   they paid, month by month, for the whole city of São Paulo.
2. **Scale**: 82k training records across three property types
   (apartment / house / commercial), with a 24k stratified sample
   ("Level A") for a three-model comparison — SAR (spatial econometrics),
   G-XGBoost (geographically weighted gradient boosting) and **TabPFN-3.5**
   — and the full base ("Level B") for TabPFN-3.5 alone.
3. **Honest evaluation**: spatial block cross-validation (K-means, k=10),
   out-of-time testing (train on 2025, predict 2026), multiple seeds,
   Moran's I of residuals, paired Wilcoxon + block bootstrap.
4. **Practice-ready outputs**: predictive distributions → 80% prediction
   intervals per property → empirical coverage and precision grading under
   the Brazilian appraisal standard **NBR 14653-2**, plus traceable
   comparables (which training rows drive each prediction) and SHAP values.

## Glossary of Brazilian terms

Portuguese terms appear throughout the source data; every one used in this
repository is translated here and at first mention in each document.

| Term | Meaning |
|---|---|
| ITBI (*Imposto sobre Transmissão de Bens Imóveis*) | Municipal tax on real-estate transfers (real-estate transfer tax); its paid forms ("guias") record declared transaction prices |
| IPTU (*Imposto Predial e Territorial Urbano*) | Annual municipal property tax; source of the physical attributes (areas, use, standard, completion year) |
| SQL (*Setor, Quadra, Lote*) | 11-digit cadastral key: sector (3) + block (3) + lot (4) + check digit (1) |
| VVR (*Valor Venal de Referência*) | Municipal reference assessed value; the ITBI tax base is max(declared price, VVR) |
| ACC (*Ano de Construção Corrigido*) | Corrected construction-completion year from the IPTU register |
| Quadra fiscal | Fiscal block — the city-block polygon of the cadastre (GeoSampa layer) |
| Guia | The individual tax form; one paid form = one transaction record |
| SFH / MCMV (*Sistema Financeiro de Habitação / Minha Casa Minha Vida*) | Housing-finance system / federal affordable-housing programme (financing categories) |
| NBR 14653-2 | Brazilian valuation standard for urban properties (precision grading used here) |
| GeoSampa | Open geoportal of the São Paulo City Hall (*Prefeitura de São Paulo*) |

## Data

All data is public. See [`DATA_NOTICE.md`](DATA_NOTICE.md) for sources,
download dates and terms. Raw files keep their original Portuguese names as
downloaded from the official São Paulo City Hall sites (provenance);
English-labelled convenience copies are provided under `data/english/`.

| File | Content |
|---|---|
| `data/raw/GUIAS DE ITBI PAGAS 2025 XLS.xlsx` | 230,525 ITBI forms paid in 2025 (12 monthly sheets + documentation sheets) |
| `data/raw/GUIAS DE ITBI PAGAS 2026 XLS.xlsx` | 135,655 ITBI forms paid Jan–Jul 2026 |
| `data/raw/quadra_fiscal_p*.gpkg` | 64,223 fiscal-block polygons (GeoSampa WFS, EPSG:31983) |
| `data/raw/estacao_metro.geojson`, `estacao_trem.geojson` | 203 subway and commuter-rail stations |
| `data/fiscal_block_centroids.csv`, `data/fiscal_block_lookup.csv` | derived block centroids (pipeline step 01) |
| `data/itbi_sp_2025_train.csv` | cleaned 2025 training base (82,187 rows) |
| `data/itbi_sp_2026_test.csv` | cleaned 2026 out-of-time test base (47,810 rows) |
| `data/itbi_sp_2025_level_a.csv` | stratified 24,000-row sample for the 3-model comparison |
| `data/english/*_en.csv` | same datasets with English column headers (convenience copies — **not** the originals; see DATA_NOTICE.md) + `column_mapping.csv` |

Cleaning decisions are fully documented with per-step record counts in
[`data/filter_report.md`](data/filter_report.md); column semantics in
[`data/data_dictionary.md`](data/data_dictionary.md). A notable step:
because the São Paulo ITBI tax base is max(declared price, assessed
reference value), under-declared prices bunch exactly at the assessed value —
we detect and remove that bunching for cash purchases (financed deals are
lender-audited); see
[`data/underdeclaration_report.md`](data/underdeclaration_report.md).

## Reproducing

```bash
pip install -r requirements.txt

# 1. fiscal-block centroids from the GeoSampa GPKG pages
python pipeline/01_fiscal_block_centroids.py

# 2. cleaning: raw workbooks -> train/test/level_a CSVs + reports
python pipeline/02_itbi_cleaning.py

# 3. under-declaration diagnostics (residual picture)
python pipeline/03_underdeclaration_diagnostics.py

# 4. English-labelled convenience copies (data/english/)
python pipeline/04_english_labels.py

# 5. evaluation-protocol smoke test (trivial baseline through the full
#    machinery: spatial-block CV, metrics, Moran's I) — no API credits used
python -m src.protocol --data data/itbi_sp_2025_level_a.csv --smoke
```

All seeds are fixed (42); no downloads or geocoding APIs are called at
runtime.

## Repository layout

```
data/raw/      original public files as downloaded (see DATA_NOTICE.md)
data/          derived datasets and data reports
pipeline/      data preparation scripts (01, 02, 03)
src/           evaluation protocol and models (SAR, G-XGBoost, TabPFN-3.5)
notebooks/     exploratory analyses
results/       metrics, figures and maps
app/           Streamlit demo ("Avaliador SP")
paper/         reference paper (English translation forthcoming)
```

## Roadmap

- [x] Data pipeline with documented filter chain and georeferencing
- [x] Evaluation protocol (`src/protocol.py`): spatial-block CV, metrics,
      Moran's I, Wilcoxon + block bootstrap, multi-seed CIs
- [ ] SAR baseline (`spreg`), G-XGBoost, TabPFN-3.5 ablation grid
      (T0 base → T1 +spatial lag → T2 +high-cardinality categoricals →
      T3 Thinking → T4 +text)
- [ ] Out-of-time test 2025 → 2026; financed-only robustness run
- [ ] 80% prediction intervals, NBR 14653-2 precision grades, traceable
      comparables, SHAP
- [ ] Streamlit app and 2–3 min video

## License

Code and documentation under [Apache 2.0](LICENSE). Data notices in
[`DATA_NOTICE.md`](DATA_NOTICE.md). To cite this work, see
[`CITATION.cff`](CITATION.cff).

## Author

**Agnaldo Calvi Benvenho** — mechanical engineer (POLI-USP, Polytechnic
School of the University of São Paulo), MSc in Appraisal Engineering
(Universidad Politécnica de Valencia), Certification Director at IBAPE
Nacional (*Instituto Brasileiro de Avaliações e Perícias de Engenharia* —
Brazilian Institute of Engineering Appraisals and Expert Examinations),
court-appointed expert at the São Paulo State Court (TJSP). Code developed
with AI assistance (Claude); all methodological decisions by the author.
