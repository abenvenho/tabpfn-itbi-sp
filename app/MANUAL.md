# App manual — *Location, location, location*

A Streamlit app that lets you walk through the study without opening a JSON file: the claim, the
scoreboard, a map of where each model fails, a per-transaction inspector, and — with an API token —
a live TabPFN-3.5 estimate for a property you describe.

Everything except the last tab is read from the files versioned in this repository (`data/`,
`results/`). No model is re-run; the numbers on screen are the ones in the README.

## 1. Run it locally

```bash
git clone https://github.com/abenvenho/tabpfn-itbi-sp.git
cd tabpfn-itbi-sp
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
streamlit run app/app.py
```

The browser opens at `http://localhost:8501`. First load takes a few seconds (47,810 transactions
and four sets of predictions are read and cached in memory).

Requirements: Python 3.10+, ~1 GB of RAM, no GPU. The street basemap is fetched from a public tile
server; without internet, untick *Street basemap* in the sidebar and the maps fall back to plain
latitude/longitude axes.

## 2. Sidebar — filters that apply everywhere

| Control | Effect |
|---|---|
| **Training base for the 2026 predictions** | *Level A* (24,000 rows): all four models are available. *Full 2025 base* (82,187 rows): the same four models fitted on everything. Switches the tables in tab 1, the map in tab 2 and the estimates in tab 3. |
| **Property type** | apartment / house / commercial — filters the map and the counters in tab 2. |
| **Financing** | all / financed only / cash only. Financed deals carry a bank appraisal and are the cleanest price signal; the README has a robustness run on them. |
| **Street basemap** | On by default. Off = offline mode. |

## 3. The tabs

### Tab 1 · The claim and the scoreboard

The hypothesis in one paragraph, the two legs of the protocol with their verdicts, and the two
headline tables (block cross-validation on 2025; out-of-time test on 2026). Below them, two charts:
RMSE per spatial block for the four models, and the paired difference TabPFN-3.5 − baseline with its
block-bootstrap 95 % interval (negative = TabPFN better). Every number is read from
`results/cv_*.json`, `results/oot_*.json` and `results/oot_paired_comparisons_*.csv`.

### Tab 2 · Where the models fail

Pick a model. The map shows its residuals on the 47,810 transactions of 2026 (residual = observed −
predicted, in ln of R$/m²): red where the model under-predicted, blue where it over-predicted.

* **Grid cells (≈500 m)** — one dot per cell, coloured by the mean residual or by the RMSE; cells with
  fewer than five transactions are hidden.
* **Individual transactions** — a random sample of points (slider), hover for SQL, type, unit value and
  residual.

Switching between XGBoost+lag and TabPFN-3.5 is the quickest way to see that both fail in the same
places (the city's edges, the mixed-use centre) and that the plain-table model fails a little less.
The counters on the left (transactions, RMSE, MAPE) follow the sidebar filters. The block table at
the bottom is the versioned per-block result and does not follow them.

### Tab 3 · Inspect a transaction

Choose one of the **50 properties explained with SHAP**, or type any 11-digit SQL (cadastral key) of
a 2026 transaction. The page shows:

* the property's attributes and declared price;
* what each of the four models estimated (R$/m², total, error vs the declared price);
* the eight nearest 2025 transactions of the same type — *data, not model*: this is the k-NN-8
  neighbourhood the SAR and XGBoost baselines were given, shown for context. Distance 0 m means the
  same fiscal block (coordinates are block centroids), so a building's own earlier sales come first.
  The mini-map shows the sixty nearest, coloured by unit value;
* for the 50 SHAP properties, the attribution of TabPFN-3.5's estimate: each attribute's contribution
  in ln and as a multiplier on R$/m² (`exp(SHAP)`), from the base value to the estimate.

If a SQL has several 2026 transactions (same building, several units), a selector lets you pick one.

### Tab 4 · Appraise a property

Calls TabPFN-3.5 through the Prior Labs API. It is **off unless a token is present**:

```bash
export TABPFN_TOKEN=...          # from https://ux.priorlabs.ai
streamlit run app/app.py
```

or, on Streamlit Community Cloud, a secret named `TABPFN_TOKEN`.

**The sidebar filters do not apply here.** They describe the 2026 test view of tabs 1–3 (which
predictions to show, which transactions to map). What changes a live estimate is the training base
chosen *inside the tab* and the property you describe. Three bases are offered, each with a cached
model record from the study: Level A (24,000 rows, the main analysis), the full 2025 base (82,187
rows) and financed deals only (7,211 rows, the robustness run). On first use of a base the app tries
to reload its cached record (`results/tabpfn_cache/oot_tabpfn_t0_*/`; works when the token belongs
to the account that fitted it) and otherwise fits a fresh model on those rows — 10–20 seconds, once
per session and base, cached in memory. The result line says which of the two happened.

Fill in the nine T0 fields (type, built and lot area, age, finish grade, distance to the nearest
station, coordinates) and press **Estimate**. The output is a unit value in R$/m² and a total, at the
December-2025 price level, plus the nearest 2025 sales of the same type for context. The model never
sees those neighbours; that is the point of the study.

Each estimate costs a few API credits (one prediction row against a 24k-row context).

## 4. Deploying a public link (Streamlit Community Cloud)

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repository.
2. *New app* → repository `abenvenho/tabpfn-itbi-sp`, branch `main`, main file `app/app.py`.
3. Optional: *Advanced settings → Secrets* → `TABPFN_TOKEN = "..."` to switch on tab 4.
4. Deploy. The first build takes a few minutes (it installs `app/requirements.txt`); the app then
   lives at `https://<name>.streamlit.app`.

The free tier (1 GB RAM) is enough: the app holds ~60 MB of data in memory.

## 5. What the app is not

It is a research demonstration on **declared** transaction prices. Estimates are model outputs with
no legal standing; they carry the errors measured in the study (MAPE ≈ 20 % on 2026 transactions,
systematic under-prediction of ~5 % in a rising market, no trend adjustment). Do not use them as an
appraisal.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Maps are empty, only a colour bar | No internet for the tile server. Untick *Street basemap*. |
| "No 2026 transaction with that SQL" | The key is not in the cleaned 2026 base (filter chain in `data/filter_report.md`). The app falls back to the first SHAP property. |
| Tab 4 shows a warning | No `TABPFN_TOKEN` in the environment or secrets. |
| Tab 4 gives the same value after changing sidebar filters | Expected: the sidebar filters the 2026 view, not the training data. Use the *Fit TabPFN-3.5 on* selector in the tab. |
| Tab 4 shows an error about `tabpfn-client` | `pip install tabpfn-client==0.6.0` (it pins pandas ≤ 2.3.3). |
| Slow first load | Normal: 65 MB of CSVs are parsed once and cached. |
| `use_container_width` deprecation notice in the terminal | Harmless on Streamlit ≥ 1.62; the app runs on 1.38+. |
