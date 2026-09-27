#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_itbi_cleaning.py — ITBI-SP data cleaning pipeline
=====================================================
Reads the raw ITBI (real-estate transfer tax) monthly workbooks published by
the São Paulo Municipal Finance Department (forms paid in 2025 and in
Jan–Jul 2026), applies a fully documented filter chain, georeferences every
record to the centroid of its fiscal block (setor+quadra prefix of the SQL
cadastral key), engineers the final feature set and writes:

  data/itbi_sp_2025_train.csv      full 2025 modelling base ("Level B")
  data/itbi_sp_2026_test.csv       out-of-time test base (forms paid in 2026)
  data/itbi_sp_2025_level_a.csv    stratified 24k sample ("Level A",
                                   3-model comparison: SAR x G-XGBoost x TabPFN-3.5)
  data/filter_report.md            record counts after every filter step,
                                   global and per property type
  data/data_dictionary.md          data dictionary (column semantics and roles)

Column names follow the Portuguese snake_case canon of the official source
(compatible with the `dadosimob` PyPI package); the data dictionary maps
each one to its meaning in English.

Reproducibility: no downloads, no geocoding APIs, fixed seeds (42).
Run from the project root:  python pipeline/02_itbi_cleaning.py

Revision history
----------------
v1 (2026-09-27)  filter chain as designed; Tukey fences estimated on 2025 and
                 re-applied to 2026; K-means spatial blocks (k=10, seed 42).
v2 (2026-09-27)  'terreno' (vacant land) and 'industrial' property types
                 dropped from the modelling scope (final samples of only
                 1,140 and 467); financing fields kept as diagnostic columns.
v3 (2026-09-27)  under-declaration treatment: cash purchases declared within
                 +/-1% of the reference assessed value (VVR) are excluded —
                 in São Paulo the ITBI tax base is max(declared, VVR), so the
                 only incentive is to declare exactly the VVR when the true
                 price is higher; the ratio histogram shows clear bunching
                 at 1 concentrated in non-financed deals. Tukey fences
                 re-estimated on the cleaner base. A robustness run on the
                 financed-only subsample is planned in the evaluation
                 protocol (lender-audited prices).
"""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
XLSX_2025 = RAW / "GUIAS DE ITBI PAGAS 2025 XLS.xlsx"   # original file names kept
XLSX_2026 = RAW / "GUIAS DE ITBI PAGAS 2026 XLS.xlsx"   # for provenance
LOOKUP_CSV = ROOT / "data" / "fiscal_block_lookup.csv"
GEOJSON_METRO = RAW / "estacao_metro.geojson"            # subway stations
GEOJSON_TREM = RAW / "estacao_trem.geojson"              # commuter-rail stations
OUT_DIR = ROOT / "data"

SEED = 42
EXCEL_ENGINE = "calamine"  # fast Rust reader; falls back to openpyxl

SUPPORT_SHEETS = {"LEGENDA", "EXPLICAÇÕES", "EXPLICACOES",
                  "TABELA DE USOS", "TABELA DE PADRÕES", "TABELA DE PADROES"}
MONTH_SHEET_RE = r"^[A-ZÇ]{3}-\d{4}$"

# Canonical column names (official 28-column layout, in file order)
CANONICAL_COLS = [
    "sql", "logradouro", "numero", "complemento", "bairro", "referencia",
    "cep", "natureza_transacao", "valor_transacao", "data_transacao",
    "valor_venal_referencia", "proporcao_transmitida",
    "valor_venal_referencia_proporcional", "base_calculo",
    "tipo_financiamento", "valor_financiado", "cartorio", "matricula",
    "situacao_sql", "area_terreno_m2", "testada_m", "fracao_ideal",
    "area_construida_m2", "uso_iptu", "descricao_uso", "padrao_iptu",
    "descricao_padrao", "acc_iptu",
]

# Property-type mapping by IPTU use code (see the 'Tabela de USOS' sheet).
# v2: vacant land (use 0) and industrial (50, 51, 26, 60) are out of scope.
PROPERTY_TYPE = {
    20: "apartment",
    10: "house", 14: "house",
    30: "commercial", 40: "commercial", 41: "commercial", 42: "commercial",
}
# Explicitly excluded uses: land, industrial, garages, flats/hotels,
# whole buildings, collective housing
EXCLUDED_USES = {0, 50, 51, 26, 60, 23, 24, 62, 63, 25, 80, 85, 21, 22, 31, 32, 12, 13}

LEVEL_A_TARGETS = {"apartment": 12_000, "house": 7_000, "commercial": 5_000}
TYPES_ORDER = ["apartment", "house", "commercial"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def read_monthly_sheets(path: Path, engine: str = EXCEL_ENGINE) -> pd.DataFrame:
    """Read every monthly sheet (JAN-YYYY ... DEZ-YYYY) of one workbook.

    Some sheets carry a stray 29th column ('Unnamed: 28') holding one or two
    orphan year cells; the rows themselves are intact, so columns are mapped
    positionally to the first 28 and the extra column is discarded.
    """
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine=engine)
    except Exception:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frames = []
    for name, df in sheets.items():
        key = unicodedata.normalize("NFC", name).strip().upper()
        if key in SUPPORT_SHEETS or not re.match(MONTH_SHEET_RE, key):
            continue
        if df.shape[1] < 28 or not str(df.columns[0]).startswith("N° do Cadastro"):
            raise ValueError(f"{path.name}/{name}: unexpected layout "
                             f"({df.shape[1]} cols, first={df.columns[0]!r})")
        extra_note = ""
        if df.shape[1] > 28:
            n_extra = int(df.iloc[:, 28:].notna().sum().sum())
            extra_note = (f" (dropped {df.shape[1] - 28} stray column(s), "
                          f"{n_extra} orphan cell(s))")
        df = df.iloc[:, :28].copy()
        df.columns = CANONICAL_COLS
        df["mes_referencia"] = key  # sheet name = month the ITBI form was PAID
        frames.append(df)
        log(f"  sheet {key}: {len(df):,} rows{extra_note}")
    return pd.concat(frames, ignore_index=True)


def to_canonical_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric/text/date columns; restore leading zeros of coded fields."""
    df = df.copy()

    def zfill_code(s: pd.Series, width: int) -> pd.Series:
        num = pd.to_numeric(s, errors="coerce")
        return num.map(lambda v: f"{int(v):0{width}d}" if pd.notna(v) else None)

    # Excel strips leading zeros of coded fields
    df["sql"] = zfill_code(df["sql"], 11)   # 3 setor + 3 quadra + 4 lote + 1 check digit
    df["cep"] = zfill_code(df["cep"], 8)    # postal code

    for col in [
        "valor_transacao", "valor_venal_referencia", "proporcao_transmitida",
        "valor_venal_referencia_proporcional", "base_calculo", "valor_financiado",
        "area_terreno_m2", "testada_m", "fracao_ideal", "area_construida_m2",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["uso_iptu"] = pd.to_numeric(df["uso_iptu"], errors="coerce").astype("Int64")
    df["acc_iptu"] = pd.to_numeric(df["acc_iptu"], errors="coerce").astype("Int64")

    padrao_num = pd.to_numeric(df["padrao_iptu"], errors="coerce")
    df["padrao_iptu"] = padrao_num.map(lambda v: f"{int(v):02d}" if pd.notna(v) else None)

    df["data_transacao"] = pd.to_datetime(df["data_transacao"], errors="coerce")

    for col in ["logradouro", "numero", "complemento", "bairro", "natureza_transacao",
                "situacao_sql", "descricao_uso", "descricao_padrao", "tipo_financiamento"]:
        df[col] = df[col].astype("string").str.strip()

    # Financing flag (diagnostic; also used by the under-declaration filter)
    df["financiado"] = (df["valor_financiado"] > 0) | df["tipo_financiamento"].notna()

    df["tipo_imovel"] = df["uso_iptu"].map(PROPERTY_TYPE)
    return df


class FilterReport:
    """Records global and per-type counts after each filter step."""

    def __init__(self, label: str):
        self.label = label
        self.rows: list[dict] = []

    def snapshot(self, step: str, df: pd.DataFrame, note: str = "") -> None:
        rec = {"step": step, "remaining": len(df), "note": note}
        vc = df["tipo_imovel"].value_counts(dropna=False)
        for t in TYPES_ORDER:
            rec[t] = int(vc.get(t, 0))
        rec["untyped"] = int(len(df) - sum(rec[t] for t in TYPES_ORDER))
        self.rows.append(rec)

    def to_markdown(self) -> str:
        tip_head = " | ".join(t.capitalize() for t in TYPES_ORDER)
        head = (f"| # | Step | Remaining | {tip_head} | Untyped | Notes |\n"
                f"|---|------|-----------|{'---|' * len(TYPES_ORDER)}---------|-------|\n")
        lines = []
        prev = None
        for i, r in enumerate(self.rows):
            delta = "" if prev is None else f" (−{prev - r['remaining']:,})"
            tips = " | ".join(f"{r[t]:,}" for t in TYPES_ORDER)
            lines.append(f"| {i} | {r['step']} | {r['remaining']:,}{delta} | "
                         f"{tips} | {r['untyped']:,} | {r['note']} |")
            prev = r["remaining"]
        return head + "\n".join(lines) + "\n"


def load_stations() -> np.ndarray:
    """Return an (n, 2) array of station coordinates in EPSG:31983
    (subway + commuter rail)."""
    pts = []
    for p in [GEOJSON_METRO, GEOJSON_TREM]:
        gj = json.loads(Path(p).read_text(encoding="utf-8"))
        for feat in gj["features"]:
            geom = feat["geometry"]
            if geom["type"] == "Point":
                pts.append(geom["coordinates"][:2])
            elif geom["type"] == "MultiPoint":
                pts.extend(c[:2] for c in geom["coordinates"])
    arr = np.asarray(pts, dtype=float)
    if not (150_000 < np.nanmean(arr[:, 0]) < 1_000_000):
        raise ValueError("Station coordinates do not look like EPSG:31983 (UTM 23S).")
    return arr


def min_station_distance(xy: np.ndarray, stations: np.ndarray) -> np.ndarray:
    """Min Euclidean distance (metres, UTM) from each row of xy to any station."""
    # ~200 stations: chunked brute force is fast and dependency-free
    out = np.empty(len(xy))
    chunk = 20_000
    for i in range(0, len(xy), chunk):
        block = xy[i:i + chunk]
        d2 = ((block[:, None, :] - stations[None, :, :]) ** 2).sum(axis=2)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1))
    return out


# --------------------------------------------------------------------------
# Filter chain
# --------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame, rep: FilterReport,
                  tukey_fences: dict | None,
                  lookup: pd.DataFrame,
                  stations: np.ndarray):
    """Apply the documented filter chain. Returns (clean_df, fences, extras)."""
    extras: dict = {}
    rep.snapshot("Raw base (all ITBI forms)", df)

    # --- 'off-plan' segment, counted separately BEFORE filtering -----------
    # Off-plan units are recorded as sales of an undivided fraction of the
    # mother lot (use 0, tiny transmitted share, status 'Ativo Territorial').
    # They never enter the comparison base (share < 100%), but are reported.
    off_plan = df[
        (df["uso_iptu"] == 0)
        & (df["proporcao_transmitida"] < 100)
        & df["natureza_transacao"].str.contains("compra e venda", case=False, na=False)
        & df["situacao_sql"].str.startswith("Ativo Territorial", na=False)
    ]
    extras["off_plan_total"] = len(off_plan)
    extras["off_plan_apto"] = int(
        off_plan["complemento"].str.contains(r"ap(?:to|art)", case=False, na=False).sum()
    )

    # 1. sale-and-purchase deeds only
    df = df[df["natureza_transacao"].str.contains("compra e venda", case=False, na=False)]
    rep.snapshot('Deed nature contains "compra e venda" (sale and purchase)', df)

    # 2. full-ownership transfers only
    df = df[df["proporcao_transmitida"] >= 99.999]
    rep.snapshot("Transmitted share = 100%", df)

    # 3. active cadastral status
    df = df[df["situacao_sql"].str.startswith("Ativo", na=False)]
    rep.snapshot('Cadastral status starts with "Ativo" (active)', df)

    # 4. valid 11-digit SQL key and valid transaction date
    df = df[df["sql"].notna() & df["sql"].str.fullmatch(r"\d{11}")
            & df["data_transacao"].notna()]
    rep.snapshot("Valid 11-digit SQL key and valid date", df)

    # 5. kept property types only
    df = df[df["tipo_imovel"].notna()]
    rep.snapshot("Property type kept (apartment/house/commercial)", df)

    # 6. positive reference area (all kept types are built properties)
    df = df.assign(area_ref=df["area_construida_m2"])
    df = df[df["area_ref"] > 0]
    rep.snapshot("Reference (built) area > 0", df)

    # 7. plausible construction year (ACC = corrected completion year)
    ano_tx = df["data_transacao"].dt.year
    acc_ok = (df["acc_iptu"] >= 1900) & (df["acc_iptu"] <= ano_tx)
    df = df[acc_ok.fillna(False)]
    rep.snapshot("Plausible ACC (1900 <= ACC <= transaction year)", df)

    # 8. deduplicate
    df = df.drop_duplicates(subset=["sql", "data_transacao", "valor_transacao"])
    rep.snapshot("Deduplication (sql, date, price)", df)

    # 9. minimum price
    df = df[df["valor_transacao"] >= 10_000]
    rep.snapshot("Declared price >= R$ 10,000", df)

    # 10. declared price vs. reference assessed value (VVR) — gross errors
    razao = df["valor_transacao"] / df["valor_venal_referencia"]
    out_of_band = (df["valor_venal_referencia"] > 0) & ~razao.between(0.3, 5.0)
    extras["vvr_ratio_excluded"] = int(out_of_band.sum())
    df = df[~out_of_band]
    df = df.assign(razao_vvr=np.where(df["valor_venal_referencia"] > 0,
                                      df["valor_transacao"] / df["valor_venal_referencia"],
                                      np.nan))
    rep.snapshot("Price/VVR ratio within [0.3, 5] (when VVR > 0)", df)

    # 10b. under-declaration (v3): in São Paulo the ITBI base is
    # max(declared, VVR), so the incentive is to declare EXACTLY the VVR when
    # the true price is higher. The ratio histogram shows clear bunching at 1
    # ([1.00,1.02) holds 4.3% of the base vs ~1.8% in neighbouring bins).
    # Cash deals declared within +/-1% of the VVR are excluded; financed
    # deals are kept (the lender's appraisal and loan contract audit the price).
    pegged = df["razao_vvr"].between(0.99, 1.01) & ~df["financiado"]
    extras["pegged_cash_excluded"] = int(pegged.sum())
    df = df[~pegged]
    rep.snapshot("Under-declaration: cash deals pegged to the VVR "
                 "(ratio in [0.99, 1.01], not financed) excluded", df)

    # 11. Tukey fences (1.5 IQR) on ln(price per m2), per property type.
    #     Fences are ESTIMATED ON THE 2025 BASE and re-applied to 2026, so
    #     the out-of-time test set never defines its own outlier limits.
    df = df.assign(ln_vu=np.log(df["valor_transacao"] / df["area_ref"]))
    if tukey_fences is None:
        tukey_fences = {}
        for t, g in df.groupby("tipo_imovel"):
            q1, q3 = g["ln_vu"].quantile([0.25, 0.75])
            iqr = q3 - q1
            tukey_fences[t] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    lo = df["tipo_imovel"].map(lambda t: tukey_fences[t][0])
    hi = df["tipo_imovel"].map(lambda t: tukey_fences[t][1])
    df = df[(df["ln_vu"] >= lo) & (df["ln_vu"] <= hi)]
    rep.snapshot("Tukey fences (1.5 IQR) on ln(R$/m2), per type — "
                 "fences from the 2025 base", df)

    # 12. georeference via fiscal-block centroid (setor+quadra = sql[:6])
    df = df.assign(setor_quadra=df["sql"].str[:6], setor=df["sql"].str[:3])
    df = df.merge(lookup, on="setor_quadra", how="left", suffixes=("", "_q"))
    matched = df["x_utm"].notna()
    extras["georef_match_rate"] = float(matched.mean())
    extras["georef_unmatched"] = int((~matched).sum())
    df = df[matched]
    rep.snapshot(f"Join sql[:6] to fiscal-block centroid "
                 f"(match {matched.mean():.2%})", df)

    # --- final engineered attributes ---------------------------------------
    df = df.assign(
        padrao_tipo=df["padrao_iptu"].str[0],
        padrao_nivel=pd.to_numeric(df["padrao_iptu"].str[1], errors="coerce").astype("Int64"),
        idade=df["data_transacao"].dt.year - df["acc_iptu"].astype("float"),
        mes_idx=(df["data_transacao"].dt.year - 2025) * 12
                + (df["data_transacao"].dt.month - 1),
        grupo_edificio=(df["sql"].str[:6] + "|"
                        + df["logradouro"].fillna("").str.upper() + "|"
                        + df["numero"].fillna("").astype(str)),
    )
    df = df.assign(dist_estacao_m=min_station_distance(
        df[["x_utm", "y_utm"]].to_numpy(dtype=float), stations))

    return df, tukey_fences, extras


FINAL_COLS = [
    # identifiers / traceability (never model features)
    "sql", "logradouro", "numero", "complemento", "grupo_edificio", "mes_referencia",
    # features
    "tipo_imovel", "uso_iptu", "descricao_uso", "padrao_iptu", "padrao_tipo",
    "padrao_nivel", "descricao_padrao", "idade", "area_construida_m2",
    "area_terreno_m2", "fracao_ideal", "testada_m", "x_utm", "y_utm", "lon", "lat",
    "bairro", "cep", "setor", "setor_quadra", "data_transacao", "mes_idx",
    "dist_estacao_m",
    # target and response components
    "valor_transacao", "area_ref", "ln_vu",
    # diagnostics only (leakage risk — never model features)
    "valor_venal_referencia", "razao_vvr",
    "financiado", "tipo_financiamento", "valor_financiado",
]


def main() -> None:
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("Reading fiscal-block centroid lookup and station layers ...")
    lookup = pd.read_csv(LOOKUP_CSV, dtype={"setor_quadra": str})
    lookup["setor_quadra"] = lookup["setor_quadra"].str.zfill(6)
    lookup = lookup[["setor_quadra", "x_utm", "y_utm", "lon", "lat"]]
    stations = load_stations()
    log(f"  lookup keys: {len(lookup):,} | stations: {len(stations)}")

    log("Reading 2025 workbook ...")
    raw25 = to_canonical_types(read_monthly_sheets(XLSX_2025))
    log(f"  2025 total: {len(raw25):,} rows")
    log("Reading 2026 workbook ...")
    raw26 = to_canonical_types(read_monthly_sheets(XLSX_2026))
    log(f"  2026 total: {len(raw26):,} rows")

    rep25 = FilterReport("2025")
    df25, fences, ex25 = apply_filters(raw25, rep25, None, lookup, stations)

    rep26 = FilterReport("2026")
    df26, _, ex26 = apply_filters(raw26, rep26, fences, lookup, stations)

    # Spatial blocks: K-means (k=10) on the 2025 UTM coordinates, shared
    # across property types; the SAME centroids are applied to 2026.
    log("Fitting K-means spatial blocks (k=10, seed 42) on the 2025 base ...")
    km = KMeans(n_clusters=10, random_state=SEED, n_init=10)
    df25 = df25.assign(bloco=km.fit_predict(df25[["x_utm", "y_utm"]].to_numpy(dtype=float)))
    df26 = df26.assign(bloco=km.predict(df26[["x_utm", "y_utm"]].to_numpy(dtype=float)))

    # Level A: stratified sample of the 2025 base for the 3-model comparison
    parts = []
    for t, target in LEVEL_A_TARGETS.items():
        g = df25[df25["tipo_imovel"] == t]
        parts.append(g.sample(n=min(target, len(g)), random_state=SEED))
    level_a = pd.concat(parts).sort_values(["tipo_imovel", "sql"]).reset_index(drop=True)

    # ---- write outputs -----------------------------------------------------
    cols = FINAL_COLS + ["bloco"]
    df25_out = df25[cols].sort_values(["data_transacao", "sql"]).reset_index(drop=True)
    df26_out = df26[cols].sort_values(["data_transacao", "sql"]).reset_index(drop=True)
    for d in (df25_out, df26_out, level_a):
        d["x_utm"] = d["x_utm"].round(2)
        d["y_utm"] = d["y_utm"].round(2)
        d["lon"] = d["lon"].round(6)
        d["lat"] = d["lat"].round(6)
        d["ln_vu"] = d["ln_vu"].round(4)
        d["razao_vvr"] = d["razao_vvr"].round(4)
        d["dist_estacao_m"] = d["dist_estacao_m"].round(1)

    p1 = OUT_DIR / "itbi_sp_2025_train.csv"
    p2 = OUT_DIR / "itbi_sp_2026_test.csv"
    p3 = OUT_DIR / "itbi_sp_2025_level_a.csv"
    df25_out.to_csv(p1, index=False)
    df26_out.to_csv(p2, index=False)
    level_a[cols].to_csv(p3, index=False)
    log(f"Wrote {p1.name} ({len(df25_out):,}), {p2.name} ({len(df26_out):,}), "
        f"{p3.name} ({len(level_a):,})")

    # ---- filter report -----------------------------------------------------
    def date_profile(d: pd.DataFrame) -> str:
        y = d["data_transacao"].dt.year.value_counts().sort_index()
        return ", ".join(f"{int(k)}: {v:,}" for k, v in y.items())

    def unit_value_table(d: pd.DataFrame) -> str:
        g = (d.assign(vu=np.exp(d["ln_vu"]))
               .groupby("tipo_imovel")["vu"]
               .agg(n="count", p25=lambda s: s.quantile(.25),
                    median="median", p75=lambda s: s.quantile(.75)))
        g = g.reindex(TYPES_ORDER)
        lines = ["| Type | n | P25 (R$/m2) | Median (R$/m2) | P75 (R$/m2) |",
                 "|---|---|---|---|---|"]
        for t, r in g.iterrows():
            if pd.isna(r["n"]) or r["n"] == 0:
                continue
            lines.append(f"| {t} | {int(r['n']):,} | {r['p25']:,.0f} | "
                         f"{r['median']:,.0f} | {r['p75']:,.0f} |")
        return "\n".join(lines)

    fence_lines = ["| Type | Lower fence ln(R$/m2) | Upper fence | ~min R$/m2 | ~max R$/m2 |",
                   "|---|---|---|---|---|"]
    for t in TYPES_ORDER:
        lo, hi = fences[t]
        fence_lines.append(f"| {t} | {lo:.3f} | {hi:.3f} | "
                           f"{math.exp(lo):,.0f} | {math.exp(hi):,.0f} |")

    rel = f"""# Cleaning report (v3) — ITBI-SP base (forms paid in 2025 and 2026)

Generated by `pipeline/02_itbi_cleaning.py` (seed = {SEED}). Workbook sheets
correspond to the **month the ITBI form was paid**; the transaction date may
be earlier. Counts after every step, global and per property type.

**v2 changes (2026-09-27):** `terreno` (vacant land) and `industrial` types
dropped from the modelling scope due to small final samples in v1 (1,140 and
467); `financiado`, `tipo_financiamento` and `valor_financiado` kept in the
CSVs as diagnostic columns (under-declaration investigation — never features).

**v3 changes (2026-09-27):** exclusion of cash purchases "pegged to the VVR"
(declared/VVR ratio in [0.99, 1.01], not financed) — in São Paulo the ITBI
tax base is max(declared, VVR), so the incentive to under-declare is to
declare exactly the VVR; the diagnostics (`data/underdeclaration_report.md`)
showed clear bunching at 1 concentrated in cash deals. Excluded:
{ex25['pegged_cash_excluded']:,} forms in 2025 and
{ex26['pegged_cash_excluded']:,} in 2026. Tukey fences re-estimated after the
exclusion. Additional decision: a robustness run of the evaluation protocol
on the financed-only subsample (lender-audited prices).

## 2025 base (training)

{rep25.to_markdown()}

## 2026 base (out-of-time test)

{rep26.to_markdown()}

## Tukey fences (estimated on 2025, re-applied to 2026)

{chr(10).join(fence_lines)}

**Methodological decision:** the outlier fences on ln(R$/m2) are estimated on
the 2025 base only and re-applied to the 2026 base, so that the out-of-time
test set never defines its own exclusion limits.

## Off-plan segment (counted separately, outside the comparison base)

Sales of an undivided fraction of the mother lot (use 0, share < 100%,
status "Ativo Territorial" — active, land-only registration — under
"compra e venda" / sale-and-purchase deeds):

| Base | Off-plan forms | of which with "Apto" in the unit field |
|---|---|---|
| 2025 | {ex25['off_plan_total']:,} | {ex25['off_plan_apto']:,} |
| 2026 | {ex26['off_plan_total']:,} | {ex26['off_plan_apto']:,} |

## Georeferencing (fiscal-block centroid, key sql[:6])

| Base | Match rate | Unmatched records (excluded) |
|---|---|---|
| 2025 | {ex25['georef_match_rate']:.2%} | {ex25['georef_unmatched']:,} |
| 2026 | {ex26['georef_match_rate']:.2%} | {ex26['georef_unmatched']:,} |

## Diagnostics — transaction-date year (final base)

- 2025: {date_profile(df25_out)}
- 2026: {date_profile(df26_out)}

Forms paid in a given year may refer to transactions from earlier years
(late regularisations). `mes_idx` is computed from the **transaction date**
(months since Jan 2025; negative = transaction before 2025).

## Unit values of the final base (sanity check)

### 2025
{unit_value_table(df25_out)}

### 2026
{unit_value_table(df26_out)}

## Spatial blocks and Level A sample

- K-means k = 10 on the UTM coordinates of the 2025 base
  (`random_state={SEED}`), the same centroids applied to the 2026 base
  (column `bloco`).
- Level A (SAR x G-XGBoost x TabPFN-3.5 comparison): stratified sample of
  the 2025 base — targets: {LEVEL_A_TARGETS} — {len(level_a):,} rows,
  `random_state={SEED}`, in `data/itbi_sp_2025_level_a.csv`.
- Level B (TabPFN-3.5 only): full 2025 base ({len(df25_out):,} rows).

## Diagnostic columns (NEVER use as features — leakage)

`valor_venal_referencia`, `razao_vvr`, `financiado`, `tipo_financiamento` and
`valor_financiado` remain in the CSVs for auditing only. Dropped from the
base entirely: proportional VVR, adopted tax base, notary and register
numbers.
"""
    (OUT_DIR / "filter_report.md").write_text(rel, encoding="utf-8")

    dic = """# Data dictionary — itbi_sp_2025_train.csv / itbi_sp_2026_test.csv / itbi_sp_2025_level_a.csv

Primary source: São Paulo Municipal Finance Department (Secretaria Municipal
da Fazenda) — ITBI (Imposto sobre Transmissão de Bens Imóveis, the municipal
real-estate transfer tax) forms paid in 2025 and Jan–Jul 2026.
Georeferencing: fiscal block ("quadra fiscal") centroid (GeoSampa layer
`geoportal:quadra_fiscal`, EPSG:31983), key `sql[:6]`. Column names keep the
Portuguese snake_case canon of the official source (compatible with the
`dadosimob` PyPI package); the "English alias" column matches the header
names used in the convenience copies under `data/english/` (see
`data/english/column_mapping.csv` and DATA_NOTICE.md).

| Column | English alias | Type | Description | Role |
|---|---|---|---|---|
| sql | cadastral_key_sql | text (11) | Cadastral key — SQL = Setor, Quadra, Lote (3 sector + 3 block + 4 lot + 1 check digit) | identifier — never a feature |
| logradouro | street_name | text | Declared street name | identifier — never a feature |
| numero | street_number | text | Street number | identifier — never a feature |
| complemento | address_unit | text | Address complement (unit/apartment) | identifier — never a feature |
| grupo_edificio | building_group | text | sql[:6] + street + number (groups units of the same building) | grouping/diagnostics |
| mes_referencia | payment_month_sheet | text | Source sheet = month the form was paid | diagnostics |
| tipo_imovel | property_type | categorical | apartment, house, commercial (from the IPTU use code) | feature |
| uso_iptu | iptu_use_code | integer | IPTU use code (IPTU = annual municipal property tax) | feature |
| descricao_uso | iptu_use_description_pt | text | Official use description (values in Portuguese) | feature (text, T4) |
| padrao_iptu | iptu_standard_code | text (2) | IPTU standard ("padrão"): 1st digit building class, 2nd finish grade (0–5) | feature |
| padrao_tipo | building_class_digit | categorical | 1st digit of the standard (building class) | feature |
| padrao_nivel | finish_grade | integer | 2nd digit of the standard (finish grade, 0–5 ascending) | feature |
| descricao_padrao | iptu_standard_description_pt | text | Official standard description (values in Portuguese) | feature (text, T4) |
| idade | age_years | float | Transaction year − ACC (Ano de Construção Corrigido, corrected completion year) | feature |
| area_construida_m2 | built_area_m2 | float | Built area, m2 (includes the common-area quota in condominiums) | feature |
| area_terreno_m2 | lot_area_m2 | float | Lot area, m2 | feature |
| fracao_ideal | undivided_share | float | Undivided interest ("fração ideal"; 1 outside condominiums) | feature |
| testada_m | frontage_m | float | Street frontage ("testada"), m; empty for landlocked lots | feature |
| x_utm, y_utm | x_utm, y_utm | float | Fiscal-block centroid, EPSG:31983 (metres) | feature |
| lon, lat | lon, lat | float | Fiscal-block centroid, WGS84 | feature |
| bairro | neighborhood_pt | text | Declared neighbourhood (values in Portuguese; high cardinality — T2) | feature |
| cep | postal_code | text (8) | Postal code (CEP) | feature |
| setor | fiscal_sector | text (3) | sql[:3] (fiscal sector) | feature |
| setor_quadra | sector_block_key | text (6) | sql[:6] — georeferencing key | feature |
| data_transacao | transaction_date | date | Declared transaction date | feature (via mes_idx) |
| mes_idx | month_index | integer | Months since Jan 2025, from the transaction date (negative = before 2025) | feature |
| dist_estacao_m | dist_station_m | float | Distance to the nearest subway/rail station (m, UTM) | feature |
| valor_transacao | declared_price_brl | float | Price declared by the taxpayer (R$) | response component |
| area_ref | reference_area_m2 | float | Reference area (= built area) | response component |
| ln_vu | ln_unit_price | float | ln(valor_transacao / area_ref) — **response variable** ("VU" = valor unitário, unit value) | response |
| valor_venal_referencia | assessed_ref_value_brl | float | Reference assessed value (VVR, Valor Venal de Referência) — diagnostics ONLY | leakage — never a feature |
| razao_vvr | price_to_assessed_ratio | float | price/VVR ratio — diagnostics ONLY | leakage — never a feature |
| financiado | is_financed | boolean | valor_financiado > 0 or a financing type declared | diagnostics — never a feature |
| tipo_financiamento | financing_type_pt | text | Financing category, values in Portuguese: SFH (housing-finance system), MCMV (federal affordable-housing programme), consórcio (consortium), SFI/carteira hipotecária (mortgage portfolio) | diagnostics — never a feature |
| valor_financiado | financed_amount_brl | float | Declared financed amount (R$) | diagnostics — never a feature |
| bloco | spatial_block | integer (0–9) | K-means spatial block (k=10, fitted on 2025) | cross-validation |

Categorical values kept in Portuguese where they are official register text:
`situacao_sql` levels seen upstream are "Ativo Predial" (active, with
building) and "Ativo Territorial" (active, land-only).
"""
    (OUT_DIR / "data_dictionary.md").write_text(dic, encoding="utf-8")
    log("Wrote filter_report.md and data_dictionary.md")
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
