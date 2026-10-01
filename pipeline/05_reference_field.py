#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_reference_field.py — the free-text "Referência" field of the ITBI forms
=========================================================================
The ITBI workbooks carry a free-text field, *Referência*, that the cleaning
pipeline reads (``referencia``) but does not write to the cleaned bases. It
is filled on about 40 % of the forms, as the taxpayer or the notary typed it:
mostly the name of the building or development ("EDIFICIO THE PARK",
"CJ HAB SAFIRA IV", "LIVING HEREDITA"), sometimes a tower, a landmark, a
registry note or the city name.

This step re-runs the documented filter chain of ``02_itbi_cleaning.py``
(same functions, same order, same deduplication) so that each cleaned row
gets the *Referência* of the very form it was built from, and writes one
side file keyed like the deduplication: ``(sql, data_transacao,
valor_transacao)``. The cleaned bases themselves are not touched, so every
earlier result stays byte-identical.

Used only by the text variant of TabPFN-3.5 (``--text``), never by the main
analysis. Output: ``data/itbi_sp_reference_field.csv.gz``.

Run:  python pipeline/05_reference_field.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "itbi_sp_reference_field.csv.gz"
KEY = ["sql", "data_transacao", "valor_transacao"]
BASES = {"2025_train": ROOT / "data" / "itbi_sp_2025_train.csv.gz",
         "2026_test": ROOT / "data" / "itbi_sp_2026_test.csv"}


def load_cleaning_module():
    spec = importlib.util.spec_from_file_location("itbi_cleaning",
                                                  ROOT / "pipeline" / "02_itbi_cleaning.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clean_text(s: pd.Series) -> pd.Series:
    txt = s.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    return txt.where(txt != "", pd.NA)


def main() -> None:
    m = load_cleaning_module()
    lookup = pd.read_csv(m.LOOKUP_CSV, dtype={"setor_quadra": str})
    lookup["setor_quadra"] = lookup["setor_quadra"].str.zfill(6)
    lookup = lookup[["setor_quadra", "x_utm", "y_utm", "lon", "lat"]]
    stations = m.load_stations()

    raw25 = m.to_canonical_types(m.read_monthly_sheets(m.XLSX_2025))
    raw26 = m.to_canonical_types(m.read_monthly_sheets(m.XLSX_2026))
    df25, fences, _ = m.apply_filters(raw25, m.FilterReport("2025"), None, lookup, stations)
    df26, _, _ = m.apply_filters(raw26, m.FilterReport("2026"), fences, lookup, stations)

    ref = pd.concat([df25, df26])[KEY + ["referencia"]].copy()
    ref["referencia"] = clean_text(ref["referencia"])
    if ref.duplicated(KEY).any():
        raise ValueError("deduplication key is not unique after the filter chain")

    lines = []
    for name, path in BASES.items():
        base = pd.read_csv(path, usecols=KEY, dtype={"sql": str}, parse_dates=["data_transacao"])
        j = base.merge(ref, on=KEY, how="left", indicator=True)
        missing = int((j["_merge"] != "both").sum())
        if missing:
            raise ValueError(f"{name}: {missing} cleaned rows without a source form")
        lines.append(f"{name}: {len(base):,} rows, Referência filled in "
                     f"{j['referencia'].notna().mean():.1%}, "
                     f"{j['referencia'].nunique():,} distinct values")

    keep = pd.concat([pd.read_csv(p, usecols=KEY, dtype={"sql": str},
                                  parse_dates=["data_transacao"]) for p in BASES.values()])
    out = keep.merge(ref, on=KEY, how="left")
    out["data_transacao"] = out["data_transacao"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUT, index=False, compression="gzip")
    print("\n".join(lines))
    print(f"written {OUT.relative_to(ROOT)} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
