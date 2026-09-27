#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_english_labels.py — English-labelled convenience copies
===========================================================
Generates copies of the three modelling datasets with all column headers
translated to English, for the convenience of English-speaking analysts:

  data/english/itbi_sp_2025_train_en.csv
  data/english/itbi_sp_2026_test_en.csv
  data/english/itbi_sp_2025_level_a_en.csv
  data/english/column_mapping.csv     (original -> English header mapping)

FULL DISCLOSURE: these files are NOT the originals. The authoritative
datasets are the Portuguese-headed CSVs in data/ (whose column names follow
the official source layout) and, upstream, the raw workbooks in data/raw/
exactly as published by the São Paulo City Hall. The English copies hold the
same records in the same order; only the headers are renamed. Columns whose
*values* remain Portuguese free text (official IPTU descriptions,
neighbourhood names, financing categories) carry a `_pt` suffix.

Run from the project root:  python pipeline/04_english_labels.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "english"

# Original (official Portuguese canon) -> English header
COLUMN_EN = {
    # identifiers / traceability
    "sql": "cadastral_key_sql",          # SQL = Setor, Quadra, Lote (sector, block, lot)
    "logradouro": "street_name",
    "numero": "street_number",
    "complemento": "address_unit",
    "grupo_edificio": "building_group",
    "mes_referencia": "payment_month_sheet",
    # features
    "tipo_imovel": "property_type",
    "uso_iptu": "iptu_use_code",
    "descricao_uso": "iptu_use_description_pt",
    "padrao_iptu": "iptu_standard_code",
    "padrao_tipo": "building_class_digit",
    "padrao_nivel": "finish_grade",
    "descricao_padrao": "iptu_standard_description_pt",
    "idade": "age_years",
    "area_construida_m2": "built_area_m2",
    "area_terreno_m2": "lot_area_m2",
    "fracao_ideal": "undivided_share",
    "testada_m": "frontage_m",
    "x_utm": "x_utm",
    "y_utm": "y_utm",
    "lon": "lon",
    "lat": "lat",
    "bairro": "neighborhood_pt",
    "cep": "postal_code",
    "setor": "fiscal_sector",
    "setor_quadra": "sector_block_key",
    "data_transacao": "transaction_date",
    "mes_idx": "month_index",
    "dist_estacao_m": "dist_station_m",
    # target and response components
    "valor_transacao": "declared_price_brl",
    "area_ref": "reference_area_m2",
    "ln_vu": "ln_unit_price",
    # diagnostics only (never model features)
    "valor_venal_referencia": "assessed_ref_value_brl",
    "razao_vvr": "price_to_assessed_ratio",
    "financiado": "is_financed",
    "tipo_financiamento": "financing_type_pt",
    "valor_financiado": "financed_amount_brl",
    # cross-validation
    "bloco": "spatial_block",
}

FILES = {
    "itbi_sp_2025_train.csv": "itbi_sp_2025_train_en.csv",
    "itbi_sp_2026_test.csv": "itbi_sp_2026_test_en.csv",
    "itbi_sp_2025_level_a.csv": "itbi_sp_2025_level_a_en.csv",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in FILES.items():
        src = ROOT / "data" / src_name
        df = pd.read_csv(src, dtype={"sql": str, "cep": str, "setor": str,
                                     "setor_quadra": str, "padrao_iptu": str,
                                     "padrao_tipo": str})
        missing = [c for c in df.columns if c not in COLUMN_EN]
        if missing:
            raise ValueError(f"{src_name}: unmapped columns {missing}")
        df.rename(columns=COLUMN_EN).to_csv(OUT_DIR / dst_name, index=False)
        print(f"{src_name} -> english/{dst_name} ({len(df):,} rows)")

    with open(OUT_DIR / "column_mapping.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["original_column", "english_column"])
        w.writerows(COLUMN_EN.items())
    print(f"Wrote english/column_mapping.csv ({len(COLUMN_EN)} columns)")


if __name__ == "__main__":
    main()
