#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_fiscal_block_centroids.py — fiscal-block centroids from GeoSampa
====================================================================
Merges the three WFS pages of the `geoportal:quadra_fiscal` layer (GeoSampa,
São Paulo City Hall geoportal, EPSG:31983 / SIRGAS 2000 UTM 23S) and derives
the two geo lookup tables used to georeference ITBI records:

  data/fiscal_block_centroids.csv   one row per sub-block (64,223), with the
                                    block polygon centroid in UTM and WGS84
  data/fiscal_block_lookup.csv      one row per setor+quadra key (47,891) —
                                    the 6-digit prefix of the SQL cadastral
                                    number — with the area-weighted centroid
                                    of its sub-blocks

Block-type priority: when a setor+quadra key mixes block types, only the
sub-blocks of the highest-priority type are kept, in the order
F (fiscal) > E (special) > M (municipal) > R (river/reservoir).

Input: data/raw/quadra_fiscal_p1.gpkg ... p3.gpkg (WFS pages of 30,000
features ordered by cd_identificador, downloaded manually from GeoSampa —
see DATA_NOTICE.md). The GPKG is read with sqlite3 + shapely, so no GDAL
stack is required.

Run from the project root:  python pipeline/01_fiscal_block_centroids.py
"""

from __future__ import annotations

import csv
import glob
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkb as shp_wkb

ROOT = Path(__file__).resolve().parents[1]
SRC = sorted(glob.glob(str(ROOT / "data" / "raw" / "quadra_fiscal_p*.gpkg")))
OUT_CENTROIDS = ROOT / "data" / "fiscal_block_centroids.csv"
OUT_LOOKUP = ROOT / "data" / "fiscal_block_lookup.csv"

TYPE_PRIORITY = {"F": 0, "E": 1, "M": 2, "R": 3}


def gpkg_geom_to_wkb(blob: bytes) -> bytes:
    """Strip the GeoPackageBinary header and return plain WKB.

    Header: magic 'GP', version, flags, srs_id (int32), then an optional
    envelope of 0–4 float64 pairs depending on bits 1–3 of the flags byte.
    """
    assert blob[:2] == b"GP", "not a GeoPackageBinary blob"
    flags = blob[3]
    env_type = (flags >> 1) & 0x07
    env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env_type]
    return blob[8 + env_len:]


def main() -> None:
    if not SRC:
        sys.exit("No data/raw/quadra_fiscal_p*.gpkg found — see DATA_NOTICE.md.")
    to_wgs = Transformer.from_crs("EPSG:31983", "EPSG:4326", always_xy=True)

    rows, seen = [], set()
    for f in SRC:
        con = sqlite3.connect(f)
        cur = con.cursor()
        q = ("select cd_identificador, cd_setor_fiscal, cd_quadra_fiscal, "
             "cd_subquadra_fiscal, cd_tipo_quadra, tx_tipo_quadra, "
             "cd_situacao_quadra, cd_situacao_setor, qt_area_quadra_fiscal, "
             "ge_poligono from quadra_fiscal")
        n = 0
        for (cid, setor, quadra, sub, tipo, tipo_tx, sit_q, sit_s,
             area, blob) in cur.execute(q):
            if cid in seen:  # WFS pages may overlap at the boundaries
                continue
            seen.add(cid)
            geom = shp_wkb.loads(gpkg_geom_to_wkb(blob))
            c = geom.centroid
            # the centroid of a concave polygon may fall outside it
            if not geom.contains(c):
                c = geom.representative_point()
            lon, lat = to_wgs.transform(c.x, c.y)
            rows.append(dict(
                cd_identificador=cid, setor=setor, quadra=quadra,
                subquadra=sub, setor_quadra=f"{setor}{quadra}",
                tipo_quadra=tipo, tipo_quadra_desc=tipo_tx,
                situacao_quadra=sit_q, situacao_setor=sit_s,
                area_m2=round(area or geom.area, 3),
                area_geom_m2=round(geom.area, 3),
                x_utm=round(c.x, 2), y_utm=round(c.y, 2),
                lon=round(lon, 7), lat=round(lat, 7),
                n_partes=len(getattr(geom, "geoms", [geom])),
            ))
            n += 1
        con.close()
        print(f"{f}: {n} features read", file=sys.stderr)

    with open(OUT_CENTROIDS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} sub-blocks -> {OUT_CENTROIDS}")

    # ---- setor+quadra lookup (area-weighted centroid) ----------------------
    c = pd.DataFrame(rows)
    c["prio"] = c["tipo_quadra"].map(TYPE_PRIORITY)
    keep = c[c["prio"] == c.groupby("setor_quadra")["prio"].transform("min")]

    def agg(d: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "setor": d["setor"].iloc[0],
            "quadra": d["quadra"].iloc[0],
            "tipo_quadra": d["tipo_quadra"].iloc[0],
            "n_subquadras": len(d),
            "area_total_m2": round(d["area_m2"].sum(), 3),
            "x_utm": round(np.average(d["x_utm"], weights=d["area_m2"]), 2),
            "y_utm": round(np.average(d["y_utm"], weights=d["area_m2"]), 2),
        })

    lk = keep.groupby("setor_quadra").apply(agg, include_groups=False).reset_index()
    lon, lat = to_wgs.transform(lk["x_utm"].to_numpy(), lk["y_utm"].to_numpy())
    lk["lon"], lk["lat"] = np.round(lon, 7), np.round(lat, 7)
    lk.to_csv(OUT_LOOKUP, index=False)
    print(f"{len(lk)} setor+quadra keys -> {OUT_LOOKUP}")


if __name__ == "__main__":
    main()
