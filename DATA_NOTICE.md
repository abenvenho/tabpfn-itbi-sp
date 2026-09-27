# Data notice

All datasets in this repository are public data published by Brazilian
government bodies. They were downloaded manually by the author on
**2026-09-27** (no scraping; the pipeline performs no downloads at runtime).

**A note on file names and language:** the raw files under `data/raw/` keep
their **original Portuguese names, exactly as downloaded from the official
São Paulo City Hall (Prefeitura de São Paulo) sites**, to preserve
provenance — e.g. "GUIAS DE ITBI PAGAS 2025" means "ITBI forms paid in
2025". Likewise, the columns of the derived CSVs in `data/` follow the
Portuguese snake_case canon of the official source layout. Every Portuguese
term is translated in the README glossary and in `data/data_dictionary.md`.
English-labelled convenience copies are provided (see the last section).

## 1. ITBI transaction forms (São Paulo Municipal Finance Department)

- **Files:** `data/raw/GUIAS DE ITBI PAGAS 2025 XLS.xlsx`,
  `data/raw/GUIAS DE ITBI PAGAS 2026 XLS.xlsx`
  ("guias de ITBI pagas" = paid ITBI forms)
- **Source:** Secretaria Municipal da Fazenda de São Paulo (São Paulo
  Municipal Finance Department) — monthly publication of paid ITBI forms
  (ITBI = *Imposto sobre Transmissão de Bens Imóveis*, the municipal
  real-estate transfer tax), https://www.prefeitura.sp.gov.br/
  (Fazenda → ITBI → "Guias de ITBI pagas").
- **Content:** one row per paid form: cadastral key (SQL = *Setor, Quadra,
  Lote* — sector, block, lot), address, deed nature, declared price,
  transaction date, reference assessed value (VVR — *Valor Venal de
  Referência*), transmitted share, financing fields, and IPTU attributes
  (areas, use, standard, completion year; IPTU = *Imposto Predial e
  Territorial Urbano*, the annual municipal property tax). The workbooks
  include the publisher's own documentation sheets, kept intact: LEGENDA
  (legend), EXPLICAÇÕES (explanations), Tabela de USOS (use-code table)
  and Tabela de PADRÕES (standard-code table).
- **Terms:** public data released by the municipality under Brazilian
  open-government rules (LAI — *Lei de Acesso à Informação*, Law
  12,527/2011, the Access to Information Act).
- **Note on personal data:** forms identify properties, not persons; no
  buyer or seller names are present.

## 2. Fiscal blocks (GeoSampa)

- **Files:** `data/raw/quadra_fiscal_p1.gpkg` … `p3.gpkg`
  ("quadra fiscal" = fiscal block; WFS pages of 30,000 features ordered by
  `cd_identificador`)
- **Source:** GeoSampa — *Mapa Digital da Cidade de São Paulo* (Digital Map
  of the City of São Paulo), WFS layer `geoportal:quadra_fiscal`,
  EPSG:31983 (SIRGAS 2000 / UTM 23S),
  https://geosampa.prefeitura.sp.gov.br/
- **Terms:** open geospatial data of the São Paulo City Hall.
- **Derived:** `data/fiscal_block_centroids.csv` and
  `data/fiscal_block_lookup.csv` via `pipeline/01_fiscal_block_centroids.py`.

## 3. Subway and commuter-rail stations (GeoSampa)

- **Files:** `data/raw/estacao_metro.geojson` ("estação de metrô" = subway
  station; 94 Metrô/ViaQuatro stations), `data/raw/estacao_trem.geojson`
  ("estação de trem" = train station; 109 CPTM/ViaMobilidade commuter-rail
  stations)
- **Source:** GeoSampa station layers, point geometries in EPSG:31983.

## Derived datasets

`data/itbi_sp_2025_train.csv`, `data/itbi_sp_2026_test.csv` and
`data/itbi_sp_2025_level_a.csv` are produced from the sources above by
`pipeline/02_itbi_cleaning.py`; every exclusion is documented with record
counts in `data/filter_report.md`. Column semantics (with English
translations of all Portuguese terms) are in `data/data_dictionary.md`.

## English-labelled copies — full disclosure

The files under `data/english/` (`itbi_sp_2025_train_en.csv`,
`itbi_sp_2026_test_en.csv`, `itbi_sp_2025_level_a_en.csv`) were **generated
by the authors to make analysis easier for English speakers. They are NOT
the original datasets.** They contain exactly the same records, in the same
order, as their Portuguese-headed counterparts in `data/`; only the column
headers are translated (mapping in `data/english/column_mapping.csv`,
generator in `pipeline/04_english_labels.py`). Columns whose *values*
remain Portuguese free text from the official registers (IPTU use/standard
descriptions, neighbourhood names, financing categories) carry a `_pt`
suffix. For any audit or reconciliation against the official publication,
use the Portuguese-headed files and the raw workbooks.
