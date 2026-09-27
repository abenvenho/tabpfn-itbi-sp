# Data dictionary — itbi_sp_2025_train.csv / itbi_sp_2026_test.csv / itbi_sp_2025_level_a.csv

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
