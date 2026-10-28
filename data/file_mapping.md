| **NAME** | **TKIO DATASET** | **OG FILE NAME** | **STANDALONE FILE NAME** |
|---|---|---|---|
| ECOSYS | NVIS_v2p1.value | NVIS_MVG_x_IBRA_CMZ_ECOSYS_CORRECTED.tif | ecosys.tif |
| COND | HCAS_v2p1.hcas_corrected | HCAS34_AHC_{EPOCH}_MASKED_TO_MVG_ECOSYS_CORRECTED.tif | cond_{EPOCH}.tif |
| CONN_NAT | HCAS_v2p1.conn | HCAS34_normconn_AHC_{EPOCH}.tif | conn_nat_{EPOCH}.tif |
| CONN_MAX | HCAS_v2p1.conn_max | HCAS34_maxconn_AHC_{EPOCH}.tif | conn_max_{EPOCH}.tif |
| IBRA | IBRA7_v2.subregions | IBRA7_subregions.zip | ibra7.tif |
| CMZ | CMZ_v2.code | Conservation_Management_Zones.zip | cmz.tif |

`{EPOCH}` is `2020_2022` or `2023_2025`; `lookup_tables/HCAS_epoch_lookup.csv` maps the payload's `project_start_date` to one of them.

`ibra7.tif` and `cmz.tif` must be on the same 90 m grid as `ecosys.tif`. `ibra7.tif` values = `INDEX` in `lookup_tables/ibra7_lookup.csv`; `cmz.tif` values = `CMZ_ID`.
