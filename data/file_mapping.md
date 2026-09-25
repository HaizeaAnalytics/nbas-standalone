| **NAME** | **TKIO DATASET** | **OG FILE NAME** | **STANDALONE FILE NAME** |
|---|---|---|---|
| ECOSYS | NVIS2024_v2.value_v2 | NVIS_MVG_x_IBRA_CMZ.tif | ecosys.tif |
| COND | HCAS.hcas | HCAS31_AHC_2020_2022.tif | cond.tif |
| CONN_NAT | HCAS.conn | HCAS31_normconn_AHC_2020_2022_100m.tif | conn_nat.tif |
| CONN_MAX | HCAS.conn_max | hcas_conn_max.tif | conn_max.tif |
| IBRA | IBRA7.subregions | IBRA7_subregions_states.zip | ibra7.tif |
| CMZ | CMZ.code | Conservation_Management_Zones.zip | cmz.tif |

`ibra7.tif` and `cmz.tif` must be on the same 90 m grid as `ecosys.tif`. `ibra7.tif` values = `INDEX` in `lookup_tables/ibra7_lookup.csv`; `cmz.tif` values = `CMZ_ID`.

Full datasets available in docker image haizeaanalytics/nbas-terrakio-corrected:latest
