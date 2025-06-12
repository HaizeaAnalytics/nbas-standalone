import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import geopandas as gpd
from copy import deepcopy
from shapely import unary_union
from shapely.geometry import box, shape
import xarray as xr
from pydantic import BaseModel
import rasterio
import rioxarray
import connectivity.connectivity as connectivity
import json
import os
import datetime
import argparse

cmz_lookup = pd.read_csv("lookup_tables/NVIS_MVG_x_IBRA_CMZ_key.csv")
ibra_code_lookup = pd.read_csv("lookup_tables/ibra7_lookup.csv")
ibra_name_lookup = pd.read_csv("lookup_tables/CMZ_IBRA_states.csv")
PADDING = 160

class PlanrPayload(BaseModel):
    areas: dict
    options: dict
    crs: str

def nbas(planr_payload):
    areas_gdf = gpd.GeoDataFrame.from_features(planr_payload['areas'])
    # NOTE: every activity area must have a unique 'id' value
    assert len(np.unique(areas_gdf['id'].values)) == len(areas_gdf)
    areas_gdf = areas_gdf.set_crs(planr_payload['crs'])
    areas_gdf = areas_gdf.to_crs(3577)
    # NOTE: this re-establishes planr_payload['areas'] as pointer to areas_gdf
    planr_payload['areas'] = areas_gdf
    debug = planr_payload['options'].get('debug', False)
    # validate values in areas_gdf fields
    assert np.all((0 <= areas_gdf['cond_b4_lo']) & (areas_gdf['cond_b4_lo'] <= 1))
    assert np.all((0 <= areas_gdf['cond_b4_hi']) & (areas_gdf['cond_b4_hi'] <= 1))
    assert np.all((0 <= areas_gdf['cond_af_lo']) & (areas_gdf['cond_af_lo'] <= 1))
    assert np.all((0 <= areas_gdf['cond_af_hi']) & (areas_gdf['cond_af_hi'] <= 1))
    ecosys_data = get_mvg_data(areas_gdf)
    assert np.all((areas_gdf['ecosys'] > 0) & (areas_gdf['ecosys'] <= 690))
    try:
        (cond_tbl,
         conn_tbl,
         cons_sig_tbl,
         persist_tbl,
         debug_rasters) = run_NBAS_first_method(planr_payload=planr_payload,
                                                      plot_it=False,
                                                      mock_conn=planr_payload['options']['mock_conn'],
                                                      debug=debug,
                                                     )
    except Exception as e:
        raise e

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if debug:
        debug_rasters.to_netcdf(path=f"debug/{ts}-rasters.nc", mode='w', engine='scipy')
        print(f"Debug rasters written to debug/{ts}-rasters.nc")

    results = {
        "cond_tbl": cond_tbl.fillna("NaN").to_dict(),
        "conn_tbl": conn_tbl.fillna("NaN").to_dict(),
        "cons_sig_tbl": cons_sig_tbl.fillna("NaN").to_dict(),
        "persist_tbl": persist_tbl.fillna("NaN").to_dict(),
        "area_data": ecosys_data,
        "version": "1.0.0-standalone"
    }

    # write to file with name using timestamp
    file_name = f"results/{ts}-results.json"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results written to {file_name}")


def get_poly_from_tif(tif_path, geom, padding=0):
    if padding != 0:
        geom = box(*geom.bounds).buffer(padding*90)
    ds = rioxarray.open_rasterio(tif_path, chunks={'x': 512, 'y': 512}, masked=True)
    ds.rio.write_crs("epsg:3577", inplace=True)
    ds = ds.rio.clip([geom], drop=True, from_disk=True, all_touched=False)
    ds = ds.squeeze()
    return ds


def get_ibra_and_cmz_from_poly(poly):
    ibra = get_poly_from_tif("data/ibra7.tif", poly)
    cmz = get_poly_from_tif("data/cmz.tif", poly)
    ibra = get_majority(ibra.values) - 1
    cmz = get_majority(cmz.values)
    return ibra, cmz
    

def get_majority(values):
    # get most common value excluding NaNs
    values = values[~np.isnan(values)]
    values, counts = np.unique(values, return_counts=True)
    ind = np.argmax(counts)

    return values[ind]


def feature_collection_to_multipolygon(feature_collection):
    # Extract coordinates from all features
    polygons = [shape(feature['geometry']) for feature in feature_collection['features']]

    poly = unary_union(polygons)

    #Create the new Feature
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": poly.__geo_interface__
        }
    return feature


def get_rasters(areas_gdf):
    ecosys = get_poly_from_tif("data/ecosys.tif", areas_gdf.geometry[0], padding=PADDING)
    cond = get_poly_from_tif("data/cond.tif", areas_gdf.geometry[0], padding=PADDING)
    conn_nat = get_poly_from_tif("data/conn_nat.tif", areas_gdf.geometry[0], padding=PADDING)
    conn_max = get_poly_from_tif("data/conn_max.tif", areas_gdf.geometry[0], padding=PADDING)

    return ecosys, cond, conn_nat, conn_max


def get_mvg_data(areas_gdf):
    """
    NOTE: this function both alters its input and returns a separate output!
    """
    results = {}
    for i, row in areas_gdf.iterrows():
        ibra, cmz = get_ibra_and_cmz_from_poly(row['geometry'])
        if 'ecosys' not in row or np.isnan(row['ecosys']):
            try:
                ecosys = int(cmz_lookup.loc[(cmz_lookup['mvg_num'] ==row["major_veg_group"]) & (cmz_lookup['cmz_num'] == cmz), 'eco_num'].values[0])
            except IndexError:
                raise Exception(f"MVG {int(row['major_veg_group'])} not yet supported")
            areas_gdf.at[i, 'ecosys'] = ecosys
        else:
            ecosys = int(row['ecosys'])
        try:
            mvg, mvg_name, cmz_name = cmz_lookup.loc[(cmz_lookup['eco_num'] == ecosys), ['mvg_num', 'mvg_name', 'cmz_name']].values[0]
        except IndexError:
            raise Exception(f"Ecosys number {ecosys} not yet supported")
        ibra_code = ibra_code_lookup.loc[ibra_code_lookup['INDEX'] == ibra, 'SUB_CODE_7'].values[0]
        ibra_name = ibra_name_lookup.loc[(ibra_name_lookup['SUB_CODE_7'] == ibra_code), 'SUB_NAME_7'].values[0]
        cmz_name = ibra_name_lookup.loc[ibra_name_lookup['CMZ_ID'] == cmz, 'CM_ZONE'].values[0]
        results[row["id"]] = {"ecosys": ecosys, "IBRA_subregion": ibra_code, "IBRA_subregion_name": ibra_name, "major_veg_group": int(mvg), "major_veg_group_name": mvg_name, "CMZ": int(cmz), "CMZ_name": cmz_name}
    return results


def run_NBAS_first_method(planr_payload,
                                mask_ecosys_b4_cond_calc=False,
                                zero_range_pctile_val=0.5,
                                within_ecosys_conn=False,
                                SAR_c=1,
                                SAR_z=0.25,
                                rescale_Δ_persist=True,
                                rescale_stat='median',
                                mock_conn=False,
                                plot_it=False,
                                debug=False,
                               ):
    '''
    Interim function, for first method, which runs
    the run_NBAS_multiple_areas function twice, once using the
    before- and after-activity condition ranges provided by PLANR,
    and once using the same before-activity condition range but setting the
    after-activity range to 0.0 (to model the biodiversity importance of
    current habitat, which will be tacitly protected if a revegetation
    project is carried out). Returns all results organized into the four output
    tables specified for the first method's protocol document.
    '''
    debug_rasters = []
    # create overall results data structure, which will hold both the results
    # projecting to the expected future conditions and the results that would
    # be expected if all current condition were reduced to zero
    overall_res = {}
    # run using both the before- and after-activity condition ranges from PLANR
    multiple_areas = {}
    areas_gdf = planr_payload['areas']
    
    curr_vs_future = run_NBAS_multiple_areas(planr_payload,
                              mask_ecosys_b4_cond_calc=mask_ecosys_b4_cond_calc,
                              zero_range_pctile_val=zero_range_pctile_val,
                              within_ecosys_conn=within_ecosys_conn,
                              SAR_c=SAR_c,
                              SAR_z=SAR_z,
                              rescale_Δ_persist=rescale_Δ_persist,
                              rescale_stat=rescale_stat,
                              mock_conn=mock_conn,
                              id = "curr_vs_future",
                              debug=debug
                             )
    changed_payload = deepcopy(planr_payload)
    changed_areas_gdf = changed_payload['areas']
    changed_areas_gdf = changed_areas_gdf.reset_index()
    changed_areas_gdf['cond_af_lo'] = [0.0]*len(changed_areas_gdf)
    changed_areas_gdf['cond_af_hi'] = [0.0]*len(changed_areas_gdf)
    changed_areas_gdf['state_af'] = [6]*len(changed_areas_gdf)
    changed_payload['areas'] = changed_areas_gdf
    curr_vs_zero = run_NBAS_multiple_areas(changed_payload,
                                    mask_ecosys_b4_cond_calc=mask_ecosys_b4_cond_calc,
                                    zero_range_pctile_val=zero_range_pctile_val,
                                    within_ecosys_conn=within_ecosys_conn,
                                    SAR_c=SAR_c,
                                    SAR_z=SAR_z,
                                    rescale_Δ_persist=rescale_Δ_persist,
                                    rescale_stat=rescale_stat,
                                    mock_conn=mock_conn,
                                    id = "curr_vs_zero",
                                    debug=debug
                                   )

    multiple_areas['curr_vs_future'] = curr_vs_future
    multiple_areas['curr_vs_zero'] = curr_vs_zero

    for id in multiple_areas:
        res, rasters = multiple_areas[id]
        debug_rasters.append(rasters)
        overall_res[id] = res

    if debug:
        debug_rasters = xr.merge(debug_rasters)

    # compile set of output tables as stipulated within the protocol, including:
    # 1: condition table
    cond_tbl_dict = {'area_ha': [],
                     'starting_cond': [],
                     'target_cond': [],
                     'change_cond': [],
                    }
    res = overall_res['curr_vs_future']['cond']
    # NOTE: re-expressing areas in hectares instead of meters
    cond_tbl_dict['area_ha'].append(np.sum(areas_gdf.area/100**2))
    cond_tbl_dict['starting_cond'].append(res[0])
    cond_tbl_dict['target_cond'].append(res[1])
    cond_tbl_dict['change_cond'].append(res[2])
    cond_tbl = pd.DataFrame.from_dict(cond_tbl_dict)

    # 2: connectivity table
    conn_tbl_dict = {'area_ha': [],
                     'starting_conn': [],
                     'target_conn': [],
                     'change_conn': [],
                     'change_conn_ecosys_area_ha': [],
                    }
    res = overall_res['curr_vs_future']['conn']
    # NOTE: re-expressing areas in hectares instead of meters
    conn_tbl_dict['area_ha'].append(np.sum(areas_gdf.area/100**2))
    conn_tbl_dict['starting_conn'].append(res[0])
    conn_tbl_dict['target_conn'].append(res[1])
    conn_tbl_dict['change_conn'].append(res[2])
    conn_tbl_dict['change_conn_ecosys_area_ha'].append(res[3])
    conn_tbl = pd.DataFrame.from_dict(conn_tbl_dict)

    # 3: conservation significance table
    cons_sig_tbl_dict = {'area_ha': [],
                         'cons_sig': [],
                        }
    res = overall_res['curr_vs_future']['cons_sig']
    # NOTE: re-expressing areas in hectares instead of meters
    cons_sig_tbl_dict['area_ha'].append(np.sum(areas_gdf.area/100**2))
    cons_sig_values = []
    # NOTE: significance reports the range of values as a tuple of (min, max);
    #       for activity area-specific runs this can simply be reduced to
    #       the point value indicated by those two, identical values
    cons_sig_tbl_dict['cons_sig'].append(res)
    cons_sig_tbl = pd.DataFrame.from_dict(cons_sig_tbl_dict)

    # 4: biodiversity persistence table 
    persist_tbl_dict = {'area_ha': [],
                        'starting_persist_contrib': [],
                        'target_persist_contrib': [],
                        'change_persist_contrib': [],
                        'starting_persist_contrib_perha': [],
                        'target_persist_contrib_perha': [],
                        'change_persist_contrib_perha': [],
                       }
    future_Δ_persist = overall_res['curr_vs_future']['persist']
    # NOTE: multiply zero_Δ_persist by -1 to express as change from 0.0 to
    #       current (because it was modeled as change from curent to 0.0)
    zero_Δ_persist = -1 * overall_res['curr_vs_zero']['persist']
    # NOTE: re-expressing areas in hectares instead of meters
    area_ha_tot = np.sum(areas_gdf.area/100**2)
    persist_tbl_dict['area_ha'].append(area_ha_tot)
    persist_tbl_dict['starting_persist_contrib'].append(zero_Δ_persist)
    persist_tbl_dict['target_persist_contrib'].append(zero_Δ_persist + future_Δ_persist)
    persist_tbl_dict['change_persist_contrib'].append(future_Δ_persist)
    persist_tbl_dict['starting_persist_contrib_perha'].append(zero_Δ_persist/area_ha_tot)
    persist_tbl_dict['target_persist_contrib_perha'].append((zero_Δ_persist + future_Δ_persist)/area_ha_tot)
    persist_tbl_dict['change_persist_contrib_perha'].append(future_Δ_persist/area_ha_tot)
    persist_tbl = pd.DataFrame.from_dict(persist_tbl_dict)

    return cond_tbl, conn_tbl, cons_sig_tbl, persist_tbl, debug_rasters


def run_NBAS_multiple_areas(planr_payload,
                                  mask_ecosys_b4_cond_calc=False,
                                  zero_range_pctile_val=0.5,
                                  within_ecosys_conn=False,
                                  SAR_c=1,
                                  SAR_z=0.25,
                                  rescale_Δ_persist=True,
                                  rescale_stat='median',
                                  mock_conn=True,
                                  id = None,
                                  debug=False,
                                 ):
    '''
    run NBAS for an entire project's set of activity areas, returning a dict
    of results for both area-by-area runs and an overall run for the conjoint
    of all areas
    '''
    # lookup table containing nationwide ecosystem accounting and conservation
    # significance values
    ecosys_acct_df = pd.read_csv("./lookup_tables/cons_signif_results_epoch2020_2022.csv")
    # read activity areas
    areas_gdf = planr_payload['areas']
    # make sure index is just incrementing integers starting at zero
    areas_gdf = areas_gdf.reset_index()
    # make sure all ecosystem types in the areas_gdf have rows in the ecosys
    # account table!
    for ecosys_num in areas_gdf['ecosys'].values:
        if ecosys_num not in ecosys_acct_df['ecosys'].values:
            raise Exception(str(f"Ecosys number {int(ecosys_num)} not yet supported"))
        assert ecosys_num in ecosys_acct_df['ecosys'].values, ('Ecosystem '
                f'{ecosys_num} is not the ecosystem account table!')

    # get clipped rasters of ecosystem type, HCAS condition, and nationwide
    # connecitivity results and their max potential raw values, from Terrakio
    # NOTE: the national connectivity raster (already normalized!)
    #       is needed to account for differences between the local values that
    #       contributed to the calculation of per-hectare expected change
    #       and conservation significance scores, on the one hand, and local
    #       values in the connectivity rasters that are calculated using new
    #       condition maps updated with starting condition values
    #       received from  PLANR, on the other
    (ecosys,
     cond_raw,
     conn_nat,
     conn_max) = get_rasters(areas_gdf=areas_gdf)
    
    # copy raw condition rasters to new rasters, where the overridden before-
    # and after-action condition values will be burned in
    cond_b4_agg = cond_raw * 1
    cond_af_agg = cond_raw * 1

    # create a dict of rasters that will indicate just the percent overlap
    # values for all the pixels pertaining to an ecosystem type
    # (where ecosystem type numbers are the keys and the aggregated rasters are
    # the values; this is for later use in the summing of ecosystem-specific
    # change in connectivity-adjusted condition)
    ecosys_overlap_pcts = {}
    for ecosys_num in np.unique(areas_gdf['ecosys'].values):
        ecosys_overlap_pcts[ecosys_num] = cond_raw * 0

    # create a single raster where we'll burn in pixel overlap percents
    # for all activity areas
    tot_overlap_pct = cond_raw * 0
    tot_overlap_pct = tot_overlap_pct.rio.write_crs("epsg:3577")
    # set up a list of debug rasters, if needed
    if debug:
        debug_rasts = [xr.Dataset({f"{id}_cond_raw": cond_raw,
                                   f"{id}_ecosys": ecosys,
                                   f"{id}_conn_nat": conn_nat,
                                   f"{id}_conn_max": conn_max,
                                  })]
    # loop over rows in the transitions DataFrame
    single_areas = {}
    for i, row in areas_gdf.iterrows():
        area_result = run_NBAS_single_area(ecosys=ecosys,
                                                   cond_raw=cond_raw,
                                                   conn_nat=conn_nat,
                                                   conn_max=conn_max,
                                                   activity_area_row=row,
                                                   ecosys_acct_df=ecosys_acct_df,
                                                   mask_ecosys_b4_cond_calc=mask_ecosys_b4_cond_calc,
                                                   zero_range_pctile_val=zero_range_pctile_val,
                                                   within_ecosys_conn=within_ecosys_conn,
                                                   SAR_c=SAR_c,
                                                   SAR_z=SAR_z,
                                                   rescale_Δ_persist=rescale_Δ_persist,
                                                   rescale_stat=rescale_stat,
                                                   mock_conn=mock_conn,
                                                  )
        single_areas[i] = area_result

    for i, area in single_areas.items():
        (ecosys_num,
         overlap_pct,
         cond_b4,
         cond_af,
         conn_b4,
         conn_af,
         persist_b4,
         persist_af,
         Δ_persist,
         ecosys_conn_change_sum,
        ) = area

        # work those condition values into the aggregate-results raster
        cond_b4_agg = cond_b4.where(overlap_pct>0).combine_first(cond_b4_agg.where(overlap_pct==0))
        cond_af_agg = cond_af.where(overlap_pct>0).combine_first(cond_af_agg.where(overlap_pct==0))

        # add the overlap pct raster into the aggregate overlap pct
        # raster for this row's ecosystem type
        # NOTE: summation assumes that multiple polygons can overlap a single
        #       pixel (possible, even if not common) but will never overlap
        #       each other (entirely reasonable, and could always be enforced
        #       if need be)
        ecosys_overlap_pcts[ecosys_num] = (ecosys_overlap_pcts[ecosys_num] +
                                           overlap_pct)

        # and also add into the overlap overlap_pct raster
        tot_overlap_pct = tot_overlap_pct + overlap_pct

        # gather the area-specific debug rasters, if needed
        if debug:
            rasts = xr.Dataset({f"{id}_area{i}_overlap_pct": overlap_pct,
                                f"{id}_area{i}_cond_b4": cond_b4,
                                f"{id}_area{i}_cond_af": cond_af,
                                f"{id}_area{i}_conn_b4": conn_b4,
                                f"{id}_area{i}_conn_af": conn_af,
                               })
            debug_rasts.append(rasts)

    # calculate before and after connectivity rasters
    # NOTE: for optimization, can skip rerunning connectivity and persistence
    #       calculations if the payload only has a single activity area!
    #       only has a single row
    if len(areas_gdf) == 1:
        conn_b4_agg = conn_b4 * 1
        conn_af_agg = conn_af * 1
        Δ_persist_agg = Δ_persist
        all_ecosys_conn_change_sum = ecosys_conn_change_sum
    # ... but for 2+ activity areas connectivity and persistence
    #     must be calculated in aggregate
    else:
        (conn_b4_agg,
         conn_af_agg) = call_connectivity_model(cond_rast_b4=cond_b4_agg,
                                                cond_rast_af=cond_af_agg,
                                                conn_rast_max=conn_max,
                                                within_ecosys_conn=within_ecosys_conn,
                                                ecosys_rast=None,
                                                ecosys_num=None,
                                                mock=mock_conn,
                                               )

        # calculate SAR results by ecosystem, then normalize and sum
        Δ_persist_agg = 0
        all_ecosys_conn_change_sum = 0
        uniq_ecosys = np.unique(areas_gdf['ecosys'].values)
        for i, ecosys_num in enumerate(uniq_ecosys):
            # get the ecosystem-aggregated overlap_pct rast
            ecosys_overlap_pct = ecosys_overlap_pcts[ecosys_num]
            # calculate the expected change in persistence
            (persist_b4,
             persist_af,
             Δ_persist,
             ecosys_conn_change_sum,
            ) = calc_persistence_change(ecosys_rast=ecosys,
                                        ecosys_num=ecosys_num,
                                        ecosys_acct_df=ecosys_acct_df,
                                        overlap_pct_rast=ecosys_overlap_pct,
                                        cond_rast_b4=cond_b4_agg,
                                        cond_rast_af=cond_af_agg,
                                        cond_rast_raw=cond_raw,
                                        conn_rast_b4=conn_b4_agg,
                                        conn_rast_af=conn_af_agg,
                                        conn_rast_nat=conn_nat,
                                        c=SAR_c,
                                        z=SAR_z,
                                        rescale_Δ_persist=rescale_Δ_persist,
                                        rescale_stat=rescale_stat,
                                       )
            Δ_persist_agg += Δ_persist
            all_ecosys_conn_change_sum += ecosys_conn_change_sum

    # add in the debug rasters whose values were cumulated across all activity
    # areas, if needed
    if debug:
        debug_rasts.append(xr.Dataset({f"{id}_cond_b4_agg": cond_b4_agg,
                                       f"{id}_cond_af_agg": cond_af_agg,
                                       f"{id}_conn_b4_agg": conn_b4_agg,
                                       f"{id}_conn_af_agg": conn_af_agg,
                                       f"{id}_tot_overlap_pct": tot_overlap_pct,
                                      }))

        debug_xr = xr.merge(debug_rasts)
    else:
        debug_xr = None

    # get condition-change summary results
    cond_summary = summarize_cond_results(b4_cond=cond_b4_agg,
                                          af_cond=cond_af_agg,
                                          overlap_rast=tot_overlap_pct,
                                          raw_cond=cond_raw,
                                         )
    #  get connectivity-change summary results
    conn_summary = summarize_conn_results(b4_conn=conn_b4_agg,
                                          af_conn=conn_af_agg,
                                          overlap_rast=tot_overlap_pct,
                                         )
    # fold in the sum of ecosys connectivity change across all ecosystems
    # NOTE: can only be used to numerically recreate NBAS results for runs
    #       consisting of a single activity area
    conn_summary = tuple([*conn_summary] + [all_ecosys_conn_change_sum])

    # look up conservation significance of each area's ecosystem type
    cons_sig_summary = summarize_cons_sig_results(areas_gdf=areas_gdf,
                                                  ecosys_acct_df=ecosys_acct_df,
                                                  on='ecosys',
                                                 )

    # structure output and return
    output = {'cond': cond_summary,
              'conn': conn_summary,
              'cons_sig': cons_sig_summary,
              'persist': Δ_persist_agg,
             }
    return output, debug_xr


def run_NBAS_single_area(ecosys,
                               cond_raw,
                               conn_nat,
                               conn_max,
                               activity_area_row,
                               ecosys_acct_df,
                               mask_ecosys_b4_cond_calc=False,
                               zero_range_pctile_val=0.5,
                               within_ecosys_conn=False,
                               SAR_c=1.0,
                               SAR_z=0.25,
                               rescale_Δ_persist=True,
                               rescale_stat='median',
                               mock_conn=True,
                              ):
    '''
    run the NBAS algorithm for a single activity area's row from an activity
    areas GeoDataFrame
    '''
    # unpack transition data
    ecosys_num = activity_area_row['ecosys']
    cond_b4_lo = activity_area_row['cond_b4_lo']
    cond_b4_hi = activity_area_row['cond_b4_hi']
    cond_af_lo = activity_area_row['cond_af_lo']
    cond_af_hi = activity_area_row['cond_af_hi']
    # NOTE: states are not actually used, as their stipulation just implies
    #       that all valid (i.e., ecosystem-concordant) pixels within the
    #       activity area already in that state, but perhaps useful in future?
    b4_state = activity_area_row['state_b4']
    af_state = activity_area_row['state_af']

    # get the activity area's geometry
    geom = activity_area_row['geometry']

    # mask out invalid ecosystem pixels, if necessary
    # NOTE: PLANR will not allow ecosystem overrides, but it will
    #       allow polygons to contain 90%<=pct<=100% percent cover
    #       of the target ecosystem. The idea is that all pixels within the
    #       polygon are to be treated as actually being of the majority
    #       ecosystem (i.e., this implies that the ecosystem mapping we're
    #       displaying is incorrect). However, it is also possible that the
    #       mapping is correct and that an activity area inappropriately
    #       contains >1 ecosystem type. In that case, technically,
    #       we should deal with this by dropping those invalid pixels from
    #       analysis by using the same ecosystem map that PLANR used
    #       to determine the percent of ecosystem type to mask them.
    #       This is left as an optional step, but the default behavior is
    #       that this is not done.
    if mask_ecosys_b4_cond_calc:
        valid_cond = mask_rast1_to_rast2_by_val(cond_raw, ecosys, ecosys_num)
    else:
        valid_cond = cond_raw * 1 # NOTE: effectively deepcopies the raster

    # calculate raster of the percent of each pixel that is w/in aoi
    overlap_pct = calc_pixel_percent_within_polygon(rast=cond_raw,
                                                    poly=geom,
                                                   )

    # get before-management condition raster, as well as a raster re-expressing
    # those values as percentile locations within the specified
    # before-management condition range
    b4_cond_range = [cond_b4_lo, cond_b4_hi]
    cond_b4, cond_pctile = calc_b4_cond_rast(raw_cond_rast=cond_raw,
                                             b4_cond_range=b4_cond_range,
                                             overlap_pct_rast=overlap_pct,
                                             zero_range_pctile_val=zero_range_pctile_val,
                                            )
    # use those and the specified after-management condition range to calculate
    # a raster of expected post-management condition
    af_cond_range = [cond_af_lo, cond_af_hi]
    cond_af = calc_af_cond_rast(raw_cond_rast=cond_raw,
                                b4_pctile_rast=cond_pctile,
                                af_cond_range=af_cond_range,
                                overlap_pct_rast=overlap_pct,
                               )

    # calculate the before and after connectivity rasters
    # NOTE: not calculating within the ecosystem, ∴ ecosys_num=None
    conn_b4, conn_af = call_connectivity_model(cond_rast_b4=cond_b4,
                                               cond_rast_af=cond_af,
                                               conn_rast_max=conn_max,
                                               within_ecosys_conn=within_ecosys_conn,
                                               ecosys_rast=None,
                                               ecosys_num=None,
                                               mock=mock_conn,
                                              )
    # calculate the expected change in persistence
    (persist_b4,
     persist_af,
     Δ_persist,
     ecosys_conn_change_sum,
    ) = calc_persistence_change(ecosys_rast=ecosys,
                                ecosys_num=ecosys_num,
                                ecosys_acct_df=ecosys_acct_df,
                                overlap_pct_rast=overlap_pct,
                                cond_rast_b4=cond_b4,
                                cond_rast_af=cond_af,
                                cond_rast_raw=cond_raw,
                                conn_rast_b4=conn_b4,
                                conn_rast_af=conn_af,
                                conn_rast_nat=conn_nat,
                                c=SAR_c,
                                z=SAR_z,
                                rescale_Δ_persist=rescale_Δ_persist,
                                rescale_stat=rescale_stat,
                               )
    # return rasters & other results, for aggregation into project-level results
    return (ecosys_num,
            overlap_pct,
            cond_b4,
            cond_af,
            conn_b4,
            conn_af,
            persist_b4,
            persist_af,
            Δ_persist,
            ecosys_conn_change_sum,
           )


def geom_mask(geom, pixels, res, x0, y0):
    mask = rasterio.features.rasterize(shapes=[geom], out_shape=pixels.shape,
        dtype='uint8', transform=(res, 0, x0,0, -res, y0), all_touched=False)
    return mask


def calc_pixel_percent_within_polygon(rast, poly):
    '''
    Calculates a raster giving the percent of each pixel's area within a polygon.
    Does this by increasing raster's resolution by 10, burning the polygon into it,
    then downsampling by summing 10x10 blocks. Divides by 100 to get percentage.
    '''
    empty = np.ones((rast.shape[0]*10, rast.shape[1]*10), dtype='uint8')
    res, _ = rast.rio.resolution()
    empty = geom_mask(poly, empty, res/10, rast.x.min() - res/2, rast.y.max() + res/2)
    # Downsample by summing 10x10 blocks
    empty = empty.reshape(rast.shape[0], 10, rast.shape[1], 10).sum(axis=(1, 3)).astype(np.float16)
    # Divide by 100 to get percentage
    empty /= 100
    return rast * 0 + empty


def mask_rast1_to_rast2_by_val(rast1, rast2, val):
    '''
    mask out of rast1 pixels that are not equal to val in rast2
    '''
    valid = (rast2 == val)
    masked = rast1.where(valid == 1)
    return masked


def calc_cond_range_pctile_rast(cond_rast,
                                cond_range,
                                zero_range_pctile_val=0.5,
                                clip_0_1=True,
                               ):
    '''
    calculate the percentile position of each raster pixel's value
    vis-a-vis the given condition range
    '''
    assert cond_range[1]>=cond_range[0]
    # if the condition range is just really a point value
    # then return the desired pctile value
    if cond_range[1] - cond_range[0] == 0:
        pctile = 0*cond_rast+zero_range_pctile_val
    else:
        pctile = ((cond_rast - min(cond_range))/
                  (max(cond_range) - min(cond_range)))
    if clip_0_1:
        pctile = np.clip(pctile, a_min=0, a_max=1)
    return pctile


def calc_b4_cond_rast(raw_cond_rast,
                      b4_cond_range,
                      overlap_pct_rast,
                      zero_range_pctile_val=0.5,
                     ):
    '''
    given the raw condition raster, the before-management condition range, and
    the raster giving each pixel's percent overlap with an activity area
    polygon, calculate the before-management condition raster,
    updated to match the condition range specified, as well as the pixel pctile
    positions within the starting range (to be used to project the raster into
    its after-management condition range)
    '''
    # numerically clip all condition values within the activity area
    # to the given before-action condition range
    # (giving the user-updated before-action condition, before adjustment
    # for partial pixel-polygon overlap)
    cond_b4_unadjusted = raw_cond_rast.clip(*b4_cond_range,
                                            keep_attrs=True,
                                           )

    # combine those to generate an adjusted, before-action condition raster
    cond_b4 = (raw_cond_rast * (1 - overlap_pct_rast) +
               cond_b4_unadjusted * overlap_pct_rast)

    # calculate raster of the percentile score of each management pixel
    # vis-a-vis its current state's condition range
    cond_pctile = calc_cond_range_pctile_rast(cond_rast=cond_b4,
                                              cond_range=b4_cond_range,
                                              zero_range_pctile_val=zero_range_pctile_val,
                                             )
    return cond_b4, cond_pctile


def calc_af_cond_rast(raw_cond_rast,
                      b4_pctile_rast,
                      af_cond_range,
                      overlap_pct_rast,
                     ):
    '''
    calculates a raster of updated condition estimates by putting
    the unmasked (i.e., activity-affected) pixels of the old raster
    into their corresponding percentile positions in the new raster,
    and then discounting the change using a raster indicating the amount of
    each pixel that is within the activity area
    '''
    af_target = raw_cond_rast*0
    af_cond_arr = ((min(af_cond_range) +
                     b4_pctile_rast*(max(af_cond_range) - min(af_cond_range))))
    af_target.values = af_cond_arr
    # calculate after-action condition, including partial pixel overlap
    af_cond_rast = (overlap_pct_rast * af_target) + ((1-overlap_pct_rast) * raw_cond_rast)
    return af_cond_rast


def call_dummy_connectivity_model(cond_rast_b4,
                                  cond_rast_af,
                                 ):
    print("\n\n\tWARNING: USING A STAND-IN FOR THE REAL CONNECTIVITY ANALYSIS!\n")
    conn = [r.rolling(x=10,
                       y=10,
                       min_periods=1,
                      ).mean() for r in [cond_rast_b4, cond_rast_af]]
    return conn


def connectivity_model(cond,
                                avg_move_ceil=2500,
                                coeff_a=60,
                                coeff_b=4.2,
                               ):
    # calculate α raster as an exponential scaling of avg_move_ceil,
    # but limiting values by using a the upper bound on the realized 1/α
    # (i.e., on avg movement)
    α = 1/((coeff_a*np.exp(coeff_b*cond)).clip(min=None,
                                               max=avg_move_ceil,
                                               keep_attrs=True,
                                              ))
    # get raster res, to express permeability calc in units of cell widths
    res = np.mean(np.abs(cond.rio.resolution()))
    # calculate the permeability raster as w_ij = e^(-αd_ij), but with
    # orthogonal cell size (i.e., res) substituted for d_ij
    perm = np.exp(-α*res)
    # change dtype to float32. Leaving as float64 will cause issues with the connectivity service
    perm = perm.astype(np.float32)
    assert np.nanmin(perm) >= 0
    assert np.nanmax(perm) <= 1
    max_dist_in_perm_rast = (1/np.log(perm)*-1*res).max()
    assert (max_dist_in_perm_rast <= avg_move_ceil or
            np.allclose(max_dist_in_perm_rast, avg_move_ceil))
    # create the payload
    payload = xr.Dataset({"habval": cond, "perm": perm})
    conn = connectivity.run_connectivity(payload)
    conn = conn.assign_coords(x=cond.x, y=cond.y)
    return conn

def run_model():
    pass

def call_connectivity_model(cond_rast_b4,
                            cond_rast_af,
                            conn_rast_max,
                            within_ecosys_conn=False,
                            ecosys_rast=None,
                            ecosys_num=None,
                            use_omniscape=True,
                            norm_curr=True,
                            mock=False,
                           ):
    '''
    call the connecitvity model on the given before and after condition rasters,
    setting the resistance of any 0 pixels in the ecosystem raster to 1 if
    connectivity is only to be calculated within the ecosystem type
    (i.e., if within_ecosys_conn is True),
    then return before and after connectivity maps
    '''
    if within_ecosys_conn:
        assert ecosys_rast is not None and ecosys_num is not None
        valid_ecosys_rast = ecosys_rast==ecosys_num
        cond_b4 = cond_rast_b4.where(valid_ecosys_rast==1, 0)
        cond_af = cond_rast_af.where(valid_ecosys_rast==1, 0)
    else:
        cond_b4 = cond_rast_b4[:, :]
        cond_af = cond_rast_af[:, :]
    if not mock:
        conn_rast_b4 = connectivity_model(cond_b4)
        conn_rast_af = connectivity_model(cond_af)
    elif mock:
        conn_rast_b4, conn_rast_af = call_dummy_connectivity_model(cond_b4, cond_af)
    # normalize connectivity results to the [0, 1] interval
    conn_rast_b4 = conn_rast_b4/conn_rast_max
    conn_rast_af = conn_rast_af/conn_rast_max
    return conn_rast_b4, conn_rast_af


def calc_persistence_change(ecosys_rast,
                            ecosys_num,
                            ecosys_acct_df,
                            overlap_pct_rast,
                            cond_rast_b4,
                            cond_rast_af,
                            cond_rast_raw,
                            conn_rast_b4,
                            conn_rast_af,
                            conn_rast_nat,
                            c=1,
                            z=0.27,
                            rescale_Δ_persist=True,
                            rescale_stat='median',
                            plot_it=False,
                           ):
    '''
    calculate the difference between expected regional biodiversity persistence
    before and after the management action, using a universal species-area
    relationship and the given before and after condition and connectivity
    rasters
    '''
    # calculate a boolean raster indicating all cells that are mapped as being
    # in the target ecosys and/or within the polygons that were flagged
    # by PLANR as being in this ecosystem (even if that disagrees with
    # our map, because the point of PLANR allowing this override is that the
    # user is indicating that the mapping is incorrect)
    ecosys_mask = ((ecosys_rast==ecosys_num) + (overlap_pct_rast>0)) > 0

    # get the ecosystem's pre-1750 spatial extent (implying
    # condition==1.0 everywhere), which will be used to normalize SAR
    # y-axis values
    ecosys_acct_row = ecosys_acct_df[ecosys_acct_df['ecosys'] == ecosys_num]
    pre1750_ecosys_extent = ecosys_acct_row['areatot_connadj_compcorr'].values[0]

    # get the ecosystem's 'current' effective extent (i.e., sum of approx.
    # current condition-adjusted area), adjusted to account for
    # community compositional overlap and connectivity
    curr_eff_ecosys_area=ecosys_acct_row['areaeff_connadj_compcorr'].values[0]

    # define the SAR
    def SAR(area,
            c=c,
            z=z,
            max_x=pre1750_ecosys_extent,
           ):
        val = c*(area**z)
        max_val = c*(max_x**z)
        return float(val/max_val)

    # calculate connectivity-adjusted condition (the sum of local
    # condition changes plus landscape-wide enhancement of condition
    # because of improved connectivity)
    conn_adj_cond_b4 = cond_rast_b4 + conn_rast_b4
    conn_adj_cond_af = cond_rast_af + conn_rast_af

    # calculate differentials between 1.) local connectivity-adjusted
    # condition, calculated on the fly using the provided condition values
    # that override the values in HCAS and the connectivity map they generate;
    # and 2.) local connectivity-adjusted condition, from the raw HCAS values
    # and the nationwide connectivity results calculated using those values
    conn_adj_cond_raw = cond_rast_raw + conn_rast_nat
    conn_adj_cond_b4_diff = conn_adj_cond_b4 - conn_adj_cond_raw
    conn_adj_cond_af_diff = conn_adj_cond_af - conn_adj_cond_raw

    # set all invalid cells (i.e., cells outside the target ecosystem,
    # whose change in condition should not contribute to this ecosystem's SAR
    # result) to have the same before and after values
    # TODO: LATER ON, CONSIDER IMPLEMENTING CONNECTIVITY-CORRECTION PROPERLY USING
    #       ECOSYSTEM COMPOSITIONAL OVERLAP INFORMATION; INACCURACY SHOULD BE
    #       MINIMAL IN THE MEANTIME
    conn_adj_cond_af_diff = ((conn_adj_cond_af_diff * (ecosys_mask == 1)) +
                             (conn_adj_cond_b4_diff * (ecosys_mask == 0)))

    # TODO: THEN CONS SIGNIF SHOULD ALSO BE UPDATED, STRICTLY SPEAKING!

    # use the raster resolution to express that sum in hectares
    assert (np.mean(np.abs(ecosys_rast.rio.resolution())) ==
            np.mean(np.abs(overlap_pct_rast.rio.resolution())) ==
            np.mean(np.abs(cond_rast_b4.rio.resolution())) ==
            np.mean(np.abs(cond_rast_af.rio.resolution())) ==
            np.mean(np.abs(cond_rast_raw.rio.resolution())) ==
            np.mean(np.abs(conn_rast_b4.rio.resolution())) ==
            np.mean(np.abs(conn_rast_af.rio.resolution())) ==
            np.mean(np.abs(conn_rast_nat.rio.resolution())))
    res = np.mean(np.abs(ecosys_rast.rio.resolution()))
    conn_adj_cond_b4_diff = np.nansum(conn_adj_cond_b4_diff)*(res**2)/(100**2)
    conn_adj_cond_af_diff = np.nansum(conn_adj_cond_af_diff)*(res**2)/(100**2)

    # update the current effective nationwide ecosystem extent to account for
    # this differential (in other words, retrofit that number to what it would
    # have been if, ceteris paribus, the local conditions were what
    # is being modeled here when the calc_cons_signif.py script was run
    b4_eff_ecosys_area = curr_eff_ecosys_area + conn_adj_cond_b4_diff
    af_eff_ecosys_area = curr_eff_ecosys_area + conn_adj_cond_af_diff

    # calculate expected species persistence
    persist_b4 = SAR(b4_eff_ecosys_area)
    persist_af = SAR(af_eff_ecosys_area)
    Δ_persist = persist_af - persist_b4
    if rescale_Δ_persist:
        assert rescale_stat in ['mean', 'median']
        # get the expected Δ_persist per hectare of perfect-condition
        # habitat, to be used to rescale all SAR results
        if rescale_stat == 'mean':
            mean_Δ_persist_per_ha = np.mean(ecosys_acct_df['delta_persist_1ha'].values)
            Δ_persist = Δ_persist/mean_Δ_persist_per_ha
        elif rescale_stat == 'median':
            medn_Δ_persist_per_ha = np.median(ecosys_acct_df['delta_persist_1ha'].values)
            Δ_persist = Δ_persist/medn_Δ_persist_per_ha

    # lastly, calculate ecosystem-wide change in condition,
    # expressed in hectares, resulting from change in connectivity
    # (NOTE: needed for ability to numerically check NBAS outputs
    #        without redoing spatial analysis)
    tot_ecosys_conn_change = ((conn_rast_af - conn_rast_b4) *
                ((ecosys_rast==ecosys_num) | (overlap_pct_rast>0))
                             ).sum().values.ravel()[0]*(res**2)/(100**2)

    return persist_b4, persist_af, Δ_persist, tot_ecosys_conn_change


def summarize_cond_results(b4_cond,
                           af_cond,
                           overlap_rast,
                           raw_cond,
                          ):
    '''
    summarize the before, after, and Δ results for the before and after
    condition rasters, for all pixels that overlap the set
    of areas represented by the overlap_rast and accounting for percent overlap
    '''
    Δ_cond = af_cond - b4_cond
    # NOTE: to correctly calculate the average per-area before and after condition values
    #       that were used in the analysis, we need to use the raw condition raster
    #       to algebraically reverse the overlap percent-weighted sum that determined
    #       the before- and after-action condition rasters
    tot_area_in_pix = overlap_rast.sum().values.ravel()[0]
    b4_summ = (b4_cond - (raw_cond * (1-overlap_rast))).sum().values.ravel()[0]/tot_area_in_pix
    af_summ = (af_cond - (raw_cond * (1-overlap_rast))).sum().values.ravel()[0]/tot_area_in_pix
    Δ_summ = Δ_cond.where(overlap_rast > 0).sum().values.ravel()[0]/tot_area_in_pix
    return b4_summ, af_summ, Δ_summ


def summarize_conn_results(b4_conn,
                           af_conn,
                           overlap_rast,
                          ):
    '''
    summarize the before, after, and Δ results for the before and after
    connectivity rasters, for all pixels that overlap the set
    of areas represented by the overlap_rast and accounting for percent overlap
    '''
    Δ_conn = af_conn - b4_conn
    tot_overlap_in_pix = overlap_rast.sum().values.ravel()[0]
    b4_summ = (b4_conn*overlap_rast).sum().values.ravel()[0]/tot_overlap_in_pix
    af_summ = (af_conn*overlap_rast).sum().values.ravel()[0]/tot_overlap_in_pix
    Δ_summ = (Δ_conn*overlap_rast).sum().values.ravel()[0]/tot_overlap_in_pix
    return b4_summ, af_summ, Δ_summ


def summarize_cons_sig_results(areas_gdf,
                               ecosys_acct_df,
                               on='ecosys',
                              ):
    '''
    returns a tuple giving the range (min and max) of conservation signficance
    values occuring within the set of activity areas defined by areas_gdf
    '''
    cons_sig_vals = pd.merge(areas_gdf, ecosys_acct_df, on=on)['cons_sig'].values
    min_sig_val = np.min(cons_sig_vals)
    max_sig_val = np.max(cons_sig_vals)
    cons_sig_summary = (min_sig_val, max_sig_val)
    return cons_sig_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NBAS on a set of activity areas')
    parser.add_argument('payload_path', type=str, help='Path to the JSON payload file')
    args = parser.parse_args()
    with open(args.payload_path, 'r') as f:
        payload = json.load(f)
    nbas(payload)