# -*- coding: utf-8 -*-
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import h3
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import circstd
from tqdm import tqdm

# ==============================================================================
# STAGE 1: CREATE H3 TO TTWA MAPPING FILE
# ==============================================================================
def stage_1_create_mapping_file(od_filepath, ttwa_filepath, output_filepath):
    # Create a mapping file from H3 cells to TTWA boundaries
    print("--- Stage 1: Creating H3 to TTWA Mapping File ---")
    print("  Loading OD data and TTWA boundaries...")
    try:
        od_df = pd.read_csv(od_filepath, usecols=['ORIGIN_CODE', 'CODE'])
    except ValueError:
        print(f"  Warning: Could not find 'ORIGIN_CODE' or 'CODE' in {od_filepath}.")
        od_df = pd.DataFrame({'ORIGIN_CODE': [], 'CODE': []})
    ttwa_gdf = gpd.read_file(ttwa_filepath)
    print(f"  Number of rows in raw data: {len(od_df)}")
    od_df = od_df[(od_df['ORIGIN_CODE'] != 'other') & (od_df['CODE'] != 'other')]
    od_df.dropna(subset=['ORIGIN_CODE', 'CODE'], inplace=True)
    print(f"  Number of valid rows after filtering: {len(od_df)}")
    print("  Extracting unique H3 IDs and converting to coordinates...")
    all_unique_h3_ids = pd.unique(np.concatenate((od_df['ORIGIN_CODE'].unique(), od_df['CODE'].unique())))
    h3_df = pd.DataFrame(all_unique_h3_ids, columns=['h3_id'])
    h3_df['lat'], h3_df['lon'] = zip(*h3_df['h3_id'].apply(h3.cell_to_latlng))
    print("  Creating GeoDataFrame and transforming to British National Grid (EPSG:27700)...")
    h3_gdf_wgs84 = gpd.GeoDataFrame(
        h3_df, geometry=gpd.points_from_xy(h3_df.lon, h3_df.lat), crs="EPSG:4326"
    )
    h3_gdf_bng = h3_gdf_wgs84.to_crs("EPSG:27700")
    h3_gdf_bng['easting'] = h3_gdf_bng.geometry.x
    h3_gdf_bng['northing'] = h3_gdf_bng.geometry.y
    print("  Performing spatial join to map H3 cells to TTWAs...")
    ttwa_gdf_bng = ttwa_gdf.to_crs("EPSG:27700")
    h3_with_ttwa = gpd.sjoin(h3_gdf_bng, ttwa_gdf_bng, how="inner", predicate='within')
    mapping_output = h3_with_ttwa[['h3_id', 'TTWA11NM', 'TTWA11CD', 'easting', 'northing']]
    mapping_output.to_parquet(output_filepath, index=False)
    print(f"  Stage 1 complete. Mapping file saved to '{output_filepath}'")
    return mapping_output

# ==============================================================================
# STAGE 2: FILTER FOR INTRA-TTWA FLOWS (INCLUDES DAY)
# ==============================================================================
def stage_2_filter_intra_ttwa_flows(od_filepath, mapping_filepath, output_filepath):
    # Process OD file in chunks, filter for intra-TTWA flows, and write results incrementally to Parquet
    print("\n--- Stage 2: Filtering for Intra-TTWA Flows (Streaming to Disk with DAY) ---")
    print("  Loading H3 mapping file...")
    h3_mapping = pd.read_parquet(mapping_filepath)
    chunk_size = 5_000_000
    required_cols = ['DAY', 'ORIGIN_CODE', 'CODE', 'EXTRAPOLATED_NUMBER_OF_USERS']
    print(f"  Processing '{od_filepath}' and streaming results to '{output_filepath}'...")
    try:
        csv_iterator = pd.read_csv(od_filepath, usecols=required_cols, chunksize=chunk_size)
    except ValueError:
        print(f"  Warning: Could not find required columns in {od_filepath}.")
        return
    parquet_writer = None
    if os.path.exists(output_filepath):
        print(f"  Removing old file '{output_filepath}' before writing.")
        os.remove(output_filepath)
    for chunk in tqdm(csv_iterator, desc="  Processing chunks"):
        merged_chunk = pd.merge(
            chunk, h3_mapping.add_prefix('o_'),
            left_on='ORIGIN_CODE', right_on='o_h3_id', how='inner'
        )
        merged_chunk = pd.merge(
            merged_chunk, h3_mapping.add_prefix('d_'),
            left_on='CODE', right_on='d_h3_id', how='inner'
        )
        intra_ttwa_chunk = merged_chunk[merged_chunk['o_TTWA11CD'] == merged_chunk['d_TTWA11CD']].copy()
        if not intra_ttwa_chunk.empty:
            intra_ttwa_chunk.rename(columns={'o_TTWA11NM': 'TTWA_Name', 'o_TTWA11CD': 'TTWA_Code'}, inplace=True)
            analysis_chunk = intra_ttwa_chunk[[
                'DAY', 'TTWA_Name', 'TTWA_Code', 'ORIGIN_CODE', 'CODE', 'EXTRAPOLATED_NUMBER_OF_USERS',
                'o_easting', 'o_northing', 'd_easting', 'd_northing'
            ]]
            table = pa.Table.from_pandas(analysis_chunk)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_filepath, table.schema)
            parquet_writer.write_table(table)
    if parquet_writer:
        parquet_writer.close()
        print(f"\n  Stage 2 complete. Cleaned data saved to '{output_filepath}'")
    else:
        print("\n  Warning: No intra-TTWA flows were found. Output file was not created.")

# ==============================================================================
# STAGE 3: CALCULATE INDICATORS PER DAY, THEN AGGREGATE
# ==============================================================================
def stage_3_calculate_indicators(flows_filepath, ttwa_filepath, output_filepath):
    # Calculate mobility indicators by processing flow data one DAY at a time, then aggregate
    print("\n--- Stage 3: Calculating Daily Indicators and Aggregating ---")
    if not os.path.exists(flows_filepath):
        print(f"  Input file '{flows_filepath}' not found. Skipping Stage 3.")
        return
    print("  Loading TTWA boundaries...")
    ttwa_gdf = gpd.read_file(ttwa_filepath).to_crs("EPSG:27700")
    print("  Getting unique days and TTWA codes from the data...")
    try:
        plan_df = pd.read_parquet(flows_filepath, columns=['DAY', 'TTWA_Code'])
        unique_days = plan_df['DAY'].unique()
        unique_ttwas = plan_df['TTWA_Code'].unique()
        del plan_df
    except (pa.lib.ArrowInvalid, ValueError, KeyError):
        print("  Flow data is empty or invalid, or is missing the 'DAY'/'TTWA_Code' column.")
        print("  Please ensure you have re-run Stage 2 with the latest code.")
        return
    daily_results_list = []
    print(f"  Starting calculation for {len(unique_days)} days and {len(unique_ttwas)} TTWAs...")
    print("  This may take a long time.")
    for day in tqdm(unique_days, desc="Processing Days"):
        daily_flows = pd.read_parquet(flows_filepath, filters=[('DAY', '==', day)])
        for ttwa_code in unique_ttwas:
            ttwa_flows = daily_flows[daily_flows['TTWA_Code'] == ttwa_code]
            if ttwa_flows.empty:
                continue
            ttwa_flows = ttwa_flows.copy()
            ttwa_flows['dx'] = ttwa_flows['d_easting'] - ttwa_flows['o_easting']
            ttwa_flows['dy'] = ttwa_flows['d_northing'] - ttwa_flows['o_northing']
            ttwa_flows['distance'] = np.sqrt(ttwa_flows['dx']**2 + ttwa_flows['dy']**2)
            ttwa_poly = ttwa_gdf[ttwa_gdf['TTWA11CD'] == ttwa_code].geometry.iloc[0]
            center_easting, center_northing = ttwa_poly.centroid.x, ttwa_poly.centroid.y
            ttwa_name = ttwa_flows['TTWA_Name'].iloc[0]
            origins_in_ttwa = ttwa_flows.groupby('ORIGIN_CODE')
            total_outflow = origins_in_ttwa['EXTRAPOLATED_NUMBER_OF_USERS'].sum()
            pmv_x = origins_in_ttwa.apply(lambda g: np.average(g['dx'], weights=g['EXTRAPOLATED_NUMBER_OF_USERS']), include_groups=False)
            pmv_y = origins_in_ttwa.apply(lambda g: np.average(g['dy'], weights=g['EXTRAPOLATED_NUMBER_OF_USERS']), include_groups=False)
            pmv_df = pd.DataFrame({'pmv_x': pmv_x, 'pmv_y': pmv_y, 'total_outflow': total_outflow}).reset_index()
            if pmv_df.empty:
                continue
            origin_coords = ttwa_flows[['ORIGIN_CODE', 'o_easting', 'o_northing']].drop_duplicates().set_index('ORIGIN_CODE')
            pmv_df = pmv_df.merge(origin_coords, on='ORIGIN_CODE')
            pmv_df['center_vec_x'] = center_easting - pmv_df['o_easting']
            pmv_df['center_vec_y'] = center_northing - pmv_df['o_northing']
            dot_product = (pmv_df['pmv_x'] * pmv_df['center_vec_x']) + (pmv_df['pmv_y'] * pmv_df['center_vec_y'])
            pmv_mag = np.sqrt(pmv_df['pmv_x']**2 + pmv_df['pmv_y']**2)
            center_vec_mag = np.sqrt(pmv_df['center_vec_x']**2 + pmv_df['center_vec_y']**2)
            cos_theta = np.divide(dot_product, (pmv_mag * center_vec_mag), out=np.zeros(len(pmv_df)), where=(pmv_mag * center_vec_mag)!=0)
            gamma = np.average(cos_theta, weights=pmv_df['total_outflow'])
            def manual_weighted_circ_std(group):
                group = group[group['distance'] > 0]
                if len(group) < 2:
                    return np.nan
                weights = group['EXTRAPOLATED_NUMBER_OF_USERS']
                angles = np.arctan2(group['dy'], group['dx'])
                sum_weights = np.sum(weights)
                if sum_weights == 0:
                    return np.nan
                w_cos = np.sum(weights * np.cos(angles))
                w_sin = np.sum(weights * np.sin(angles))
                r = np.sqrt(w_cos**2 + w_sin**2) / sum_weights
                if r > 1.0:
                    r = 1.0
                circ_std = np.sqrt(-2 * np.log(r + 1e-12))
                return circ_std
            anisotropy_per_origin = origins_in_ttwa.apply(manual_weighted_circ_std, include_groups=False)
            valid_anisotropy = anisotropy_per_origin.dropna()
            if valid_anisotropy.empty:
                lambda_val = np.nan
            else:
                valid_weights = pmv_df.set_index('ORIGIN_CODE').loc[valid_anisotropy.index, 'total_outflow']
                lambda_val = np.average(valid_anisotropy, weights=valid_weights)
            daily_results_list.append({
                'DAY': day,
                'TTWA_Code': ttwa_code,
                'TTWA_Name': ttwa_name,
                'Centripetality_Gamma': gamma,
                'Anisotropy_Lambda': lambda_val,
                'Daily_Intra_TTWA_Users': ttwa_flows['EXTRAPOLATED_NUMBER_OF_USERS'].sum()
            })
    print("\n  Aggregating daily results...")
    if not daily_results_list:
        print("  No valid daily results were calculated. Saving an empty file.")
        final_aggregated_df = pd.DataFrame()
    else:
        daily_results_df = pd.DataFrame(daily_results_list)
        aggregations = {
            'Centripetality_Gamma': ['mean', 'std'],
            'Anisotropy_Lambda': ['mean', 'std'],
            'Daily_Intra_TTWA_Users': ['mean', 'std', 'sum']
        }
        final_aggregated_df = daily_results_df.groupby(['TTWA_Code', 'TTWA_Name']).agg(aggregations)
        final_aggregated_df.columns = ['_'.join(col).strip() for col in final_aggregated_df.columns.values]
        final_aggregated_df = final_aggregated_df.reset_index()
    final_aggregated_df.to_csv(output_filepath, index=False)
    print(f"\n  Stage 3 complete. Final aggregated indicators saved to '{output_filepath}'")
    return final_aggregated_df

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    # Main workflow for the analysis
    OD_DATA_FILE = 'cleaned_data_without_day_17.csv'
    TTWA_BOUNDARY_FILE = 'boundary/Travel_to_Work_Areas_Dec_2011_FCB_in_United_Kingdom_2022.geojson'
    MAPPING_FILE = 'h3_to_ttwa_mapping.parquet'
    INTRA_TTWA_FLOWS_FILE = 'intra_ttwa_flows_bng.parquet'
    FINAL_INDICATORS_FILE = 'ttwa_mobility_indicators_daily_aggregated.csv'
    if not os.path.exists(MAPPING_FILE):
        print("--- Running Stage 1 ---")
        stage_1_create_mapping_file(OD_DATA_FILE, TTWA_BOUNDARY_FILE, MAPPING_FILE)
    else:
        print("--- Skipping Stage 1 (mapping file exists) ---")
    print("--- Running Stage 2 (required for daily analysis) ---")
    stage_2_filter_intra_ttwa_flows(OD_DATA_FILE, MAPPING_FILE, INTRA_TTWA_FLOWS_FILE)
    print("--- Running Stage 3 ---")
    stage_3_calculate_indicators(INTRA_TTWA_FLOWS_FILE, TTWA_BOUNDARY_FILE, FINAL_INDICATORS_FILE)
    print("\nAll stages completed successfully.")

if __name__ == '__main__':
    main()
