# -*- coding: utf-8 -*-
import os
import gc
import pandas as pd
import geopandas as gpd
import numpy as np
import h3
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ==============================================================================
# STAGE 1: CREATE H3 TO TTWA MAPPING FILE (FROM MULTIPLE LARGE FILES)
# ==============================================================================


def stage_1_create_mapping_file_memory_safe(od_filepaths, ttwa_filepath, output_filepath):
    """
    MODIFIED: This memory-safe version scans a list of potentially massive OD
    files. It reads each file in chunks to extract all unique H3 IDs without
    loading any entire file into memory.
    """
    print("--- Stage 1: Creating H3 to TTWA Mapping File (Memory-Safe) ---")

    unique_h3_ids = set()
    chunk_size = 5_000_000
    required_cols = ['ORIGIN_CODE', 'CODE']

    print(
        f"  Scanning {len(od_filepaths)} files in chunks for unique H3 IDs...")

    # Outer loop for each file path
    for f_path in tqdm(od_filepaths, desc="Scanning Files"):
        try:
            # Inner loop for each chunk within a file
            chunk_iterator = pd.read_csv(
                f_path, sep='\t', usecols=required_cols, chunksize=chunk_size, low_memory=True)
            for chunk in chunk_iterator:
                chunk.dropna(subset=required_cols, inplace=True)
                chunk = chunk[(chunk['ORIGIN_CODE'] != 'other')
                              & (chunk['CODE'] != 'other')]
                unique_h3_ids.update(chunk['ORIGIN_CODE'])
                unique_h3_ids.update(chunk['CODE'])
        except FileNotFoundError:
            print(f"\n  Warning: File not found, skipping: {f_path}")
        except ValueError:
            print(
                f"\n  Warning: Could not find columns {required_cols} in {f_path}, skipping.")
        except Exception as e:
            print(f"\n  An error occurred with {f_path}: {e}")

    if 'other' in unique_h3_ids:
        unique_h3_ids.remove('other')

    print(f"  Found {len(unique_h3_ids)} unique H3 IDs across all files.")

    # The rest of the process uses the relatively small set of unique H3s
    print("  Loading TTWA boundaries...")
    ttwa_gdf = gpd.read_file(ttwa_filepath)

    print("  Converting H3 IDs to coordinates...")
    h3_df = pd.DataFrame(list(unique_h3_ids), columns=['h3_id'])
    h3_df['lat'], h3_df['lon'] = zip(*h3_df['h3_id'].apply(h3.cell_to_latlng))

    print("  Creating GeoDataFrame and transforming to British National Grid (EPSG:27700)...")
    h3_gdf_wgs84 = gpd.GeoDataFrame(h3_df, geometry=gpd.points_from_xy(
        h3_df.lon, h3_df.lat), crs="EPSG:4326")
    h3_gdf_bng = h3_gdf_wgs84.to_crs("EPSG:27700")
    h3_gdf_bng['easting'] = h3_gdf_bng.geometry.x
    h3_gdf_bng['northing'] = h3_gdf_bng.geometry.y

    print("  Performing spatial join to map H3 cells to TTWAs...")
    ttwa_gdf_bng = ttwa_gdf.to_crs("EPSG:27700")
    h3_with_ttwa = gpd.sjoin(h3_gdf_bng, ttwa_gdf_bng,
                             how="inner", predicate='within')

    mapping_output = h3_with_ttwa[[
        'h3_id', 'TTWA11NM', 'TTWA11CD', 'easting', 'northing']]
    mapping_output.to_parquet(output_filepath, index=False)

    print(
        f"  SUCCESS: Stage 1 complete. Mapping file saved to '{output_filepath}'")

    del h3_df, h3_gdf_wgs84, h3_gdf_bng, h3_with_ttwa, mapping_output, ttwa_gdf
    gc.collect()
    return True

# ==============================================================================
# STAGE 2: FILTER AND COMBINE FLOWS FROM MULTIPLE FILES
# ==============================================================================


def stage_2_filter_and_combine_flows(od_filepaths, mapping_filepath, output_filepath):
    """
    MODIFIED: Processes a list of large OD files. It streams each file in chunks,
    extracts the date from the filename, filters for intra-TTWA flows, and appends
    the results to a single output Parquet file.
    """
    print("\n--- Stage 2: Filtering and Combining Flows from All Files ---")

    print("  Loading H3 mapping file...")
    h3_mapping = pd.read_parquet(mapping_filepath)

    chunk_size = 5_000_000
    required_cols = ['ORIGIN_CODE', 'CODE', 'EXTRAPOLATED_NUMBER_OF_USERS']

    print(
        f"  Processing {len(od_filepaths)} files and streaming to '{output_filepath}'...")

    parquet_writer = None

    if os.path.exists(output_filepath):
        print(f"  Removing old file '{output_filepath}' before writing.")
        os.remove(output_filepath)

    # Outer loop for each file
    for f_path in tqdm(od_filepaths, desc="Processing Files"):
        filename = os.path.basename(f_path)
        try:
            # Extract date from filename (e.g., ..._2025-03-03_...)
            date_str = filename.split('_')[3]
            chunk_iterator = pd.read_csv(
                f_path, sep='\t', usecols=required_cols, chunksize=chunk_size, low_memory=True)
        except FileNotFoundError:
            print(f"\n  Warning: File not found, skipping: {f_path}")
            continue
        except (ValueError, IndexError):
            print(
                f"\n  Warning: Could not find columns or parse date for {filename}, skipping.")
            continue

        # Inner loop for each chunk
        for chunk in chunk_iterator:
            # Add the DAY column extracted from the filename
            chunk['DAY'] = date_str

            merged_chunk = pd.merge(chunk, h3_mapping.add_prefix(
                'o_'), left_on='ORIGIN_CODE', right_on='o_h3_id', how='inner')
            merged_chunk = pd.merge(merged_chunk, h3_mapping.add_prefix(
                'd_'), left_on='CODE', right_on='d_h3_id', how='inner')

            intra_ttwa_chunk = merged_chunk[merged_chunk['o_TTWA11CD']
                                            == merged_chunk['d_TTWA11CD']].copy()

            if not intra_ttwa_chunk.empty:
                intra_ttwa_chunk.rename(
                    columns={'o_TTWA11NM': 'TTWA_Name', 'o_TTWA11CD': 'TTWA_Code'}, inplace=True)
                analysis_chunk = intra_ttwa_chunk[['DAY', 'TTWA_Name', 'TTWA_Code', 'ORIGIN_CODE', 'CODE',
                                                   'EXTRAPOLATED_NUMBER_OF_USERS', 'o_easting', 'o_northing', 'd_easting', 'd_northing']]

                table = pa.Table.from_pandas(
                    analysis_chunk, preserve_index=False)

                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(
                        output_filepath, table.schema)

                parquet_writer.write_table(table)

    if parquet_writer:
        parquet_writer.close()
        print(
            f"\n  SUCCESS: Stage 2 complete. Combined filtered data saved to '{output_filepath}'")
    else:
        print("\n  Warning: No intra-TTWA flows were found in any file. Output file not created.")

    return True

# ==============================================================================
# STAGE 3: CALCULATE AND AGGREGATE INDICATORS (NO CHANGE NEEDED)
# ==============================================================================


def stage_3_calculate_indicators(flows_filepath, ttwa_filepath, output_filepath):
    """
    This function remains unchanged. It is already designed to read the single,
    combined Parquet file created by Stage 2 and process it day by day.
    """
    print("\n--- Stage 3: Calculating Daily Indicators and Aggregating ---")

    if not os.path.exists(flows_filepath):
        print(
            f"  FATAL: Input file '{flows_filepath}' not found. Did Stage 2 run successfully?")
        return False

    print("  Loading TTWA boundaries...")
    ttwa_gdf_bng = gpd.read_file(ttwa_filepath).to_crs("EPSG:27700")

    print("  Getting unique days from the data...")
    try:
        unique_days = pd.read_parquet(flows_filepath, columns=['DAY'])[
            'DAY'].unique()
    except (pa.lib.ArrowInvalid, KeyError):
        print("  FATAL: Flow data is empty or invalid, or is missing the 'DAY' column.")
        return False

    daily_results_list = []
    print(f"  Starting calculation for {len(unique_days)} days...")

    for day in tqdm(unique_days, desc="Processing Days"):
        daily_flows = pd.read_parquet(
            flows_filepath, filters=[('DAY', '==', day)])

        for ttwa_code, ttwa_flows in daily_flows.groupby('TTWA_Code'):
            ttwa_flows = ttwa_flows.copy()
            ttwa_flows['dx'] = ttwa_flows['d_easting'] - \
                ttwa_flows['o_easting']
            ttwa_flows['dy'] = ttwa_flows['d_northing'] - \
                ttwa_flows['o_northing']
            ttwa_flows['distance'] = np.sqrt(
                ttwa_flows['dx']**2 + ttwa_flows['dy']**2)

            ttwa_poly = ttwa_gdf_bng[ttwa_gdf_bng['TTWA11CD']
                                     == ttwa_code].geometry.iloc[0]
            center_easting, center_northing = ttwa_poly.centroid.x, ttwa_poly.centroid.y
            ttwa_name = ttwa_flows['TTWA_Name'].iloc[0]

            origins_in_ttwa = ttwa_flows.groupby('ORIGIN_CODE')
            total_outflow = origins_in_ttwa['EXTRAPOLATED_NUMBER_OF_USERS'].sum(
            )

            pmv_x = origins_in_ttwa.apply(lambda g: np.average(
                g['dx'], weights=g['EXTRAPOLATED_NUMBER_OF_USERS']), include_groups=False)
            pmv_y = origins_in_ttwa.apply(lambda g: np.average(
                g['dy'], weights=g['EXTRAPOLATED_NUMBER_OF_USERS']), include_groups=False)

            pmv_df = pd.DataFrame(
                {'pmv_x': pmv_x, 'pmv_y': pmv_y, 'total_outflow': total_outflow}).reset_index()

            if pmv_df.empty:
                continue

            origin_coords = ttwa_flows[[
                'ORIGIN_CODE', 'o_easting', 'o_northing']].drop_duplicates().set_index('ORIGIN_CODE')
            pmv_df = pmv_df.merge(origin_coords, on='ORIGIN_CODE')

            pmv_df['center_vec_x'] = center_easting - pmv_df['o_easting']
            pmv_df['center_vec_y'] = center_northing - pmv_df['o_northing']

            dot_product = (pmv_df['pmv_x'] * pmv_df['center_vec_x']) + \
                (pmv_df['pmv_y'] * pmv_df['center_vec_y'])
            pmv_mag = np.sqrt(pmv_df['pmv_x']**2 + pmv_df['pmv_y']**2)
            center_vec_mag = np.sqrt(
                pmv_df['center_vec_x']**2 + pmv_df['center_vec_y']**2)

            cos_theta = np.divide(dot_product, (pmv_mag * center_vec_mag),
                                  out=np.zeros(len(pmv_df)), where=(pmv_mag * center_vec_mag) != 0)
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
                return np.sqrt(-2 * np.log(r + 1e-12)) if r < 1.0 else np.nan

            anisotropy_per_origin = origins_in_ttwa.apply(
                manual_weighted_circ_std, include_groups=False)
            valid_anisotropy = anisotropy_per_origin.dropna()

            lambda_val = np.nan
            if not valid_anisotropy.empty:
                valid_weights = pmv_df.set_index(
                    'ORIGIN_CODE').loc[valid_anisotropy.index, 'total_outflow']
                if valid_weights.sum() > 0:
                    lambda_val = np.average(
                        valid_anisotropy, weights=valid_weights)

            daily_results_list.append({'DAY': day, 'TTWA_Code': ttwa_code, 'TTWA_Name': ttwa_name, 'Centripetality_Gamma': gamma,
                                      'Anisotropy_Lambda': lambda_val, 'Daily_Intra_TTWA_Users': ttwa_flows['EXTRAPOLATED_NUMBER_OF_USERS'].sum()})
        del daily_flows
        gc.collect()

    print("\n  Aggregating daily results...")
    if not daily_results_list:
        print(
            "  Warning: No valid daily results were calculated. Final output will be empty.")
        final_aggregated_df = pd.DataFrame()
    else:
        daily_results_df = pd.DataFrame(daily_results_list)
        aggregations = {'Centripetality_Gamma': ['mean', 'std'], 'Anisotropy_Lambda': [
            'mean', 'std'], 'Daily_Intra_TTWA_Users': ['mean', 'std', 'sum']}
        final_aggregated_df = daily_results_df.groupby(
            ['TTWA_Code', 'TTWA_Name']).agg(aggregations)
        final_aggregated_df.columns = [
            '_'.join(col).strip() for col in final_aggregated_df.columns.values]
        final_aggregated_df = final_aggregated_df.reset_index()

    final_aggregated_df.to_csv(output_filepath, index=False)
    print(
        f"\n  SUCCESS: Stage 3 complete. Final aggregated indicators saved to '{output_filepath}'")
    return True

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================


def main():
    """
    Main function to execute the entire memory-safe analysis workflow for multiple files.
    """
    # ----- USER-CONFIGURABLE PATHS -----
    DATA_DIR = 'ALL'
    TTWA_BOUNDARY_FILE = 'boundary/Travel_to_Work_Areas_Dec_2011_FCB_in_United_Kingdom_2022.geojson'

    daily_filenames = [
        'Audience_Profiles_Destination_2025-03-03_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-04_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-10_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-11_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-12_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-13_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-17_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-18_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-19_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-20_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-24_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-25_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-26_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-27_loco_all_tracks.tsv',
        'Audience_Profiles_Destination_2025-03-31_loco_all_tracks.tsv'
    ]
    od_filepaths = [os.path.join(DATA_DIR, f) for f in daily_filenames]

    # ----- Paths for intermediate and final files -----
    MAPPING_FILE = 'h3_to_ttwa_mapping.parquet'
    INTRA_TTWA_FLOWS_FILE = 'intra_ttwa_flows_bng_combined.parquet'
    FINAL_INDICATORS_FILE = 'ttwa_mobility_indicators_daily_aggregated.csv'

    # --- Execute the analysis workflow ---
    if not os.path.exists(MAPPING_FILE):
        print("--- Running Stage 1 ---")
        if not stage_1_create_mapping_file_memory_safe(od_filepaths, TTWA_BOUNDARY_FILE, MAPPING_FILE):
            print("Stage 1 failed. Aborting workflow.")
            return
    else:
        print("--- SKIPPING STAGE 1 (mapping file exists) ---")

    print("\n--- Running Stage 2 ---")
    if not stage_2_filter_and_combine_flows(od_filepaths, MAPPING_FILE, INTRA_TTWA_FLOWS_FILE):
        print("Stage 2 failed. Aborting workflow.")
        return

    print("\n--- Running Stage 3 ---")
    if not stage_3_calculate_indicators(INTRA_TTWA_FLOWS_FILE, TTWA_BOUNDARY_FILE, FINAL_INDICATORS_FILE):
        print("Stage 3 failed.")
        return

    print("\n✅ All stages completed successfully!")


if __name__ == '__main__':
    main()
