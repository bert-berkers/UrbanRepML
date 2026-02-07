#!/usr/bin/env python3
"""
Optimized SRAI-Compliant AlphaEarth TIFF-to-H3 Processing (Sequential Tile, Parallel Chunk)

Strategy: Sequential tile processing with parallel chunk processing within each tile.
Optimized for high CPU utilization and managed memory consumption.
"""

import argparse
import json
import logging
import os
import pickle
import time
import shutil
import gc
import hashlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Essential imports for the main process
try:
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import box
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}. Please install required packages.")
    exit(1)

# Set GDAL environment variable for better behavior in multiprocessing environments
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'TRUE'

# Global debug data collector
DEBUG_DATA = {
    "processing_info": {},
    "tiff_files": {},
    "chunks": {},
    "errors": [],
    "summary": {}
}
DEBUG_ENABLED = False


def add_debug_data(category: str, key: str, data: dict):
    """Add debug data to global collector if debugging enabled."""
    global DEBUG_DATA, DEBUG_ENABLED
    if DEBUG_ENABLED:
        if category not in DEBUG_DATA:
            DEBUG_DATA[category] = {}
        DEBUG_DATA[category][key] = data


def save_debug_json(output_path: str):
    """Save debug data to JSON file."""
    global DEBUG_DATA
    try:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        debug_data_json = convert_numpy(DEBUG_DATA)

        with open(output_path, 'w') as f:
            json.dump(debug_data_json, f, indent=2, default=str)
        print(f"Debug data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving debug data: {e}")


def setup_logging(log_dir: str, study_area: str, resolution: int) -> logging.Logger:
    """Setup comprehensive logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/tiff_to_h3_{study_area}_res{resolution}_{timestamp}.log"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('tiff_to_h3')
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Quieten noisy libraries
    logging.getLogger('fiona').setLevel(logging.WARNING)
    logging.getLogger('rasterio').setLevel(logging.WARNING)

    return logger


def get_optimized_config(max_workers: Optional[int] = None) -> Dict:
    """Get optimized processing configuration."""
    config = {}

    available_cores = os.cpu_count() or 1
    if max_workers:
        config['max_workers'] = max_workers
    else:
        # Optimized for high-core CPUs; leave 1-2 cores free for system stability.
        config['max_workers'] = max(1, available_cores - 2)

    # <<< NEW COMMENT EXPLAINING CHUNK SIZE >>>
    # The chunk_size is a performance tuning parameter, not an error-handling one.
    # The error handling in `process_chunk` is now robust against 'nodata' chunks.
    # - Larger chunks (e.g., 1024) are generally faster due to lower overhead.
    #   Each chunk has a fixed startup cost, so fewer chunks means faster processing.
    # - Smaller chunks (e.g., 256) use less memory per worker but are much slower
    #   due to high overhead. Only use smaller chunks if you are running into
    #   memory (RAM) limitations.
    # A size of 1024 is a good balance for efficiency and memory usage.
    config['chunk_size'] = 1024
    return config


def get_tiff_spatial_bounds(tiff_path: str) -> Tuple[float, float, float, float]:
    """Get spatial bounds of TIFF file in WGS84 coordinates using rasterio."""
    try:
        with rasterio.open(tiff_path) as src:
            bounds = src.bounds
            crs = src.crs

            if crs is None or not crs.is_valid:
                raise ValueError(f"Invalid or missing CRS in {tiff_path}")

            if crs.to_epsg() != 4326:
                transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                lons, lats = transformer.transform(
                    [bounds.left, bounds.right, bounds.left, bounds.right],
                    [bounds.bottom, bounds.bottom, bounds.top, bounds.top]
                )
                return (min(lons), min(lats), max(lons), max(lats))
            else:
                return (bounds.left, bounds.bottom, bounds.right, bounds.top)
    except Exception as e:
        raise IOError(f"Could not read spatial bounds from {tiff_path}: {e}")


def get_tile_hexagons(
        tiff_path: str,
        all_regions: gpd.GeoDataFrame,
        temp_dir: str
) -> Tuple[str, int]:
    """
    Get hexagons from study area regions that intersect with tile bounds.
    """
    # Get tile spatial bounds
    bounds = get_tiff_spatial_bounds(tiff_path)
    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])

    # Find hexagons that intersect with tile
    possible_matches_idx = list(all_regions.sindex.intersection(bbox.bounds))
    possible_matches = all_regions.iloc[possible_matches_idx]
    tile_hexagons = possible_matches[possible_matches.intersects(bbox)].copy()

    # Ensure we have the region identifier as a column for processing
    if tile_hexagons.index.name == 'region_id':
        # SRAI format: reset index to make region_id a column
        tile_hexagons = tile_hexagons.reset_index()
        region_col = 'region_id'
    elif 'h3_index' in tile_hexagons.columns:
        # Legacy format: use h3_index column
        region_col = 'h3_index'
    else:
        # Fallback: assume index is the region identifier
        tile_hexagons = tile_hexagons.reset_index()
        region_col = tile_hexagons.index.name or 'region_id'

    # Prepare for pickling
    hexagons_to_pickle = tile_hexagons[[region_col, 'geometry']]
    hexagons_to_pickle = hexagons_to_pickle.rename(columns={region_col: 'region_id'})  # Standardize

    # Save to temporary pickle for workers
    os.makedirs(temp_dir, exist_ok=True)
    path_hash = hashlib.md5(tiff_path.encode()).hexdigest()[:10]
    pkl_path = f"{temp_dir}/tile_hexagons_{path_hash}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(hexagons_to_pickle, f)

    return pkl_path, len(tile_hexagons)


def process_chunk(args: Tuple) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """
    Worker function: Process a single chunk of a TIFF file.
    Reads windowed data, performs spatial join with pre-defined hexagons, and calculates SUM and COUNT per hexagon.
    """
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    import gc

    tiff_path, chunk_coords, tile_hexagons_pkl = args
    x_start, y_start, width, height = chunk_coords
    chunk_id = f"{os.path.basename(tiff_path)}_{x_start}_{y_start}_{width}_{height}"

    try:
        with open(tile_hexagons_pkl, 'rb') as f:
            tile_hexagons = pickle.load(f)
        tile_hexagons = tile_hexagons.set_index('region_id')

        window = Window(x_start, y_start, width, height)

        with rasterio.open(tiff_path) as src:
            chunk_data = src.read(window=window, masked=True)
            transform = src.window_transform(window)
            crs = src.crs
            num_bands = src.count

        # <<< CORRECTED LOGIC: ROBUST MASK HANDLING >>>
        # 3. Identify valid pixels (vectorized)
        # Check the mask (True means invalid data/nodata)
        if isinstance(chunk_data, np.ma.MaskedArray):
            # The mask can be a single boolean if all values are uniformly masked or unmasked.
            # This is the critical check to prevent the '0d array' error.
            if np.isscalar(chunk_data.mask):
                if chunk_data.mask:  # If scalar mask is True, all pixels are invalid
                    return {}  # Gracefully exit, no valid pixels in this chunk
                else:  # If scalar mask is False, all pixels are valid
                    invalid_mask = np.zeros((height, width), dtype=bool)
            else:
                # Mask is an array as expected. A pixel is invalid if any band is masked.
                invalid_mask = chunk_data.mask.any(axis=0)

        elif np.issubdtype(chunk_data.dtype, np.floating):
            # Handle float data with NaNs if not masked
            invalid_mask = np.isnan(chunk_data).any(axis=0)
        else:
            # Assume no invalid pixels if not a masked array or float
            invalid_mask = np.zeros((height, width), dtype=bool)

        valid_y, valid_x = np.where(~invalid_mask)

        if len(valid_y) == 0:
            return {}

        valid_data = chunk_data[:, valid_y, valid_x]
        if isinstance(valid_data, np.ma.MaskedArray):
            valid_data = valid_data.data

        world_x, world_y = transform * (valid_x + 0.5, valid_y + 0.5)

        pixel_data_transposed = valid_data.T
        df_data = {f'A{i:02d}': pixel_data_transposed[:, i] for i in range(num_bands)}

        pixels_gdf = gpd.GeoDataFrame(
            df_data,
            geometry=gpd.points_from_xy(world_x, world_y),
            crs=crs
        )

        if pixels_gdf.crs.to_epsg() != 4326:
            pixels_gdf = pixels_gdf.to_crs('EPSG:4326')

        joined = gpd.sjoin(
            pixels_gdf,
            tile_hexagons[['geometry']],
            how='inner',
            predicate='within'
        )

        if joined.empty:
            return {}

        # Group by region_id (automatically created by spatial join when using region_id index)
        if 'region_id' not in joined.columns and 'index_right' not in joined.columns:
            return {}  # No spatial matches found

        group_col = 'region_id' if 'region_id' in joined.columns else 'index_right'
        embedding_cols = [f'A{i:02d}' for i in range(num_bands)]
        grouped = joined.groupby(group_col)[embedding_cols]

        chunk_sum = grouped.sum()
        chunk_count = grouped.count().iloc[:, 0]

        chunk_results = {}
        for region_id in chunk_sum.index:
            sum_embedding = chunk_sum.loc[region_id].to_numpy(dtype=np.float64)
            count = int(chunk_count.loc[region_id])
            chunk_results[region_id] = {'sum': sum_embedding, 'count': count}

        return chunk_results

    except Exception as e:
        print(f"Error in process_chunk for {chunk_id}: {e}")
        import traceback
        traceback.print_exc()
        # Return empty dict on failure so the whole process doesn't stop
        return {}
    finally:
        if 'pixels_gdf' in locals(): del pixels_gdf
        if 'joined' in locals(): del joined
        if 'tile_hexagons' in locals(): del tile_hexagons
        gc.collect()


def process_tile_chunk_parallel(
        tiff_path: str,
        tile_hexagons_pkl: str,
        intermediate_dir: str,
        config: Dict,
        logger: logging.Logger,
        tile_id: int
) -> str:
    filename = os.path.basename(tiff_path)
    json_filename = f"{os.path.splitext(filename)[0]}.json"
    json_path = os.path.join(intermediate_dir, json_filename)
    prefix = f"[T{tile_id + 1:03d}]"

    max_workers = config['max_workers']
    chunk_size = config['chunk_size']

    if os.path.exists(json_path):
        logger.info(f"{prefix} Skipping {filename} (already processed)")
        return json_path

    start_time = time.time()

    try:
        with rasterio.open(tiff_path) as src:
            height = src.height
            width = src.width

        chunks = []
        for y_start in range(0, height, chunk_size):
            c_height = min(chunk_size, height - y_start)
            for x_start in range(0, width, chunk_size):
                c_width = min(chunk_size, width - x_start)
                chunks.append((x_start, y_start, c_width, c_height))

        logger.info(
            f"{prefix} Dimensions: {width}x{height}. Dividing into {len(chunks)} chunks. Starting {max_workers} workers.")

        task_args = [(tiff_path, chunk, tile_hexagons_pkl) for chunk in chunks]

        aggregated_results = {}
        total_pixels_processed = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results_iterator = executor.map(process_chunk, task_args)
            for chunk_result in tqdm(results_iterator, total=len(chunks), desc=f"{prefix} Chunks", unit="chunk",
                                     leave=False):
                if chunk_result:
                    for region_id, data in chunk_result.items():
                        total_pixels_processed += data['count']
                        if region_id not in aggregated_results:
                            aggregated_results[region_id] = {'sum': data['sum'], 'count': data['count']}
                        else:
                            aggregated_results[region_id]['sum'] += data['sum']
                            aggregated_results[region_id]['count'] += data['count']

        final_hexagon_data = {}
        for region_id, data in aggregated_results.items():
            mean_embedding = (data['sum'] / data['count']).tolist()
            final_hexagon_data[region_id] = {'embedding': mean_embedding, 'pixel_count': data['count']}

        os.makedirs(intermediate_dir, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(final_hexagon_data, f)

        processing_time = time.time() - start_time
        logger.info(
            f"{prefix} Finished {filename}: {len(final_hexagon_data):,} hexagons, {total_pixels_processed:,} pixels in {processing_time:.1f}s")
        return json_path

    except Exception as e:
        logger.error(f"{prefix} Failed processing {filename}: {e}", exc_info=True)
        with open(json_path, 'w') as f:
            json.dump({}, f)
        return json_path
    finally:
        if os.path.exists(tile_hexagons_pkl):
            try:
                os.remove(tile_hexagons_pkl)
            except OSError as e:
                logger.warning(f"Could not remove temp pickle {tile_hexagons_pkl}: {e}")
        gc.collect()


def find_tiff_files(tiff_path: str, pattern: str = "*.tif*", year_filter: Optional[str] = None) -> List[str]:
    path = Path(tiff_path)
    if path.is_file(): return [str(path)]
    if not path.exists(): raise FileNotFoundError(f"TIFF path not found: {tiff_path}")
    tiff_files = path.rglob(pattern)
    if year_filter:
        tiff_files = (f for f in tiff_files if year_filter in f.name)
    sorted_files = sorted([str(f) for f in tiff_files if f.is_file()])
    if not sorted_files:
        print(f"Warning: No TIFF files found in {tiff_path} matching criteria.")
    return sorted_files


def merge_results_with_regions(
        intermediate_dir: str,
        regions_gdf: gpd.GeoDataFrame,
        logger: logging.Logger
) -> gpd.GeoDataFrame:
    logger.info("=" * 60)
    logger.info("STEP 2: MERGING RESULTS (Weighted Averaging)")
    logger.info("=" * 60)
    json_files = list(Path(intermediate_dir).glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to merge")
    if not json_files: return gpd.GeoDataFrame()

    merged_sums = {}
    merged_counts = {}
    tile_counts = {}
    num_bands = None

    for json_file in tqdm(json_files, desc="Merging JSON files"):
        try:
            with open(json_file, 'r') as f:
                tile_data = json.load(f)
            for region_id, data in tile_data.items():
                if not data or 'embedding' not in data: continue
                if num_bands is None: num_bands = len(data['embedding'])
                embedding = np.array(data['embedding'], dtype=np.float64)
                pixel_count = data['pixel_count']
                current_sum = embedding * pixel_count
                if region_id not in merged_sums:
                    merged_sums[region_id] = current_sum
                    merged_counts[region_id] = pixel_count
                    tile_counts[region_id] = 1
                else:
                    merged_sums[region_id] += current_sum
                    merged_counts[region_id] += pixel_count
                    tile_counts[region_id] += 1
        except Exception as e:
            logger.warning(f"Error processing {json_file}: {e}")

    if not merged_sums: return gpd.GeoDataFrame()

    logger.info("Calculating final weighted means...")
    final_data = []
    for region_id in merged_sums:
        weighted_mean = merged_sums[region_id] / merged_counts[region_id]
        row_data = {'region_id': region_id}
        for i in range(num_bands):
            row_data[f'A{i:02d}'] = float(weighted_mean[i])
        row_data['pixel_count'] = merged_counts[region_id]
        row_data['tile_count'] = tile_counts[region_id]
        final_data.append(row_data)

    df = pd.DataFrame(final_data)
    logger.info(f"Merged {len(df):,} unique hexagons.")

    # Handle both SRAI (region_id) and legacy (h3_index) formats
    if regions_gdf.index.name == 'region_id':
        regions_gdf = regions_gdf.reset_index()
        merge_col = 'region_id'
    elif 'h3_index' in regions_gdf.columns:
        merge_col = 'h3_index'
        # Rename for consistency
        df = df.rename(columns={'region_id': 'h3_index'})
    else:
        regions_gdf = regions_gdf.reset_index()
        merge_col = 'region_id'

    logger.info("Adding geometries from SRAI regions...")
    result = regions_gdf[[merge_col, 'geometry']].merge(df, on=merge_col, how='inner')
    return gpd.GeoDataFrame(result, crs='EPSG:4326')


def validate_and_save(gdf: gpd.GeoDataFrame, metadata: Dict, study_area: str, h3_resolution: int,
                      year_filter: Optional[str], logger: logging.Logger) -> bool:
    logger.info("Validating results...")
    if len(gdf) == 0:
        logger.error("Validation Failed: No data generated.")
        return False
    EXPECTED_DIMS = 64
    embedding_cols = [c for c in gdf.columns if c.startswith('A') and len(c) == 3]
    if len(embedding_cols) != EXPECTED_DIMS:
        logger.error(f"Validation Failed: Expected {EXPECTED_DIMS} dimensions, got {len(embedding_cols)}")
        return False
    if gdf[embedding_cols].isna().any().any():
        logger.error("Validation Failed: NaN values found in embeddings.")
        return False

    logger.info("Validation passed. Saving results...")
    output_dir = f"data/study_areas/{study_area}/embeddings/alphaearth"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    year_suffix = f"_{year_filter}" if year_filter else ""
    base_name = f"{study_area}_res{h3_resolution}{year_suffix}_{timestamp}"
    parquet_file = f"{output_dir}/{base_name}.parquet"
    metadata_file = f"{output_dir}/{base_name}.json"

    gdf[embedding_cols] = gdf[embedding_cols].astype('float32')
    gdf['pixel_count'] = gdf['pixel_count'].astype('int32')
    gdf['tile_count'] = gdf['tile_count'].astype('int16')

    gdf.to_parquet(parquet_file, index=False)
    logger.info(f"Saved embeddings to: {parquet_file}")

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_file}")
    return True


def cleanup(intermediate_dir: str, temp_dir: str, logger: logging.Logger, keep_intermediate: bool):
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary run directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up temp directory {temp_dir}: {e}")
    if not keep_intermediate:
        logger.info(f"Cleaning up intermediate embeddings stage1_modalities directory: {intermediate_dir}...")
        try:
            if os.path.exists(intermediate_dir):
                shutil.rmtree(intermediate_dir)
                logger.info("Intermediate directory removed.")
        except Exception as e:
            logger.warning(f"Could not clean up intermediate embeddings stage1_modalities directory {intermediate_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Optimized AlphaEarth Processing (Sequential Tile, Parallel Chunk)')
    parser.add_argument('--study-area', required=True, help='Study area name (e.g., netherlands)')
    parser.add_argument('--tiff-path', required=True, help='Path to TIFF files directory')
    parser.add_argument('--resolution', type=int, required=True, help='H3 resolution')
    parser.add_argument('--pattern', default='*.tif*', help='TIFF file pattern (default: *.tif*)')
    parser.add_argument('--year-filter', help='Optional year filter (e.g., 2023)')
    parser.add_argument('--max-workers', type=int, help='Max parallel workers (default: auto-detect)')
    parser.add_argument('--chunk-size', type=int, help='Chunk size in pixels for performance tuning (default: 1024)')
    parser.add_argument('--keep-intermediate', action='store_true',
                        help='Keep intermediate embeddings stage1_modalities JSON files')
    parser.add_argument('--skip-step1', action='store_true',
                        help='Skip Step 1 (use existing intermediate embeddings stage1_modalities files)')
    parser.add_argument('--debug-json', action='store_true', help='Enable extensive JSON debugging for error analysis')
    args = parser.parse_args()

    log_dir = "logs/alphaearth"
    logger = setup_logging(log_dir, args.study_area, args.resolution)
    global DEBUG_ENABLED
    DEBUG_ENABLED = args.debug_json
    if DEBUG_ENABLED: logger.info("DEBUG JSON MODE ENABLED")

    logger.info("=" * 80)
    logger.info("STARTING ALPHAEARTH PROCESSING (Sequential Tile, Parallel Chunk)")
    logger.info("=" * 80)
    start_time = time.time()
    temp_dir, intermediate_dir = None, None

    try:
        config = get_optimized_config(args.max_workers)
        if args.chunk_size:
            config['chunk_size'] = args.chunk_size
        logger.info(f"Configuration: Workers={config['max_workers']}, Chunk Size={config['chunk_size']}")

        year_suffix = f"_{args.year_filter}" if args.year_filter else ""
        intermediate_dir = f"data/study_areas/{args.study_area}/embeddings/intermediate/alphaearth/res{args.resolution}{year_suffix}"
        run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        temp_dir = f"data/study_areas/temp/run_{run_id}"

        regions_file = f"data/study_areas/{args.study_area}/regions_gdf/h3_res{args.resolution}.parquet"
        if not os.path.exists(regions_file):
            raise FileNotFoundError(f"SRAI regions not found: {regions_file}")

        logger.info("Loading SRAI regions and building spatial index...")
        all_regions = gpd.read_parquet(regions_file)
        if all_regions.crs is None or all_regions.crs.to_epsg() != 4326:
            all_regions = all_regions.to_crs("EPSG:4326")
        all_regions.sindex  # Force spatial index creation
        logger.info(f"Loaded {len(all_regions):,} hexagons.")

        tiff_files = find_tiff_files(args.tiff_path, args.pattern, args.year_filter)
        logger.info(f"Found {len(tiff_files)} TIFF files.")
        if not tiff_files: return

        if not args.skip_step1:
            logger.info("=" * 60)
            logger.info("STEP 1: TILE PROCESSING")
            logger.info("=" * 60)
            os.makedirs(intermediate_dir, exist_ok=True)

            # TEMPORARY: Select substantial tiles for testing (includes large urban areas)
            if DEBUG_ENABLED:
                # Choose tiles that are likely to contain urban data (large files)
                test_files = []
                large_tiles = ['0020-0000000000-0000000000', '0021-0000000000-0000000000', '0023-0000000000-0000000000']
                for tiff_file in tiff_files:
                    if any(large_tile in tiff_file for large_tile in large_tiles):
                        test_files.append(tiff_file)
                        if len(test_files) >= 3:
                            break
                # If we didn't find large tiles, use first 3
                if not test_files:
                    test_files = tiff_files[:3]
                test_files = test_files[:3]  # Limit to 3 for testing
            else:
                test_files = tiff_files

            for i, tiff_file in enumerate(tqdm(test_files, desc="Processing Tiles", unit="tile")):
                logger.info(f"\n--- Tile {i + 1}/{len(tiff_files)}: {os.path.basename(tiff_file)} ---")
                try:
                    pkl_path, n_hexagons = get_tile_hexagons(tiff_file, all_regions, temp_dir)
                except Exception as e:
                    logger.error(f"  Failed to get hexagons for {tiff_file}: {e}. Skipping tile.")
                    continue

                if n_hexagons == 0:
                    logger.info("  No relevant hexagons found for this tile. Creating empty result.")
                    json_filename = f"{os.path.splitext(os.path.basename(tiff_file))[0]}.json"
                    json_path = os.path.join(intermediate_dir, json_filename)
                    with open(json_path, 'w') as f:
                        json.dump({}, f)
                    if 'pkl_path' in locals() and os.path.exists(pkl_path): os.remove(pkl_path)
                    continue

                process_tile_chunk_parallel(tiff_file, pkl_path, intermediate_dir, config, logger, i)

            logger.info("Step 1 complete.")
        else:
            logger.info("Skipping Step 1.")

        gc.collect()
        result_gdf = merge_results_with_regions(intermediate_dir, all_regions, logger)

        processing_time = time.time() - start_time
        metadata = {
            'study_area': args.study_area, 'h3_resolution': args.resolution,
            'processing_duration_minutes': round(processing_time / 60, 2),
            'hexagon_count': len(result_gdf),
            'method': 'sequential_tile_parallel_chunk_vectorized',
            'configuration': config
        }
        success = validate_and_save(result_gdf, metadata, args.study_area, args.resolution, args.year_filter, logger)
        if success:
            logger.info(f"\nPROCESSING COMPLETE SUCCESSFULLY. Total time: {processing_time / 60:.2f} minutes.")
        else:
            logger.error("Processing finished with errors.")
    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)
    finally:
        if DEBUG_ENABLED:
            debug_file = f"debug_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_debug_json(debug_file)
        cleanup(intermediate_dir, temp_dir, logger, args.keep_intermediate)


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()