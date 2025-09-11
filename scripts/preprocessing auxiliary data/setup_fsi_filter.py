"""
FSI filtering script for UrbanRepML preprocessing auxiliary data.
Filters H3 regions based on FSI (Floor Space Index) thresholds.

This script:
1. Loads regions with density data from input directory
2. Applies FSI filtering (absolute value or percentile)
3. Maintains hierarchical relationships (if res-8 selected, all children included)
4. Saves filtered regions to output directory
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Set, List
import argparse

import pandas as pd
import geopandas as gpd
import numpy as np
import h3

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_complete_data(input_dir: Path, city_name: str, resolutions: List[int]) -> Dict[int, gpd.GeoDataFrame]:
    """Load complete region data with density information."""
    data_dict = {}
    
    for resolution in resolutions:
        data_path = input_dir / "total" / f"{city_name}_res{resolution}_complete.parquet"
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Complete data file not found: {data_path}\n"
                "Please run setup_density.py first!"
            )
        
        print(f"[LOAD] Loading complete data for resolution {resolution}...")
        data_gdf = gpd.read_parquet(data_path)
        print(f"[OK] Loaded {len(data_gdf)} regions at resolution {resolution}")
        
        # Ensure FSI_24 column exists
        if 'FSI_24' not in data_gdf.columns:
            raise ValueError(f"FSI_24 column not found in {data_path}")
        
        data_dict[resolution] = data_gdf
    
    return data_dict


def calculate_fsi_threshold(data_dict: Dict[int, gpd.GeoDataFrame], 
                           fsi_percentile: float = None,
                           fsi_threshold: float = None) -> float:
    """Calculate FSI threshold value from percentile or use absolute value."""
    
    if fsi_threshold is not None:
        print(f"[THRESHOLD] Using absolute FSI threshold: {fsi_threshold}")
        return fsi_threshold
    
    if fsi_percentile is not None:
        # Use resolution 8 data for percentile calculation
        if 8 in data_dict:
            res8_data = data_dict[8]
        else:
            # Use the lowest resolution available
            min_res = min(data_dict.keys())
            res8_data = data_dict[min_res]
            print(f"[WARN] Resolution 8 not found, using resolution {min_res} for percentile calculation")
        
        # Filter to regions with FSI > 0
        valid_fsi = res8_data[res8_data['FSI_24'] > 0]['FSI_24']
        
        if len(valid_fsi) == 0:
            raise ValueError("No regions with FSI > 0 found")
        
        threshold_value = np.percentile(valid_fsi, fsi_percentile)
        print(f"[THRESHOLD] Calculated {fsi_percentile}th percentile FSI threshold: {threshold_value:.4f}")
        print(f"   Based on {len(valid_fsi)} regions with FSI > 0")
        print(f"   FSI range: [{valid_fsi.min():.4f}, {valid_fsi.max():.4f}]")
        print(f"   FSI mean: {valid_fsi.mean():.4f}, median: {valid_fsi.median():.4f}")
        
        return threshold_value
    
    raise ValueError("Either fsi_threshold or fsi_percentile must be specified")


def filter_hierarchically(data_dict: Dict[int, gpd.GeoDataFrame], 
                         fsi_threshold: float,
                         base_resolution: int = 8) -> Dict[int, Set[str]]:
    """
    Filter regions hierarchically based on FSI threshold.
    If a parent hex meets the threshold, all its children are included.
    """
    
    selected_hexes = {res: set() for res in data_dict.keys()}
    
    # First, select base resolution hexagons that meet the threshold
    if base_resolution in data_dict:
        base_data = data_dict[base_resolution]
        selected_base = base_data[base_data['FSI_24'] >= fsi_threshold]
        selected_hexes[base_resolution] = set(selected_base.index)
        
        print(f"\n[FILTER] Resolution {base_resolution} (base):")
        print(f"   Selected {len(selected_hexes[base_resolution])} / {len(base_data)} hexagons")
        print(f"   FSI >= {fsi_threshold:.4f}")
        
        # For each selected base hex, include all children
        for base_hex in selected_hexes[base_resolution]:
            # Get children at resolution 9
            if base_resolution + 1 in data_dict:
                children_9 = list(h3.cell_to_children(base_hex, base_resolution + 1))
                selected_hexes[base_resolution + 1].update(children_9)
                
                # Get grandchildren at resolution 10
                if base_resolution + 2 in data_dict:
                    for child_9 in children_9:
                        grandchildren_10 = list(h3.cell_to_children(child_9, base_resolution + 2))
                        selected_hexes[base_resolution + 2].update(grandchildren_10)
        
        # Report statistics for child resolutions
        for res in sorted(data_dict.keys()):
            if res > base_resolution and res in selected_hexes:
                total_hexes = len(data_dict[res])
                selected_count = len(selected_hexes[res])
                print(f"   Resolution {res}: {selected_count} / {total_hexes} hexagons (children of selected parents)")
    
    else:
        # If base resolution not available, filter each resolution independently
        print(f"[WARN] Base resolution {base_resolution} not found, filtering independently")
        
        for resolution, gdf in data_dict.items():
            selected = gdf[gdf['FSI_24'] >= fsi_threshold]
            selected_hexes[resolution] = set(selected.index)
            
            print(f"[FILTER] Resolution {resolution}:")
            print(f"   Selected {len(selected_hexes[resolution])} / {len(gdf)} hexagons")
            print(f"   FSI >= {fsi_threshold:.4f}")
    
    return selected_hexes


def create_parent_child_mappings(selected_hexes: Dict[int, Set[str]]) -> tuple:
    """Create mappings between parent and child hexagons."""
    
    mappings = {}
    
    # Create mappings between consecutive resolutions
    resolutions = sorted(selected_hexes.keys())
    
    for i in range(len(resolutions) - 1):
        lower_res = resolutions[i]
        higher_res = resolutions[i + 1]
        
        if higher_res == lower_res + 1:  # Only for consecutive resolutions
            mapping_name = f"mapping_{higher_res}_to_{lower_res}"
            parent_to_children = {}
            
            for parent_hex in selected_hexes[lower_res]:
                children = [h for h in h3.cell_to_children(parent_hex, higher_res) 
                           if h in selected_hexes[higher_res]]
                if children:
                    parent_to_children[parent_hex] = children
            
            mappings[mapping_name] = parent_to_children
            
            print(f"[MAPPING] Created {mapping_name}:")
            print(f"   {len(parent_to_children)} parent hexagons with children")
            total_children = sum(len(children) for children in parent_to_children.values())
            print(f"   {total_children} total child hexagons")
    
    return mappings


def save_filtered_data(data_dict: Dict[int, gpd.GeoDataFrame],
                       selected_hexes: Dict[int, Set[str]],
                       mappings: dict,
                       city_name: str,
                       output_dir: Path,
                       experiment_name: str):
    """Save filtered region data and mappings."""
    
    # Create output directories
    (output_dir / "regions").mkdir(parents=True, exist_ok=True)
    (output_dir / "density").mkdir(parents=True, exist_ok=True)
    (output_dir / "boundaries").mkdir(parents=True, exist_ok=True)
    
    # Save filtered regions for each resolution
    for resolution, gdf in data_dict.items():
        if resolution in selected_hexes:
            # Filter to selected hexagons
            filtered_gdf = gdf[gdf.index.isin(selected_hexes[resolution])].copy()
            
            # Save regions
            regions_path = output_dir / "regions" / f"regions_{resolution}_gdf.parquet"
            filtered_gdf[['geometry', 'region_id']].to_parquet(regions_path)
            print(f"[SAVE] Regions for resolution {resolution}: {regions_path}")
            
            # Save density data
            density_df = pd.DataFrame({
                'FSI_24': filtered_gdf['FSI_24'],
                'in_study_area': filtered_gdf.get('in_study_area', True),
                'building_volume': filtered_gdf.get('building_volume', 0),
                'total_area': filtered_gdf.get('total_area', 0)
            }, index=filtered_gdf.index)
            
            density_path = output_dir / "density" / f"building_density_res{resolution}_preprocessed.parquet"
            density_df.to_parquet(density_path)
            print(f"[SAVE] Density for resolution {resolution}: {density_path}")
    
    # Save area boundary (union of all selected regions at base resolution)
    if 8 in data_dict and 8 in selected_hexes:
        base_gdf = data_dict[8][data_dict[8].index.isin(selected_hexes[8])]
        area_boundary = gpd.GeoDataFrame(
            [{'geometry': base_gdf.unary_union, 'name': experiment_name}],
            crs=base_gdf.crs
        )
    else:
        # Use lowest resolution available
        min_res = min(selected_hexes.keys())
        base_gdf = data_dict[min_res][data_dict[min_res].index.isin(selected_hexes[min_res])]
        area_boundary = gpd.GeoDataFrame(
            [{'geometry': base_gdf.unary_union, 'name': experiment_name}],
            crs=base_gdf.crs
        )
    
    boundary_path = output_dir / "boundaries" / "area_study_gdf.parquet"
    area_boundary.to_parquet(boundary_path)
    print(f"[SAVE] Area boundary: {boundary_path}")
    
    # Save mappings as pickle
    if mappings:
        import pickle
        mappings_path = output_dir / "mappings.pkl"
        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings, f)
        print(f"[SAVE] Parent-child mappings: {mappings_path}")
    
    # Save metadata
    metadata = {
        'city_name': city_name,
        'experiment_name': experiment_name,
        'total_hexagons': {res: len(hexes) for res, hexes in selected_hexes.items()},
        'fsi_statistics': {}
    }
    
    for resolution in selected_hexes.keys():
        if resolution in data_dict:
            filtered_data = data_dict[resolution][data_dict[resolution].index.isin(selected_hexes[resolution])]
            metadata['fsi_statistics'][resolution] = {
                'min': float(filtered_data['FSI_24'].min()),
                'max': float(filtered_data['FSI_24'].max()),
                'mean': float(filtered_data['FSI_24'].mean()),
                'median': float(filtered_data['FSI_24'].median())
            }
    
    import json
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVE] Metadata: {metadata_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Filter H3 regions by FSI threshold')
    parser.add_argument('--city_name', type=str, default='south_holland',
                        help='Name of the city/region to process')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with complete region/density data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for filtered data')
    parser.add_argument('--fsi_threshold', type=float, default=None,
                        help='Absolute FSI threshold value (e.g., 0.1)')
    parser.add_argument('--fsi_percentile', type=float, default=None,
                        help='FSI percentile threshold (e.g., 95 for 95th percentile)')
    parser.add_argument('--resolutions', type=str, default='8,9,10',
                        help='Comma-separated H3 resolutions to process')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment/filtering')
    parser.add_argument('--base_resolution', type=int, default=8,
                        help='Base resolution for hierarchical filtering')
    return parser.parse_args()


def main():
    """Main FSI filtering workflow."""
    
    args = parse_args()
    
    # Validate arguments
    if args.fsi_threshold is None and args.fsi_percentile is None:
        raise ValueError("Either --fsi_threshold or --fsi_percentile must be specified")
    
    if args.fsi_threshold is not None and args.fsi_percentile is not None:
        raise ValueError("Only one of --fsi_threshold or --fsi_percentile can be specified")
    
    print("==== UrbanRepML FSI Filtering ====")
    print("=" * 40)
    
    # Parse resolutions
    resolutions = [int(r) for r in args.resolutions.split(',')]
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        if args.fsi_threshold is not None:
            args.experiment_name = f"{args.city_name}_fsi{args.fsi_threshold:.2f}"
        else:
            args.experiment_name = f"{args.city_name}_fsi_p{args.fsi_percentile:.0f}"
    
    print(f"[CONFIG] Experiment: {args.experiment_name}")
    print(f"[CONFIG] Input: {input_dir}")
    print(f"[CONFIG] Output: {output_dir}")
    print(f"[CONFIG] Resolutions: {resolutions}")
    
    # Load complete data
    data_dict = load_complete_data(input_dir, args.city_name, resolutions)
    
    # Calculate FSI threshold
    fsi_threshold_value = calculate_fsi_threshold(
        data_dict,
        fsi_percentile=args.fsi_percentile,
        fsi_threshold=args.fsi_threshold
    )
    
    # Filter hierarchically
    selected_hexes = filter_hierarchically(
        data_dict,
        fsi_threshold_value,
        base_resolution=args.base_resolution
    )
    
    # Create parent-child mappings
    mappings = create_parent_child_mappings(selected_hexes)
    
    # Save filtered data
    save_filtered_data(
        data_dict,
        selected_hexes,
        mappings,
        args.city_name,
        output_dir,
        args.experiment_name
    )
    
    print("\n[SUCCESS] FSI filtering completed!")
    print(f"[OUTPUT] Filtered data saved to: {output_dir}")
    print("\n[SUMMARY] Selected hexagons by resolution:")
    for res in sorted(selected_hexes.keys()):
        print(f"   Resolution {res}: {len(selected_hexes[res])} hexagons")
    
    print("\n[NEXT] Next steps:")
    print(f"   Run setup_hierarchical_graphs.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()