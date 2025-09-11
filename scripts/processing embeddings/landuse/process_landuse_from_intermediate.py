"""
Process landuse embeddings from saved intermediate data.
This script uses pre-downloaded OSM data to generate count and diversity embeddings.
Hex2Vec and GeoVex can be run separately to avoid timeouts.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# SRAI imports
from srai.embedders import CountEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_intermediate_data(intermediate_dir: Path, study_area_name: str, h3_resolution: int):
    """Load saved intermediate data from previous run."""
    base_name = f"{study_area_name}_res{h3_resolution}"
    
    features_path = intermediate_dir / f"{base_name}_features.parquet"
    regions_path = intermediate_dir / f"{base_name}_regions.parquet"
    joint_path = intermediate_dir / f"{base_name}_joint.parquet"
    
    logger.info(f"Loading intermediate data from {intermediate_dir}")
    
    if not all([features_path.exists(), regions_path.exists(), joint_path.exists()]):
        missing = [p for p in [features_path, regions_path, joint_path] if not p.exists()]
        raise FileNotFoundError(f"Missing intermediate files: {missing}")
    
    # Try to load as GeoDataFrame first, fall back to DataFrame if needed
    try:
        features_gdf = gpd.read_parquet(features_path)
    except:
        features_df = pd.read_parquet(features_path)
        features_gdf = gpd.GeoDataFrame(features_df)
    
    try:
        regions_gdf = gpd.read_parquet(regions_path)
    except:
        regions_df = pd.read_parquet(regions_path)
        regions_gdf = gpd.GeoDataFrame(regions_df)
    
    try:
        joint_gdf = gpd.read_parquet(joint_path)
    except:
        joint_df = pd.read_parquet(joint_path)
        joint_gdf = gpd.GeoDataFrame(joint_df)
    
    logger.info(f"Loaded features: {len(features_gdf)} landuse polygons")
    logger.info(f"Loaded regions: {len(regions_gdf)} H3 hexagons")
    logger.info(f"Loaded joint: {len(joint_gdf)} polygon-hexagon pairs")
    
    return features_gdf, regions_gdf, joint_gdf


def calculate_diversity_metrics(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate landuse diversity metrics."""
    # Get landuse columns (exclude h3_index)
    landuse_cols = [col for col in counts_df.columns 
                   if col != 'h3_index' and not col.startswith('h3_')]
    
    if not landuse_cols:
        return pd.DataFrame()

    counts = counts_df[landuse_cols].fillna(0)
    total = counts.sum(axis=1)
    
    # Avoid division by zero
    proportions = counts.divide(total.replace(0, 1), axis=0)

    # Shannon entropy
    shannon_entropy = -1 * (proportions * np.log(proportions.replace(0, np.nan))).sum(axis=1, skipna=True)
    
    # Simpson diversity
    simpson_diversity = 1 - (proportions**2).sum(axis=1)
    
    # Richness (number of different landuse types)
    richness = (counts > 0).sum(axis=1)
    
    # Evenness
    max_entropy = np.log(richness.replace(0, 1))
    evenness = shannon_entropy / max_entropy.replace(0, 1)
    
    # Dominant landuse
    dominant_landuse = counts.idxmax(axis=1)
    dominant_landuse[total == 0] = 'none'
    
    # Calculate landuse mix ratios
    urban_cols = [col for col in landuse_cols if any(x in col for x in ['residential', 'commercial', 'industrial', 'retail'])]
    rural_cols = [col for col in landuse_cols if any(x in col for x in ['farmland', 'meadow', 'orchard', 'vineyard', 'grass'])]
    natural_cols = [col for col in landuse_cols if any(x in col for x in ['forest', 'wood', 'water', 'wetland', 'natural'])]
    
    urban_ratio = counts[urban_cols].sum(axis=1) / total.replace(0, 1) if urban_cols else pd.Series([0] * len(counts_df))
    rural_ratio = counts[rural_cols].sum(axis=1) / total.replace(0, 1) if rural_cols else pd.Series([0] * len(counts_df))
    natural_ratio = counts[natural_cols].sum(axis=1) / total.replace(0, 1) if natural_cols else pd.Series([0] * len(counts_df))

    return pd.DataFrame({
        'landuse_shannon_entropy': shannon_entropy.fillna(0),
        'landuse_simpson_diversity': simpson_diversity.fillna(0),
        'landuse_richness': richness,
        'landuse_evenness': evenness.fillna(0),
        'dominant_landuse': dominant_landuse,
        'total_landuse_area': total,
        'urban_ratio': urban_ratio,
        'rural_ratio': rural_ratio,
        'natural_ratio': natural_ratio
    })


def process_counts_and_diversity(features_gdf, regions_gdf, joint_gdf, study_area_name, h3_resolution):
    """Process count embeddings and diversity metrics from intermediate data."""
    
    logger.info("Computing count-based embeddings...")
    count_embedder = CountEmbedder()
    embeddings_gdf = count_embedder.transform(
        regions_gdf=regions_gdf,
        features_gdf=features_gdf,
        joint_gdf=joint_gdf
    )
    logger.info(f"Count embeddings complete: {embeddings_gdf.shape[1]-1} feature columns")
    
    # Convert to DataFrame and prepare counts
    counts_df = pd.DataFrame(embeddings_gdf.drop(columns='geometry', errors='ignore'))
    
    # Ensure h3_index column exists
    if counts_df.index.name == 'region_id':
        counts_df = counts_df.reset_index()
        counts_df = counts_df.rename(columns={'region_id': 'h3_index'})
    elif 'h3_index' not in counts_df.columns:
        counts_df = counts_df.reset_index()
        counts_df = counts_df.rename(columns={'index': 'h3_index'})
    
    counts_df['h3_index'] = counts_df['h3_index'].astype(str)
    
    # Calculate diversity metrics
    logger.info("Calculating diversity metrics...")
    diversity_df = calculate_diversity_metrics(counts_df)
    diversity_df['h3_index'] = counts_df['h3_index']
    
    # Reorder columns to have h3_index first
    cols = ['h3_index'] + [col for col in diversity_df.columns if col != 'h3_index']
    diversity_df = diversity_df[cols]
    
    # Log statistics
    logger.info("\nLanduse statistics:")
    landuse_cols = [col for col in counts_df.columns if col != 'h3_index']
    logger.info(f"  Total landuse types found: {len(landuse_cols)}")
    
    if landuse_cols:
        # Show top 10 most common landuse types
        totals = counts_df[landuse_cols].sum().sort_values(ascending=False)
        logger.info("  Top 10 landuse types by coverage:")
        for i, (landuse, count) in enumerate(totals.head(10).items(), 1):
            logger.info(f"    {i}. {landuse}: {count:.0f} hexagons")
    
    logger.info(f"\nDiversity metrics summary:")
    logger.info(f"  Mean richness: {diversity_df['landuse_richness'].mean():.2f}")
    logger.info(f"  Mean Shannon entropy: {diversity_df['landuse_shannon_entropy'].mean():.3f}")
    logger.info(f"  Urban ratio: {diversity_df['urban_ratio'].mean():.1%}")
    logger.info(f"  Rural ratio: {diversity_df['rural_ratio'].mean():.1%}")
    logger.info(f"  Natural ratio: {diversity_df['natural_ratio'].mean():.1%}")
    
    return counts_df, diversity_df


def save_outputs(counts_df, diversity_df, output_dir, study_area_name, h3_resolution):
    """Save count and diversity embeddings to separate parquet files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define file names
    counts_path = Path(output_dir) / f'landuse_counts_{study_area_name}_res{h3_resolution}.parquet'
    diversity_path = Path(output_dir) / f'landuse_diversity_{study_area_name}_res{h3_resolution}.parquet'
    
    # Save files
    counts_df.to_parquet(counts_path, index=False)
    diversity_df.to_parquet(diversity_path, index=False)
    
    logger.info(f"\nSaved outputs:")
    logger.info(f"  Counts: {counts_path}")
    logger.info(f"  Diversity: {diversity_path}")
    
    return counts_path, diversity_path


def verify_outputs(counts_path, diversity_path):
    """Verify output file structure."""
    logger.info("\nVerifying output files...")
    
    # Check counts
    counts_df = pd.read_parquet(counts_path)
    logger.info(f"\nCounts file:")
    logger.info(f"  Shape: {counts_df.shape}")
    logger.info(f"  Has h3_index: {'h3_index' in counts_df.columns}")
    logger.info(f"  Sample columns: {counts_df.columns[:10].tolist()}")
    
    # Check diversity
    diversity_df = pd.read_parquet(diversity_path)
    logger.info(f"\nDiversity file:")
    logger.info(f"  Shape: {diversity_df.shape}")
    logger.info(f"  Has h3_index: {'h3_index' in diversity_df.columns}")
    logger.info(f"  Columns: {diversity_df.columns.tolist()}")
    
    # Verify H3 index validity
    if 'h3_index' in counts_df.columns:
        import h3
        sample_h3 = counts_df['h3_index'].iloc[0]
        try:
            resolution = h3.h3_get_resolution(sample_h3)
            logger.info(f"  H3 resolution verified: {resolution}")
        except:
            logger.warning(f"  Could not validate H3 index: {sample_h3}")


def main():
    """Main execution."""
    logger.info("="*60)
    logger.info("Processing landuse embeddings from intermediate data")
    logger.info("="*60)
    
    # Configuration
    intermediate_dir = Path("data/intermediate/landuse")
    output_dir = Path("data/processed/embeddings/landuse")
    study_area_name = "netherlands_test"  # Change to "netherlands" for full data
    h3_resolution = 10
    
    try:
        # Load intermediate data
        features_gdf, regions_gdf, joint_gdf = load_intermediate_data(
            intermediate_dir, study_area_name, h3_resolution
        )
        
        # Process counts and diversity
        counts_df, diversity_df = process_counts_and_diversity(
            features_gdf, regions_gdf, joint_gdf, study_area_name, h3_resolution
        )
        
        # Save outputs
        counts_path, diversity_path = save_outputs(
            counts_df, diversity_df, output_dir, study_area_name, h3_resolution
        )
        
        # Verify outputs
        verify_outputs(counts_path, diversity_path)
        
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Run separate hex2vec training script if needed")
        logger.info("2. Run separate geovex training script if needed")
        logger.info("3. Use these embeddings for multi-modal fusion")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()