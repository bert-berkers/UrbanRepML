"""
Embedding Validation

Validates embeddings and checks compatibility across modalities.

Checks:
1. H3 alignment between modalities
2. Feature dimensions and data types
3. Coverage statistics
4. Intermediate file validation
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
# Low-level H3 cell primitives not wrapped by SRAI
import h3 as _h3
from typing import Dict, List, Tuple, Optional
import numpy as np

from utils import StudyAreaPaths

logger = logging.getLogger(__name__)


class EmbeddingValidator:
    """Validate embeddings across different modalities."""

    def __init__(
        self,
        base_dir: str = "data/processed",
        paths: Optional[StudyAreaPaths] = None,
    ):
        # NOTE: This class uses legacy directory layout (data/processed/).
        # When a StudyAreaPaths is provided, paths.root is used instead.
        if paths is not None:
            self.base_dir = paths.root
        else:
            self.base_dir = Path(base_dir)
        self.embeddings_dir = self.base_dir / "embeddings"
        self.intermediate_dir = self.base_dir / "intermediate embeddings stage1_modalities"

    def validate_modality(self, modality: str, resolution: int = 10) -> Dict:
        """Validate a single modality's embeddings."""
        logger.info(f"Validating {modality} embeddings at resolution {resolution}")

        results = {
            'modality': modality,
            'resolution': resolution,
            'embeddings_found': False,
            'intermediate_found': False,
            'stats': {}
        }

        # Check embeddings file
        embeddings_path = self.embeddings_dir / modality / f"{modality}_embeddings_res{resolution}.parquet"
        if embeddings_path.exists():
            results['embeddings_found'] = True
            df = pd.read_parquet(embeddings_path)

            results['stats'] = {
                'n_hexagons': len(df),
                'n_features': df.shape[1] - 2,  # Minus h3_index and resolution
                'h3_cells': df['h3_index'].nunique() if 'h3_index' in df.columns else 0,
                'columns': list(df.columns)[:10],
                'missing_values': df.isnull().sum().sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }

            # Validate H3 indices
            if 'h3_index' in df.columns:
                sample_indices = df['h3_index'].head(5).tolist()
                h3_valid = all(_h3.is_valid_cell(idx) for idx in sample_indices)
                results['stats']['h3_valid'] = h3_valid

                resolutions = [_h3.get_resolution(idx) for idx in sample_indices]
                results['stats']['h3_resolutions'] = list(set(resolutions))

        # Check intermediate data
        intermediate_path = self.intermediate_dir / modality
        if intermediate_path.exists():
            results['intermediate_found'] = True

            features_count = len(list((intermediate_path / 'features_gdf').glob('*.parquet'))) if (intermediate_path / 'features_gdf').exists() else 0
            regions_count = len(list((intermediate_path / 'regions_gdf').glob('*.parquet'))) if (intermediate_path / 'regions_gdf').exists() else 0
            joint_count = len(list((intermediate_path / 'joint_gdf').glob('*.parquet'))) if (intermediate_path / 'joint_gdf').exists() else 0

            results['intermediate_stats'] = {
                'features_gdf_files': features_count,
                'regions_gdf_files': regions_count,
                'joint_gdf_files': joint_count
            }

        return results

    def compare_modalities(self, modalities: List[str], resolution: int = 10) -> Dict:
        """Compare H3 coverage and compatibility across modalities."""
        logger.info(f"Comparing {len(modalities)} modalities at resolution {resolution}")

        comparison = {
            'resolution': resolution,
            'modalities': {},
            'overlap': {},
            'alignment': {}
        }

        h3_sets = {}
        dataframes = {}

        for modality in modalities:
            embeddings_path = self.embeddings_dir / modality / f"{modality}_embeddings_res{resolution}.parquet"

            if embeddings_path.exists():
                df = pd.read_parquet(embeddings_path)
                dataframes[modality] = df

                if 'h3_index' in df.columns:
                    h3_sets[modality] = set(df['h3_index'])
                    comparison['modalities'][modality] = {
                        'n_hexagons': len(h3_sets[modality]),
                        'n_features': df.shape[1] - 2
                    }
                else:
                    logger.warning(f"{modality} missing h3_index column")
            else:
                logger.warning(f"{modality} embeddings not found at {embeddings_path}")

        # Calculate overlaps
        if len(h3_sets) >= 2:
            modality_pairs = [
                (m1, m2) for i, m1 in enumerate(modalities)
                for m2 in modalities[i+1:] if m1 in h3_sets and m2 in h3_sets
            ]

            for m1, m2 in modality_pairs:
                overlap = h3_sets[m1].intersection(h3_sets[m2])
                union = h3_sets[m1].union(h3_sets[m2])

                comparison['overlap'][f"{m1}-{m2}"] = {
                    'intersection': len(overlap),
                    'union': len(union),
                    'jaccard': len(overlap) / len(union) if union else 0,
                    'm1_only': len(h3_sets[m1] - h3_sets[m2]),
                    'm2_only': len(h3_sets[m2] - h3_sets[m1]),
                    'overlap_pct_m1': len(overlap) / len(h3_sets[m1]) * 100 if h3_sets[m1] else 0,
                    'overlap_pct_m2': len(overlap) / len(h3_sets[m2]) * 100 if h3_sets[m2] else 0
                }

        # Find common H3 cells
        if h3_sets:
            common_h3 = set.intersection(*h3_sets.values())
            comparison['alignment'] = {
                'common_hexagons': len(common_h3),
                'coverage_pct': {
                    m: len(common_h3) / len(h3_sets[m]) * 100
                    for m in h3_sets
                } if common_h3 else {}
            }

        return comparison

    def validate_intermediate_data(self, modality: str, study_area: str = "netherlands",
                                  resolution: int = 10) -> Dict:
        """Validate intermediate SRAI data for a modality."""
        logger.info(f"Validating intermediate data for {modality}/{study_area} at resolution {resolution}")

        results = {
            'modality': modality,
            'study_area': study_area,
            'resolution': resolution,
            'files_found': {},
            'data_stats': {}
        }

        base_name = f"{study_area}_res{resolution}"
        intermediate_path = self.intermediate_dir / modality

        for data_type in ['features_gdf', 'regions_gdf', 'joint_gdf']:
            file_path = intermediate_path / data_type / f"{base_name}_{data_type.split('_')[0]}.parquet"

            if file_path.exists():
                results['files_found'][data_type] = True

                try:
                    gdf = gpd.read_parquet(file_path)

                    results['data_stats'][data_type] = {
                        'n_rows': len(gdf),
                        'n_columns': len(gdf.columns),
                        'columns': list(gdf.columns)[:10],
                        'geometry_type': gdf.geometry.type.value_counts().to_dict() if 'geometry' in gdf.columns else None,
                        'crs': str(gdf.crs) if hasattr(gdf, 'crs') else None,
                        'memory_mb': gdf.memory_usage(deep=True).sum() / 1024 / 1024
                    }

                    # Check for valid H3 hexagons
                    if data_type == 'regions_gdf' and 'geometry' in gdf.columns:
                        if gdf.index.name == 'region_id' or 'region_id' in gdf.columns:
                            sample_ids = (
                                gdf.index if gdf.index.name == 'region_id'
                                else gdf['region_id']
                            ).head(3).tolist()
                            h3_valid = all(_h3.is_valid_cell(str(idx)) for idx in sample_ids)
                            results['data_stats'][data_type]['h3_valid'] = h3_valid

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    results['data_stats'][data_type] = {'error': str(e)}
            else:
                results['files_found'][data_type] = False
                logger.warning(f"Missing: {file_path}")

        return results

    def generate_report(self, output_path: Optional[str] = None):
        """Generate a comprehensive validation report."""
        logger.info("Generating validation report...")

        report = []
        report.append("=" * 80)
        report.append("EMBEDDING VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        modalities = ['alphaearth', 'poi', 'roads', 'gtfs', 'buildings', 'streetview']
        available_modalities = []

        for modality in modalities:
            modality_path = self.embeddings_dir / modality
            if modality_path.exists() and any(modality_path.glob("*.parquet")):
                available_modalities.append(modality)

                report.append(f"\n{modality.upper()} MODALITY")
                report.append("-" * 40)

                validation = self.validate_modality(modality)

                if validation['embeddings_found']:
                    stats = validation['stats']
                    report.append(f"  Embeddings found: {stats['n_hexagons']} hexagons, {stats['n_features']} features")
                    report.append(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")
                    report.append(f"  Missing values: {stats['missing_values']}")

                    if 'h3_valid' in stats:
                        report.append(f"  H3 validation: {'Valid' if stats['h3_valid'] else 'Invalid'}")
                        report.append(f"  H3 resolutions: {stats['h3_resolutions']}")
                else:
                    report.append("  Embeddings not found")

                if validation['intermediate_found']:
                    inter_stats = validation.get('intermediate_stats', {})
                    report.append(f"  Intermediate data found:")
                    report.append(f"    Features: {inter_stats.get('features_gdf_files', 0)} files")
                    report.append(f"    Regions: {inter_stats.get('regions_gdf_files', 0)} files")
                    report.append(f"    Joints: {inter_stats.get('joint_gdf_files', 0)} files")
                else:
                    report.append("  Intermediate data not found")

        # Compare modalities
        if len(available_modalities) >= 2:
            report.append("\n" + "=" * 80)
            report.append("MODALITY COMPARISON")
            report.append("=" * 80)

            comparison = self.compare_modalities(available_modalities)

            for pair, stats in comparison['overlap'].items():
                report.append(f"\n{pair}:")
                report.append(f"  Overlap: {stats['intersection']:,} hexagons")
                report.append(f"  Jaccard similarity: {stats['jaccard']:.3f}")
                report.append(f"  Coverage: {stats['overlap_pct_m1']:.1f}% / {stats['overlap_pct_m2']:.1f}%")

            if 'common_hexagons' in comparison['alignment']:
                report.append(f"\nCommon hexagons across all modalities: {comparison['alignment']['common_hexagons']:,}")
                for modality, pct in comparison['alignment']['coverage_pct'].items():
                    report.append(f"  {modality}: {pct:.1f}% coverage")

        report_text = "\n".join(report)

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report_text, encoding='utf-8')
            logger.info(f"Report saved to {output_file}")

        print(report_text)
        return report_text
