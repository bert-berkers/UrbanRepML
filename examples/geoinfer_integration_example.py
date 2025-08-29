"""
Example: Integrating UrbanRepML with GEO-INFER for Cascadia agricultural analysis.

This example shows how to:
1. Load UrbanRepML AlphaEarth embeddings
2. Convert to GEO-INFER format
3. Use GEO-INFER modules for agricultural analysis
4. Combine results back into UrbanRepML pipeline
"""

import sys
from pathlib import Path
import pandas as pd

# Add GEO-INFER to path
sys.path.append(str(Path(__file__).parent.parent / "external/geo-infer"))

# Import bridge utilities
from bridge import (
    urbanreml_to_geoinfer,
    geoinfer_to_urbanreml,
    h3_data_bridge,
    combine_embeddings
)


def main():
    # 1. Load UrbanRepML data (Cascadia AlphaEarth embeddings)
    cascadia_data_path = Path("experiments/cascadia_geoinfer_alphaearth/data/h3_processed/resolution_8")
    
    if cascadia_data_path.exists():
        # Load 2023 data
        embeddings_2023 = pd.read_parquet(cascadia_data_path / "cascadia_2023_h3_res8.parquet")
        print(f"Loaded {len(embeddings_2023)} H3 cells from Cascadia 2023")
        
        # 2. Convert to GEO-INFER format
        geoinfer_data = urbanreml_to_geoinfer(
            cascadia_data_path / "cascadia_2023_h3_res8.parquet",
            data_type="alphaearth",
            resolution=8
        )
        print(f"Converted to GEO-INFER format with {len(geoinfer_data['h3_index'])} cells")
        
        # 3. Bridge to specific GEO-INFER modules
        bridged_data = h3_data_bridge(
            embeddings_2023,
            geoinfer_modules=["agricultural_analysis", "climate_impact"],
            resolution=8
        )
        
        print("\nBridged data for GEO-INFER modules:")
        for module, data in bridged_data.items():
            print(f"  - {module}: {len(data.get('h3_cells', []))} cells")
        
        # 4. Example: Combine with GEO-INFER features (mock example)
        urbanreml_features = embeddings_2023.select_dtypes(include='number').values
        
        # Mock GEO-INFER features (in practice, these would come from GEO-INFER modules)
        import numpy as np
        mock_geoinfer_features = np.random.randn(len(embeddings_2023), 10)
        
        combined_features = combine_embeddings(
            urbanreml_features,
            mock_geoinfer_features,
            combination_method="concatenate"
        )
        
        print(f"\nCombined features shape: {combined_features.shape}")
        print(f"  - UrbanRepML features: {urbanreml_features.shape}")
        print(f"  - GEO-INFER features: {mock_geoinfer_features.shape}")
        
    else:
        print("Cascadia data not found. Creating example with South Holland data...")
        
        # Alternative: Use South Holland data as example
        sh_data_path = Path("data/preprocessed [TODO SORT & CLEAN UP]/south_holland_fsi99/regions/regions_8_gdf.parquet")
        if sh_data_path.exists():
            sh_data = pd.read_parquet(sh_data_path)
            print(f"Using South Holland data: {len(sh_data)} regions")
            
            # Convert to GEO-INFER format
            geoinfer_data = {
                "metadata": {
                    "source": "UrbanRepML",
                    "region": "South Holland",
                    "h3_resolution": 8
                },
                "data": sh_data.to_dict('records'),
                "h3_index": sh_data.index.tolist()
            }
            
            print(f"Prepared {len(geoinfer_data['h3_index'])} cells for GEO-INFER")


if __name__ == "__main__":
    main()