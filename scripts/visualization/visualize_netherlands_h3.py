"""
Visualize Netherlands H3 hexagons at resolution 8 using SRAI
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import folium
from folium import plugins
from srai.plotting import plot_regions, plot_numeric_data
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def load_h3_data(resolution=8):
    """Load H3 hexagons and related data"""
    
    data_dir = Path('data/processed/h3_regions/netherlands')
    
    # Load H3 hexagons
    h3_path = data_dir / f'netherlands_h3_res{resolution}.parquet'
    print(f"Loading H3 hexagons at resolution {resolution}...")
    h3_gdf = gpd.read_parquet(h3_path)
    print(f"  Loaded {len(h3_gdf):,} hexagons")
    
    # Load boundary
    boundary_path = data_dir / 'netherlands_boundary.geojson'
    boundary_gdf = gpd.read_file(boundary_path)
    
    # Load AlphaEarth coverage
    coverage_path = data_dir / 'alphaearth_coverage.geojson'
    coverage_gdf = None
    if coverage_path.exists():
        coverage_gdf = gpd.read_file(coverage_path)
    
    return h3_gdf, boundary_gdf, coverage_gdf


def create_interactive_map(h3_gdf, boundary_gdf, coverage_gdf=None):
    """Create interactive Folium map with SRAI plotting"""
    
    # Calculate map center
    bounds = boundary_gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='CartoDB positron',
        prefer_canvas=True  # Better performance
    )
    
    # Add tile layers
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB dark_matter').add_to(m)
    
    # Sample hexagons for better performance (show every nth hexagon)
    sample_size = min(10000, len(h3_gdf))  # Max 10k hexagons for performance
    if len(h3_gdf) > sample_size:
        print(f"  Sampling {sample_size:,} hexagons for visualization...")
        h3_sample = h3_gdf.sample(n=sample_size, random_state=42)
    else:
        h3_sample = h3_gdf
    
    # Add hexagons with SRAI
    print("  Adding H3 hexagons to map...")
    
    # Create a simple choropleth based on hexagon density
    # Add a dummy column for visualization
    h3_sample['hex_id_numeric'] = range(len(h3_sample))
    h3_sample['density'] = np.random.random(len(h3_sample))  # Placeholder for actual data
    
    # Use SRAI plotting
    try:
        plot_regions(h3_sample, m)  # SRAI plot_regions takes fewer arguments
    except Exception as e:
        print(f"  Warning: SRAI plot_regions failed: {e}")
        # Fallback to standard Folium
        folium.GeoJson(
            h3_sample.to_json(),
            style_function=lambda x: {
                'fillColor': '#00B0F0',
                'color': '#0080C0',
                'weight': 0.5,
                'fillOpacity': 0.3
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['h3_resolution', 'area_km2'],
                aliases=['Resolution:', 'Area (km²):'],
                localize=True
            )
        ).add_to(m)
    
    # Add Netherlands boundary
    print("  Adding Netherlands boundary...")
    folium.GeoJson(
        boundary_gdf.to_json(),
        name='Netherlands Boundary',
        style_function=lambda x: {
            'fillColor': 'none',
            'color': '#FF0000',
            'weight': 2,
            'dashArray': '5, 5'
        }
    ).add_to(m)
    
    # Add AlphaEarth coverage if available
    if coverage_gdf is not None:
        print("  Adding AlphaEarth coverage extent...")
        folium.GeoJson(
            coverage_gdf.to_json(),
            name='AlphaEarth Coverage',
            style_function=lambda x: {
                'fillColor': '#00FF00',
                'color': '#00AA00',
                'weight': 2,
                'fillOpacity': 0.1,
                'dashArray': '10, 5'
            }
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add mouse position
    plugins.MousePosition().add_to(m)
    
    # Add minimap
    minimap = plugins.MiniMap(toggle_display=True)
    m.add_child(minimap)
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; 
                left: 50px; 
                width: 400px; 
                height: 90px; 
                background-color: white; 
                z-index: 9999; 
                font-size: 16px;
                border: 2px solid grey;
                border-radius: 5px;
                padding: 10px">
        <b>Netherlands H3 Hexagons - Resolution 8</b><br>
        <i>132,603 total hexagons (~0.61 km² each)</i><br>
        <span style="color: red;">━━</span> Netherlands boundary<br>
        <span style="color: green;">━━</span> AlphaEarth data coverage (partial)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m


def create_static_plots(h3_gdf, boundary_gdf, coverage_gdf=None):
    """Create static matplotlib visualizations"""
    
    print("\nCreating static visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: H3 Hexagon Grid
    ax1 = axes[0]
    h3_sample = h3_gdf.sample(n=min(5000, len(h3_gdf)), random_state=42)
    h3_sample.plot(ax=ax1, color='lightblue', edgecolor='blue', linewidth=0.1, alpha=0.5)
    boundary_gdf.plot(ax=ax1, color='none', edgecolor='red', linewidth=2, linestyle='--')
    if coverage_gdf is not None:
        coverage_gdf.plot(ax=ax1, color='none', edgecolor='green', linewidth=2, linestyle=':')
    ax1.set_title(f'Netherlands H3 Hexagons\nResolution 8 ({len(h3_gdf):,} hexagons)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hexagon Density Heatmap
    ax2 = axes[1]
    
    # Create a density grid
    bounds = h3_gdf.total_bounds
    lon_bins = np.linspace(bounds[0], bounds[2], 50)
    lat_bins = np.linspace(bounds[1], bounds[3], 50)
    
    # Get centroids
    centroids = h3_gdf.geometry.centroid
    lons = centroids.x.values
    lats = centroids.y.values
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    
    # Plot heatmap
    im = ax2.imshow(H.T, origin='lower', extent=[bounds[0], bounds[2], bounds[1], bounds[3]], 
                    cmap='YlOrRd', aspect='auto', interpolation='gaussian')
    boundary_gdf.plot(ax=ax2, color='none', edgecolor='black', linewidth=2)
    if coverage_gdf is not None:
        coverage_gdf.plot(ax=ax2, color='none', edgecolor='green', linewidth=2, linestyle=':')
    ax2.set_title('H3 Hexagon Density Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, label='Hexagon Count')
    
    plt.suptitle('Netherlands H3 Spatial Coverage Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def save_visualizations(m, fig):
    """Save all visualizations"""
    
    output_dir = Path('data/processed/h3_regions/netherlands/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save interactive map
    map_path = output_dir / 'netherlands_h3_res8_interactive.html'
    m.save(str(map_path))
    print(f"\nSaved interactive map to: {map_path}")
    
    # Save static plots
    static_path = output_dir / 'netherlands_h3_res8_static.png'
    fig.savefig(static_path, dpi=150, bbox_inches='tight')
    print(f"Saved static visualization to: {static_path}")
    
    # Save high-res version
    static_hires_path = output_dir / 'netherlands_h3_res8_static_hires.png'
    fig.savefig(static_hires_path, dpi=300, bbox_inches='tight')
    print(f"Saved high-res visualization to: {static_hires_path}")


def print_statistics(h3_gdf):
    """Print H3 statistics"""
    
    print("\n" + "="*60)
    print("H3 Hexagon Statistics (Resolution 8)")
    print("="*60)
    
    # Basic stats
    print(f"Total hexagons: {len(h3_gdf):,}")
    print(f"Area per hexagon: {h3_gdf['area_km2'].iloc[0]:.3f} km²")
    print(f"Total coverage: {len(h3_gdf) * h3_gdf['area_km2'].iloc[0]:,.0f} km²")
    
    # Bounding box
    bounds = h3_gdf.total_bounds
    print(f"\nBounding box:")
    print(f"  West: {bounds[0]:.3f}°")
    print(f"  East: {bounds[2]:.3f}°")
    print(f"  South: {bounds[1]:.3f}°")
    print(f"  North: {bounds[3]:.3f}°")
    
    # Geographic span
    print(f"\nGeographic span:")
    print(f"  Width: {bounds[2] - bounds[0]:.3f}° (~{(bounds[2] - bounds[0]) * 111:.0f} km)")
    print(f"  Height: {bounds[3] - bounds[1]:.3f}° (~{(bounds[3] - bounds[1]) * 111:.0f} km)")


def main():
    """Main execution"""
    
    print("="*60)
    print("Netherlands H3 Visualization - Resolution 8")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    h3_gdf, boundary_gdf, coverage_gdf = load_h3_data(resolution=8)
    
    # Print statistics
    print_statistics(h3_gdf)
    
    # Create interactive map
    print("\nCreating interactive map...")
    m = create_interactive_map(h3_gdf, boundary_gdf, coverage_gdf)
    
    # Create static plots
    fig = create_static_plots(h3_gdf, boundary_gdf, coverage_gdf)
    
    # Save visualizations
    save_visualizations(m, fig)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print("\nOpen the HTML file in your browser to view the interactive map.")
    print("The static plots provide overview visualizations.")


if __name__ == "__main__":
    main()