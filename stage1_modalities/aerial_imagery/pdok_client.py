"""
PDOK API Client for fetching aerial RGB images.

PDOK (Publieke Dienstverlening Op de Kaart) provides free access to
high-resolution aerial imagery of the Netherlands.

Design: 1 PDOK request = 1 hex = 1 DINOv3 input (224x224).
Hexagon geometry comes from the regions_gdf (via SRAI H3Regionalizer),
not from direct h3 boundary calls (per srai-spatial rule).
"""

import io
import time
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageTile:
    """Container for an image tile with metadata."""
    image_path: str  # Path to saved PNG on disk
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy in EPSG:28992
    h3_cell: str
    crs: str = "EPSG:28992"  # Dutch RD New


class PDOKClient:
    """Client for fetching aerial imagery from PDOK WMS services.

    Fetches one 224x224 image per H3 hexagon. Hexagon geometry is derived
    from a regions_gdf (produced by SRAI H3Regionalizer), reprojected to
    EPSG:28992 for the PDOK WMS bounding box.
    """

    # PDOK WMS endpoints for aerial imagery
    WMS_ENDPOINTS = {
        'current': 'https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0',
        '2023': 'https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0',
        '2022': 'https://service.pdok.nl/hwh/luchtfotorgb2022/wms/v1_0',
        '2021': 'https://service.pdok.nl/hwh/luchtfotorgb2021/wms/v1_0',
        '2020': 'https://service.pdok.nl/hwh/luchtfotorgb2020/wms/v1_0',
    }

    def __init__(
        self,
        year: str = 'current',
        image_size: int = 224,
        max_retries: int = 3,
        max_workers: int = 64,
    ):
        """Initialize PDOK client.

        Args:
            year: Year of imagery or 'current' for most recent.
            image_size: Size of image tiles to fetch (pixels). Default 224
                matches DINOv3 native input.
            max_retries: Maximum number of retry attempts per request.
            max_workers: Thread pool size for parallel fetching.
        """
        self.wms_url = self.WMS_ENDPOINTS.get(year, self.WMS_ENDPOINTS['current'])
        self.image_size = image_size
        self.max_retries = max_retries
        self.max_workers = max_workers

        # Standard WMS parameters
        self.wms_params = {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetMap',
            'LAYERS': 'Actueel_orthoRGB',
            'STYLES': '',
            'FORMAT': 'image/png',
            'TRANSPARENT': 'false',
            'CRS': 'EPSG:28992',
            'WIDTH': image_size,
            'HEIGHT': image_size,
        }

    def get_h3_bounds(
        self, geometry, buffer_m: float = 10.0
    ) -> Tuple[float, float, float, float]:
        """Get bounding box for a hexagon geometry in Dutch RD coordinates.

        Args:
            geometry: Shapely geometry (from regions_gdf) in EPSG:4326.
            buffer_m: Buffer in meters around hexagon. Tight (10m) for 224px.

        Returns:
            Tuple of (minx, miny, maxx, maxy) in EPSG:28992.
        """
        # Create GeoSeries in WGS84 and reproject to Dutch RD
        poly = gpd.GeoSeries([geometry], crs='EPSG:4326')
        poly_rd = poly.to_crs('EPSG:28992')

        # Get bounds with small buffer
        bounds = poly_rd.total_bounds
        minx, miny, maxx, maxy = bounds

        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m

        return minx, miny, maxx, maxy

    def fetch_image_for_h3(
        self, h3_cell: str, geometry, cache_dir: Path
    ) -> Optional[str]:
        """Fetch aerial image for a single H3 cell and save to disk.

        Args:
            h3_cell: H3 cell index string.
            geometry: Shapely geometry for this cell (from regions_gdf).
            cache_dir: Directory to save PNG files.

        Returns:
            Path to saved PNG file, or None on failure.
        """
        out_path = cache_dir / f"{h3_cell}.png"

        # Skip if already cached (resume support)
        if out_path.exists():
            return str(out_path)

        try:
            bounds = self.get_h3_bounds(geometry)
            bbox_str = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"

            params = self.wms_params.copy()
            params['BBOX'] = bbox_str

            for attempt in range(self.max_retries):
                try:
                    response = requests.get(
                        self.wms_url, params=params, timeout=30
                    )
                    if response.status_code == 200:
                        # Validate image content
                        img = Image.open(io.BytesIO(response.content))
                        img = img.convert('RGB')
                        img.save(out_path, 'PNG')
                        return str(out_path)
                    else:
                        logger.warning(
                            f"HTTP {response.status_code} for H3 {h3_cell}"
                        )

                except requests.RequestException as e:
                    logger.warning(
                        f"Request failed for {h3_cell} "
                        f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff

            return None

        except Exception as e:
            logger.error(f"Error fetching image for {h3_cell}: {e}")
            return None

    def fetch_images_parallel(
        self,
        regions_gdf: gpd.GeoDataFrame,
        cache_dir: Path,
    ) -> Dict[str, str]:
        """Fetch images for all hexagons in parallel using ThreadPoolExecutor.

        Args:
            regions_gdf: GeoDataFrame indexed by region_id with geometry
                (from SRAI H3Regionalizer, CRS EPSG:4326).
            cache_dir: Directory to save/cache PNG files.

        Returns:
            Dictionary mapping h3_cell -> image_path for successful fetches.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        hex_ids = regions_gdf.index.tolist()
        total = len(hex_ids)
        results: Dict[str, str] = {}

        # Check how many are already cached
        already_cached = sum(
            1 for h in hex_ids if (cache_dir / f"{h}.png").exists()
        )
        if already_cached > 0:
            logger.info(
                f"Found {already_cached}/{total} images already cached"
            )

        def _fetch_one(h3_cell: str) -> Tuple[str, Optional[str]]:
            geometry = regions_gdf.loc[h3_cell, 'geometry']
            path = self.fetch_image_for_h3(h3_cell, geometry, cache_dir)
            return h3_cell, path

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_fetch_one, h): h for h in hex_ids
            }

            completed = 0
            for future in as_completed(futures):
                h3_cell, path = future.result()
                if path is not None:
                    results[h3_cell] = path

                completed += 1
                if completed % 10_000 == 0:
                    logger.info(
                        f"Progress: {completed}/{total} fetched, "
                        f"{len(results)} successful"
                    )

        logger.info(
            f"Fetch complete: {len(results)}/{total} images successful"
        )
        return results
