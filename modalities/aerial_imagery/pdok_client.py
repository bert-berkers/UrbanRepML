"""
PDOK API Client for fetching aerial RGB images.

PDOK (Publieke Dienstverlening Op de Kaart) provides free access to 
high-resolution aerial imagery of the Netherlands.
"""

import io
import time
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import requests
import numpy as np
from PIL import Image
import geopandas as gpd
from shapely.geometry import box, Polygon
import h3
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageTile:
    """Container for an image tile with metadata."""
    image: np.ndarray
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy
    h3_cells: List[str]
    resolution: int
    crs: str = "EPSG:28992"  # Dutch RD New


class PDOKClient:
    """Client for fetching aerial imagery from PDOK WMS services."""
    
    # PDOK WMS endpoints for aerial imagery
    WMS_ENDPOINTS = {
        'current': 'https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0',
        '2023': 'https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0',
        '2022': 'https://service.pdok.nl/hwh/luchtfotorgb2022/wms/v1_0',
        '2021': 'https://service.pdok.nl/hwh/luchtfotorgb2021/wms/v1_0',
        '2020': 'https://service.pdok.nl/hwh/luchtfotorgb2020/wms/v1_0',
    }
    
    def __init__(self, 
                 year: str = 'current',
                 image_size: int = 512,
                 max_retries: int = 3,
                 rate_limit: float = 0.1):
        """
        Initialize PDOK client.
        
        Args:
            year: Year of imagery or 'current' for most recent
            image_size: Size of image tiles to fetch (pixels)
            max_retries: Maximum number of retry attempts
            rate_limit: Seconds to wait between requests
        """
        self.wms_url = self.WMS_ENDPOINTS.get(year, self.WMS_ENDPOINTS['current'])
        self.image_size = image_size
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        
        # Standard WMS parameters
        self.wms_params = {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetMap',
            'LAYERS': 'Actueel_orthoRGB',
            'STYLES': '',
            'FORMAT': 'image/png',
            'TRANSPARENT': 'false',
            'CRS': 'EPSG:28992',  # Dutch coordinate system
            'WIDTH': image_size,
            'HEIGHT': image_size,
        }
    
    def get_h3_bounds(self, h3_cell: str, buffer_m: float = 100) -> Tuple[float, float, float, float]:
        """
        Get bounding box for H3 cell in Dutch RD coordinates.
        
        Args:
            h3_cell: H3 cell index
            buffer_m: Buffer in meters around hexagon
            
        Returns:
            Tuple of (minx, miny, maxx, maxy) in EPSG:28992
        """
        # Get hexagon boundary in lat/lon
        boundary = h3.h3_to_geo_boundary(h3_cell)
        coords = [(lon, lat) for lat, lon in boundary]
        
        # Create polygon and convert to Dutch RD
        poly = gpd.GeoSeries([Polygon(coords)], crs='EPSG:4326')
        poly_rd = poly.to_crs('EPSG:28992')
        
        # Get bounds with buffer
        bounds = poly_rd.total_bounds
        minx, miny, maxx, maxy = bounds
        
        # Add buffer
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m
        
        return minx, miny, maxx, maxy
    
    def fetch_image_for_h3(self, h3_cell: str) -> Optional[ImageTile]:
        """
        Fetch aerial image for a single H3 cell.
        
        Args:
            h3_cell: H3 cell index
            
        Returns:
            ImageTile with image data and metadata
        """
        try:
            # Get bounding box
            bounds = self.get_h3_bounds(h3_cell)
            bbox_str = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"
            
            # Prepare WMS request
            params = self.wms_params.copy()
            params['BBOX'] = bbox_str
            
            # Fetch with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.get(self.wms_url, params=params, timeout=30)
                    if response.status_code == 200:
                        # Load image
                        img = Image.open(io.BytesIO(response.content))
                        img_array = np.array(img)
                        
                        return ImageTile(
                            image=img_array,
                            bounds=bounds,
                            h3_cells=[h3_cell],
                            resolution=h3.h3_get_resolution(h3_cell)
                        )
                    else:
                        logger.warning(f"HTTP {response.status_code} for H3 {h3_cell}")
                        
                except requests.RequestException as e:
                    logger.warning(f"Request failed (attempt {attempt+1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        
            return None
            
        except Exception as e:
            logger.error(f"Error fetching image for {h3_cell}: {e}")
            return None
    
    def fetch_images_for_hexagons(self, 
                                  h3_cells: List[str],
                                  batch_size: int = 10) -> Dict[str, ImageTile]:
        """
        Fetch images for multiple H3 cells with batching.
        
        Args:
            h3_cells: List of H3 cell indices
            batch_size: Number of images to fetch before pausing
            
        Returns:
            Dictionary mapping H3 cells to ImageTiles
        """
        results = {}
        
        for i, h3_cell in enumerate(h3_cells):
            # Rate limiting
            if i > 0:
                time.sleep(self.rate_limit)
            
            # Batch pausing
            if i > 0 and i % batch_size == 0:
                logger.info(f"Processed {i}/{len(h3_cells)} hexagons, pausing...")
                time.sleep(2.0)  # Longer pause between batches
            
            # Fetch image
            tile = self.fetch_image_for_h3(h3_cell)
            if tile:
                results[h3_cell] = tile
                logger.debug(f"Fetched image for {h3_cell}")
            else:
                logger.warning(f"Failed to fetch image for {h3_cell}")
        
        logger.info(f"Fetched {len(results)}/{len(h3_cells)} images successfully")
        return results
    
    def get_composite_image(self, 
                           h3_cells: List[str],
                           resolution: int = 256) -> Optional[np.ndarray]:
        """
        Create a composite image covering multiple H3 cells.
        
        Args:
            h3_cells: List of H3 cells to cover
            resolution: Output image resolution
            
        Returns:
            Composite image as numpy array
        """
        if not h3_cells:
            return None
        
        # Get overall bounding box
        all_bounds = []
        for h3_cell in h3_cells:
            bounds = self.get_h3_bounds(h3_cell, buffer_m=0)
            all_bounds.append(bounds)
        
        # Calculate composite bounds
        all_bounds = np.array(all_bounds)
        minx = all_bounds[:, 0].min()
        miny = all_bounds[:, 1].min()
        maxx = all_bounds[:, 2].max()
        maxy = all_bounds[:, 3].max()
        
        # Fetch composite image
        bbox_str = f"{minx},{miny},{maxx},{maxy}"
        params = self.wms_params.copy()
        params['BBOX'] = bbox_str
        params['WIDTH'] = resolution
        params['HEIGHT'] = resolution
        
        try:
            response = requests.get(self.wms_url, params=params, timeout=60)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                return np.array(img)
        except Exception as e:
            logger.error(f"Failed to fetch composite image: {e}")
            
        return None