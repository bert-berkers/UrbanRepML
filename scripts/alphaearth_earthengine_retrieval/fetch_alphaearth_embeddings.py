"""
Fetches AlphaEarth embeddings from the Google Earth Engine API.

This script authenticates with Earth Engine using a service account,
loads a study area boundary, and exports the corresponding AlphaEarth
imagery to Google Drive as a GeoTIFF.

⚠️  INTEGRATION NOTE: This script exports a single large GeoTIFF per study area.
    For better integration with the existing AlphaEarth processing pipeline that
    expects tiled inputs, consider using:
    
    fetch_alphaearth_embeddings_tiled.py
    
    The tiled version provides:
    - Automatic tiling for large study areas
    - Naming conventions that match existing processors  
    - Tile boundary metadata for seamless stitching
    - Better integration with H3 processing pipeline

Prerequisites:
1.  A Google Cloud project with Earth Engine API enabled.
2.  A service account with appropriate permissions.
3.  A `.env` file in the `keys/` directory with the following variables:
    - GEE_SERVICE_ACCOUNT: Your service account email.
    - GEE_PRIVATE_KEY_PATH: Path to your service account's JSON key file.

Example usage:
    python scripts/alphaearth_earthengine_retrieval/fetch_alphaearth_embeddings.py --study-area cascadia_oldremove --year 2022
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict

import ee
import geopandas as gpd
from dotenv import load_dotenv

# --- Configuration ---
# Set up logging for clear output
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# The Earth Engine asset ID for the AlphaEarth collection.
# This ID was provided by the user and points to the Google Satellite Embedding dataset.
ALPHAEARTH_COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


# --- Functions ---
def get_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fetch AlphaEarth embeddings from Google Earth Engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--study-area",
        type=str,
        required=True,
        help="Name of the study area (e.g., 'netherlands', 'cascadia_oldremove').",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Year to filter the AlphaEarth data for.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10,
        help="Resolution for the Earth Engine export in meters.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="UrbanRepML_Exports",
        help="Google Drive folder to save the exported GeoTIFF.",
    )
    return parser.parse_args()


def initialize_ee():
    """
    Initialize Google Earth Engine using service account credentials.

    Loads credentials from a `.env` file and authenticates.
    Raises:
        Exception: If initialization fails.
    """
    logger.info("Initializing Earth Engine...")
    load_dotenv()
    try:
        # ee.Initialize() will automatically use the service account credentials
        # if the GOOGLE_APPLICATION_CREDENTIALS environment variable is set,
        # which is common practice. Alternatively, you can be more explicit.
        # For this script, we rely on the standard `gcloud auth` flow or
        # a properly configured .env file that python-dotenv can pick up.
        ee.Initialize(credentials="service_account")
        logger.info("Earth Engine initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine: {e}")
        logger.error(
            "Please ensure your service account credentials are set up correctly "
            "in a .env file or as environment variables (e.g., GOOGLE_APPLICATION_CREDENTIALS)."
        )
        raise


def get_study_area_geometry(study_area_name: str) -> ee.Geometry:
    """
    Load a study area boundary from a GeoJSON file and convert it to an ee.Geometry.

    Args:
        study_area_name (str): The name of the study area.

    Returns:
        ee.Geometry: The Earth Engine geometry object for the study area.
    Raises:
        FileNotFoundError: If the boundary file cannot be found.
    """
    boundary_path = (
        Path(f"data/boundaries/{study_area_name}/{study_area_name}_states.geojson")
    )
    logger.info(f"Loading boundary from: {boundary_path}")

    if not boundary_path.exists():
        logger.error(f"Boundary file not found: {boundary_path}")
        raise FileNotFoundError(f"Boundary file not found at {boundary_path}")

    gdf = gpd.read_file(boundary_path)
    # Dissolve to ensure a single, unified geometry for the study area
    gdf_dissolved = gdf.dissolve()
    geom = gdf_dissolved.geometry.iloc[0]

    # Convert the shapely geometry to a GeoJSON-like dictionary
    gjson = geom.__geo_interface__

    # Create an Earth Engine Geometry from the GeoJSON
    ee_geom = ee.Geometry(gjson)
    logger.info("Successfully loaded and converted study area geometry.")
    return ee_geom


def export_alphaearth_to_drive(
    geometry: ee.Geometry, year: int, scale: int, study_area_name: str, drive_folder: str
) -> ee.batch.Task:
    """
    Create and start an Earth Engine task to export AlphaEarth imagery to Google Drive.

    Args:
        geometry (ee.Geometry): The geometry of the area to export.
        year (int): The year of the data to export.
        scale (int): The export resolution in meters.
        study_area_name (str): The name of the study area for file naming.
        drive_folder (str): The Google Drive folder for the export.

    Returns:
        ee.batch.Task: The started Earth Engine task.
    """
    logger.info(f"Fetching AlphaEarth data for {year} at {scale}m resolution.")

    # Filter the image collection by date and location
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    image = (
        ee.ImageCollection(ALPHAEARTH_COLLECTION_ID)
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .mosaic()
        .clip(geometry)
    )

    # Configure the export task
    file_name = f"alphaearth_{study_area_name}_{year}_{scale}m"
    task_config: Dict[str, object] = {
        "image": image,
        "description": file_name,
        "folder": drive_folder,
        "fileNamePrefix": file_name,
        "scale": scale,
        "region": geometry,
        "fileFormat": "GeoTIFF",
        "maxPixels": 1e13,  # Allow for large exports
    }

    task = ee.batch.Export.image.toDrive(**task_config)
    task.start()

    logger.info(f"Started export task: {task.id} ({file_name})")
    logger.info(
        f"Check the 'Tasks' tab in the Earth Engine Code Editor or your Google Drive folder '{drive_folder}' for progress."
    )
    return task


def monitor_task(task: ee.batch.Task):
    """
    Monitor an Earth Engine task, providing status updates until it completes.

    Args:
        task (ee.batch.Task): The task to monitor.
    """
    logger.info(f"Monitoring task: {task.id}")
    while task.active():
        status = task.status()
        logger.info(f"Task status ({status['id']}): {status['state']}")
        time.sleep(30)  # Check status every 30 seconds

    final_status = task.status()
    if final_status["state"] == "COMPLETED":
        logger.info(f"Task {final_status['id']} completed successfully.")
    else:
        logger.error(f"Task {final_status['id']} failed or was cancelled.")
        logger.error(f"Final state: {final_status['state']}")
        logger.error(f"Error message: {final_status.get('error_message', 'N/A')}")


def main():
    """
    Main execution block for the script.
    """
    try:
        args = get_arguments()
        logger.info(f"Starting AlphaEarth fetch for study area: '{args.study_area}'")

        initialize_ee()
        study_area_geom = get_study_area_geometry(args.study_area)

        export_task = export_alphaearth_to_drive(
            geometry=study_area_geom,
            year=args.year,
            scale=args.scale,
            study_area_name=args.study_area,
            drive_folder=args.output_folder,
        )

        monitor_task(export_task)

        logger.info("Script finished.")

    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
