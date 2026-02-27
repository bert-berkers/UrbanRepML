"""
Spatial query engine backed by SedonaDB with a GeoPandas fallback.

Wraps SedonaDB to provide efficient spatial queries (centroids, geometry,
extents) over pre-computed H3 regions_gdf GeoParquet files.  Falls back
to plain GeoPandas when SedonaDB is not importable.

The fallback chain is:
    SedonaDB  →  regions_gdf parquet via GeoPandas  →  FileNotFoundError

``h3_to_geoseries`` is intentionally NOT used anywhere in this module.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from srai.neighbourhoods import H3Neighbourhood

from utils.paths import StudyAreaPaths

# Type alias for the hex-ID inputs accepted by all public methods.
HexIDs = Union[pd.Index, pd.Series, list, np.ndarray]


class SpatialDB:
    """Spatial query engine backed by SedonaDB with GeoPandas fallback.

    Loads pre-computed regions_gdf GeoParquet files and provides
    spatial query methods (centroids, geometry, extent).  Views are
    registered lazily on first access per resolution.  Instances are
    cached per study area via ``for_study_area()``.

    Usage::

        db = SpatialDB.for_study_area("netherlands")
        cx, cy = db.centroids(hex_ids, resolution=9)
        gdf = db.geometry(hex_ids, resolution=10)
        minx, miny, maxx, maxy = db.extent(hex_ids, resolution=9, crs=28992)
    """

    _instances: dict[str, "SpatialDB"] = {}

    # ------------------------------------------------------------------
    # Construction / caching
    # ------------------------------------------------------------------

    @classmethod
    def for_study_area(cls, study_area: str) -> "SpatialDB":
        """Get or create a cached SpatialDB instance for a study area."""
        if study_area not in cls._instances:
            cls._instances[study_area] = cls(study_area)
        return cls._instances[study_area]

    def __init__(self, study_area: str) -> None:
        self._paths = StudyAreaPaths(study_area)
        self._views: dict[int, str] = {}  # resolution -> registered view name
        try:
            import sedonadb  # type: ignore[import]
            self._ctx = sedonadb.connect()
            self._backend = "sedonadb"
        except ImportError:
            self._ctx = None
            self._backend = "geopandas"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_view(self, resolution: int) -> str:
        """Register a regions parquet as a SedonaDB view if not already loaded.

        Returns the view name so callers can reference it in SQL.
        """
        if resolution not in self._views:
            path = self._paths.region_file(resolution)
            if not path.exists():
                raise FileNotFoundError(
                    f"Region file not found: {path}. "
                    f"Run setup_regions.py first."
                )
            view = f"regions_res{resolution}"
            self._ctx.read_parquet(str(path)).to_view(view)
            self._views[resolution] = view
        return self._views[resolution]

    def _ensure_ids_view(self, hex_ids: HexIDs, view_name: str = "query_ids") -> None:
        """Register hex IDs as a SedonaDB view for use in JOINs.

        Using a DataFrame JOIN avoids emitting a massive SQL IN clause for
        the ~935K hex IDs typical in a Netherlands res10 probe.
        """
        ids_series = pd.Series(np.asarray(hex_ids, dtype=str), name="region_id")
        ids_df = ids_series.to_frame()
        self._ctx.create_data_frame(ids_df).to_view(view_name, overwrite=True)

    @staticmethod
    def _transform_clause(geom_col: str, from_crs: int, to_crs: int) -> str:
        """Return an SQL expression that transforms a geometry column.

        When source and target CRS are identical no transform is emitted.
        SedonaDB uses the ``ST_Transform(geom, 'EPSG:src', 'EPSG:dst')`` form.
        """
        if from_crs == to_crs:
            return geom_col
        return f"ST_Transform({geom_col}, 'EPSG:{from_crs}', 'EPSG:{to_crs}')"

    def _load_gdf_fallback(self, hex_ids: HexIDs, resolution: int):
        """Load the regions parquet with GeoPandas and filter to hex_ids."""
        import geopandas as gpd

        path = self._paths.region_file(resolution)
        if not path.exists():
            raise FileNotFoundError(
                f"Region file not found: {path}. "
                f"Run setup_regions.py first."
            )
        gdf = gpd.read_parquet(path)

        # Defensive normalisation: ensure index is named region_id.
        if gdf.index.name != "region_id":
            gdf.index.name = "region_id"

        ids_arr = np.asarray(hex_ids, dtype=str)
        return gdf.loc[gdf.index.isin(ids_arr)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def centroids(
        self,
        hex_ids: HexIDs,
        resolution: int,
        crs: int = 4326,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (cx, cy) coordinate arrays for the given hex IDs.

        Parameters
        ----------
        hex_ids:
            Hex ID strings to look up.  Accepts pd.Index, pd.Series,
            list, or np.ndarray.
        resolution:
            H3 resolution of the regions parquet to query.
        crs:
            Target EPSG code.  The regions parquet is stored in EPSG:4326;
            pass a different value to reproject (e.g. 28992 for RD New).

        Returns
        -------
        (cx, cy):
            Two float64 numpy arrays of centroid coordinates.  The order
            matches the order of *hex_ids*.
        """
        if self._backend == "sedonadb":
            view = self._ensure_view(resolution)
            self._ensure_ids_view(hex_ids, view_name="query_ids")
            geom_expr = self._transform_clause("r.geometry", 4326, crs)
            sql = f"""
                SELECT
                    q.region_id,
                    ST_X(ST_Centroid({geom_expr})) AS cx,
                    ST_Y(ST_Centroid({geom_expr})) AS cy
                FROM query_ids q
                JOIN {view} r ON q.region_id = r.region_id
            """
            result_df = self._ctx.sql(sql).to_pandas()
            # Reindex to match input order
            result_df = result_df.set_index("region_id")
            ids_arr = np.asarray(hex_ids, dtype=str)
            result_df = result_df.reindex(ids_arr)
            return result_df["cx"].values, result_df["cy"].values

        # GeoPandas fallback
        filtered = self._load_gdf_fallback(hex_ids, resolution)
        if crs != 4326:
            filtered = filtered.to_crs(epsg=crs)
        centroids = filtered.geometry.centroid
        ids_arr = np.asarray(hex_ids, dtype=str)
        centroids = centroids.reindex(ids_arr)
        return centroids.x.values, centroids.y.values

    def geometry(
        self,
        hex_ids: HexIDs,
        resolution: int,
        crs: int = 4326,
    ):
        """Return a GeoDataFrame with polygon geometry for the given hex IDs.

        Parameters
        ----------
        hex_ids:
            Hex ID strings to look up.
        resolution:
            H3 resolution of the regions parquet to query.
        crs:
            Target EPSG code.  Defaults to 4326 (WGS84).

        Returns
        -------
        geopandas.GeoDataFrame
            Indexed by region_id.  Row order matches *hex_ids*.
        """
        import geopandas as gpd

        if self._backend == "sedonadb":
            view = self._ensure_view(resolution)
            self._ensure_ids_view(hex_ids, view_name="query_ids")
            geom_expr = self._transform_clause("r.geometry", 4326, crs)
            sql = f"""
                SELECT q.region_id, {geom_expr} AS geometry
                FROM query_ids q
                JOIN {view} r ON q.region_id = r.region_id
            """
            result = self._ctx.sql(sql).to_pandas()
            if not isinstance(result, gpd.GeoDataFrame):
                result = gpd.GeoDataFrame(result, geometry="geometry", crs=crs)
            result = result.set_index("region_id")
            ids_arr = np.asarray(hex_ids, dtype=str)
            return result.reindex(ids_arr)

        # GeoPandas fallback
        filtered = self._load_gdf_fallback(hex_ids, resolution)
        if crs != 4326:
            filtered = filtered.to_crs(epsg=crs)
        ids_arr = np.asarray(hex_ids, dtype=str)
        return filtered.reindex(ids_arr)

    def extent(
        self,
        hex_ids: HexIDs,
        resolution: int,
        crs: int = 4326,
    ) -> tuple[float, float, float, float]:
        """Return the bounding box (minx, miny, maxx, maxy) for the given hex IDs.

        Parameters
        ----------
        hex_ids:
            Hex ID strings to include in the bounding box.
        resolution:
            H3 resolution of the regions parquet to query.
        crs:
            Target EPSG code.  Defaults to 4326 (WGS84).

        Returns
        -------
        (minx, miny, maxx, maxy):
            Bounding box in the requested CRS.
        """
        if self._backend == "sedonadb":
            view = self._ensure_view(resolution)
            self._ensure_ids_view(hex_ids, view_name="query_ids")
            geom_expr = self._transform_clause("r.geometry", 4326, crs)
            sql = f"""
                SELECT
                    MIN(ST_XMin({geom_expr})) AS minx,
                    MIN(ST_YMin({geom_expr})) AS miny,
                    MAX(ST_XMax({geom_expr})) AS maxx,
                    MAX(ST_YMax({geom_expr})) AS maxy
                FROM query_ids q
                JOIN {view} r ON q.region_id = r.region_id
            """
            row = self._ctx.sql(sql).to_pandas().iloc[0]
            return float(row["minx"]), float(row["miny"]), float(row["maxx"]), float(row["maxy"])

        # GeoPandas fallback
        filtered = self._load_gdf_fallback(hex_ids, resolution)
        if crs != 4326:
            filtered = filtered.to_crs(epsg=crs)
        minx, miny, maxx, maxy = filtered.total_bounds
        return float(minx), float(miny), float(maxx), float(maxy)

    def neighbours(
        self,
        hex_ids: HexIDs,
        resolution: int,
        k: int = 1,
    ) -> dict[str, set[str]]:
        """Return k-ring neighbours for each hex ID (topology only, no geometry).

        Uses SRAI ``H3Neighbourhood`` for all topology lookups.  No SedonaDB
        call is made; this method is always available regardless of backend.

        Parameters
        ----------
        hex_ids:
            Hex ID strings to look up.
        resolution:
            H3 resolution of the input hex IDs.  Not used in the SRAI call
            itself, but retained in the signature for consistency with other
            methods and for future validation.
        k:
            Neighbourhood distance.  ``k=1`` returns the 6 immediately
            adjacent hexagons; ``k=2`` returns all hexagons within 2 steps,
            excluding the center (18 total for interior hexagons), etc.

        Returns
        -------
        dict[str, set[str]]
            Mapping from each input hex ID to the set of neighbour hex IDs
            within distance *k*.  The center hex is not included.
        """
        nh = H3Neighbourhood()
        ids_arr = np.asarray(hex_ids, dtype=str)
        return {
            hex_id: nh.get_neighbours_up_to_distance(hex_id, k, unchecked=True)
            for hex_id in ids_arr
        }

    def neighbours_geometry(
        self,
        hex_ids: HexIDs,
        resolution: int,
        k: int = 1,
        crs: int = 4326,
    ):
        """Return a GeoDataFrame with polygon geometry for all k-ring neighbours.

        Combines SRAI topology (``H3Neighbourhood``) with a single bulk
        geometry lookup via ``geometry()``.  All unique neighbour IDs are
        collected first; then a single SedonaDB or GeoPandas query retrieves
        their geometries — no per-hex queries.

        Parameters
        ----------
        hex_ids:
            Hex ID strings whose neighbours to look up.
        resolution:
            H3 resolution of the input hex IDs.
        k:
            Neighbourhood distance.  See ``neighbours()`` for semantics.
        crs:
            Target EPSG code.  Defaults to 4326 (WGS84).

        Returns
        -------
        geopandas.GeoDataFrame
            Indexed by region_id, containing polygon geometry for all unique
            neighbour hexagons (excluding the input hex IDs themselves).
        """
        neighbour_map = self.neighbours(hex_ids, resolution, k)
        unique_neighbours = list(
            {nbr for nbrs in neighbour_map.values() for nbr in nbrs}
        )
        if not unique_neighbours:
            import geopandas as gpd
            empty = gpd.GeoDataFrame(columns=["geometry"], crs=crs)
            empty.index.name = "region_id"
            return empty
        return self.geometry(unique_neighbours, resolution, crs)

    def k_ring_centroids(
        self,
        hex_ids: HexIDs,
        resolution: int,
        k: int = 1,
        crs: int = 4326,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return flat centroid arrays for all unique k-ring neighbours.

        Combines SRAI topology with a single bulk centroid lookup via
        ``centroids()``.  All unique neighbour IDs are collected first; then
        a single SedonaDB or GeoPandas query retrieves their centroids.

        Parameters
        ----------
        hex_ids:
            Hex ID strings whose neighbours to look up.
        resolution:
            H3 resolution of the input hex IDs.
        k:
            Neighbourhood distance.  See ``neighbours()`` for semantics.
        crs:
            Target EPSG code.  Defaults to 4326 (WGS84).

        Returns
        -------
        (neighbour_ids, cx, cy):
            Three arrays of equal length.  ``neighbour_ids`` is a string
            array of unique neighbour hex IDs; ``cx`` and ``cy`` are float64
            arrays of centroid coordinates in the requested CRS.  Order is
            determined by the bulk centroid query and may not match input
            order.
        """
        neighbour_map = self.neighbours(hex_ids, resolution, k)
        unique_neighbours = list(
            {nbr for nbrs in neighbour_map.values() for nbr in nbrs}
        )
        if not unique_neighbours:
            empty = np.array([], dtype=str)
            return empty, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        ids_arr = np.asarray(unique_neighbours, dtype=str)
        cx, cy = self.centroids(ids_arr, resolution, crs)
        return ids_arr, cx, cy
