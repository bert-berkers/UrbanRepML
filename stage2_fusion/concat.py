"""
Minimal stage-2 fusion: concatenate stage-1 modality embeddings.

Inner join by default; sparse modalities (e.g. roads) use left join with
zero-fill so that missing coverage does not drop hexagons from denser modalities.

Usage::

    python -m stage2_fusion.concat --modalities alphaearth,poi --study-area netherlands
    python -m stage2_fusion.concat --modalities alphaearth,poi,roads --pca 64
"""

import argparse
import logging
import sys

import pandas as pd

from stage1_modalities import MODALITY_PREFIXES, SUB_EMBEDDER_MAP
from utils import StudyAreaPaths
from utils.paths import write_run_info

logger = logging.getLogger(__name__)

# Modalities that use left join + zero-fill instead of inner join.
# "No data" for these modalities is semantically meaningful (e.g. no roads = zero vector).
SPARSE_MODALITIES: set[str] = {"roads", "gtfs"}


def _load_modality(paths: StudyAreaPaths, modality: str, resolution: int, year: int) -> pd.DataFrame:
    """Load a single modality embedding parquet and normalize its index to ``region_id``.

    Some stage-1 processors (e.g. AlphaEarth) write the H3 index under
    ``h3_index`` instead of the canonical ``region_id``.  This function
    normalizes any known variant so downstream joins always work on a
    common index name.
    """
    # Resolve sub-embedder modalities (e.g. "hex2vec" -> poi/hex2vec/)
    parent_modality, sub_embedder = SUB_EMBEDDER_MAP.get(modality, (modality, None))

    path = paths.embedding_file(parent_modality, resolution, year, sub_embedder=sub_embedder)
    if not path.exists():
        # Try common fallback years: "latest" (Overpass-sourced) and "2022" (GEE-sourced)
        tried = [path]
        for fallback_year in ("latest", "2022"):
            if str(fallback_year) == str(year):
                continue
            fallback = paths.embedding_file(parent_modality, resolution, fallback_year, sub_embedder=sub_embedder)
            if fallback.exists():
                logger.info("  %s: year=%s not found, falling back to '%s'", modality, year, fallback_year)
                path = fallback
                break
            tried.append(fallback)
        else:
            raise FileNotFoundError(
                f"Embedding file not found for {modality}: tried {[str(p) for p in tried]}"
            )
    df = pd.read_parquet(path)

    # --- Normalize index name to ``region_id`` ---
    # Case 1: h3_index is a column (index is default RangeIndex)
    if "h3_index" in df.columns and df.index.name != "region_id":
        df = df.set_index("h3_index")
        df.index.name = "region_id"
    # Case 2: h3_index is the index name
    elif df.index.name == "h3_index":
        df.index.name = "region_id"
    # Case 3: region_id is a column (index is default RangeIndex)
    elif "region_id" in df.columns and df.index.name != "region_id":
        df = df.set_index("region_id")

    if df.index.name != "region_id":
        raise ValueError(f"{modality}: expected index name 'region_id', got '{df.index.name}'")

    # Drop metadata columns that are not embedding features
    _META_COLS = {"pixel_count", "tile_count", "h3_resolution", "geometry"}
    drop = [c for c in df.columns if c in _META_COLS]
    if drop:
        logger.info("  %s: dropping metadata columns %s", modality, drop)
        df = df.drop(columns=drop)

    return df


def _validate_prefixes(df: pd.DataFrame, modalities: list[str]) -> None:
    """Log column breakdown per modality; warn on unexpected prefixes."""
    known_prefixes = {MODALITY_PREFIXES[m] for m in modalities if m in MODALITY_PREFIXES}
    for m in modalities:
        prefix = MODALITY_PREFIXES.get(m)
        if prefix is None:
            logger.warning("No known prefix for modality '%s' -- skipping prefix check", m)
            continue
        cols = [c for c in df.columns if c.startswith(prefix)]
        logger.info("  %-12s  prefix=%s  cols=%d", m, prefix, len(cols))

    unexpected = [
        c for c in df.columns
        if not any(c.startswith(p) for p in known_prefixes)
    ]
    if unexpected:
        logger.warning("Columns with unknown prefixes (kept anyway): %s", unexpected[:10])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Concatenate stage-1 embeddings (inner join; sparse modalities use left join)")
    parser.add_argument("--modalities", required=True, help="Comma-separated modality names")
    parser.add_argument("--study-area", default="netherlands")
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--year", type=str, default="2022")
    parser.add_argument("--pca", type=int, default=None, help="PCA components after concat")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    modalities = [m.strip() for m in args.modalities.split(",")]
    paths = StudyAreaPaths(args.study_area)

    # --- Load & join ---
    logger.info("Loading %d modalities for %s res%d year=%s", len(modalities), args.study_area, args.resolution, args.year)
    frames: list[pd.DataFrame] = []
    for mod in modalities:
        df = _load_modality(paths, mod, args.resolution, args.year)
        logger.info("  %-12s  %d hexagons  %d cols", mod, len(df), len(df.columns))
        frames.append(df)

    fused = frames[0]
    initial_count = len(fused)
    logger.info("Starting join chain with %d hexagons (from %s)", initial_count, modalities[0])
    for i, df in enumerate(frames[1:], start=1):
        mod = modalities[i]
        pre_join = len(fused)

        if mod in SPARSE_MODALITIES:
            # Left join: keep all hexagons from the accumulated frame,
            # zero-fill where this modality has no data.
            fused = fused.join(df, how="left")
            new_cols = df.columns.tolist()
            n_missing = fused[new_cols].isna().any(axis=1).sum()
            fused[new_cols] = fused[new_cols].fillna(0.0)
            logger.info(
                "  After LEFT joining %s: %d hexagons (%d had data, %d zero-filled)",
                mod, len(fused), len(fused) - n_missing, n_missing,
            )
        else:
            fused = fused.join(df, how="inner")
            dropped = pre_join - len(fused)
            logger.info(
                "  After joining %s: %d hexagons (%d dropped, %d survived from %s side)",
                mod, len(fused), dropped, len(fused), mod,
            )

    total_dropped = initial_count - len(fused)
    coverage_loss_pct = (total_dropped / initial_count * 100) if initial_count > 0 else 0.0
    logger.info(
        "Final: %d hexagons, %d columns (lost %d hexagons, %.1f%% coverage loss)",
        len(fused), len(fused.columns), total_dropped, coverage_loss_pct
    )
    if fused.empty:
        logger.error("Join produced 0 hexagons -- check that modalities share region_ids")
        sys.exit(1)

    # --- Validate column prefixes ---
    _validate_prefixes(fused, modalities)

    # --- Optional PCA ---
    if args.pca is not None:
        from sklearn.decomposition import PCA

        n = min(args.pca, len(fused.columns), len(fused))
        logger.info("Applying PCA: %d -> %d components", len(fused.columns), n)
        pca = PCA(n_components=n)
        reduced = pca.fit_transform(fused.values)
        col_names = [f"PC{i:03d}" for i in range(n)]
        fused = pd.DataFrame(reduced, index=fused.index, columns=col_names)
        logger.info("PCA explained variance: %.2f%%", pca.explained_variance_ratio_.sum() * 100)

    # --- Write output ---
    out_path = paths.fused_embedding_file("concat", args.resolution, args.year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(out_path)
    logger.info("Wrote %s", out_path)

    # --- Provenance ---
    run_dir = paths.stage2("concat")
    write_run_info(
        run_dir,
        stage="stage2",
        study_area=args.study_area,
        config={
            "method": "concat",
            "modalities": modalities,
            "resolution": args.resolution,
            "year": args.year,
            "pca_components": args.pca,
            "n_hexagons": len(fused),
            "n_columns": len(fused.columns),
        },
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
