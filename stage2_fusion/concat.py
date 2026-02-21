"""
Minimal stage-2 fusion: concatenate stage-1 modality embeddings via inner join.

Usage::

    python -m stage2_fusion.concat --modalities alphaearth,poi --study-area netherlands
    python -m stage2_fusion.concat --modalities alphaearth,poi,roads --pca 64
"""

import argparse
import logging
import sys

import pandas as pd

from stage1_modalities import MODALITY_PREFIXES
from utils import StudyAreaPaths
from utils.paths import write_run_info

logger = logging.getLogger(__name__)


def _load_modality(paths: StudyAreaPaths, modality: str, resolution: int, year: int) -> pd.DataFrame:
    """Load a single modality embedding parquet and validate its index."""
    path = paths.embedding_file(modality, resolution, year)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    df = pd.read_parquet(path)
    if df.index.name != "region_id":
        raise ValueError(f"{modality}: expected index name 'region_id', got '{df.index.name}'")
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
    parser = argparse.ArgumentParser(description="Concatenate stage-1 embeddings (inner join)")
    parser.add_argument("--modalities", required=True, help="Comma-separated modality names")
    parser.add_argument("--study-area", default="netherlands")
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--pca", type=int, default=None, help="PCA components after concat")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    modalities = [m.strip() for m in args.modalities.split(",")]
    paths = StudyAreaPaths(args.study_area)

    # --- Load & join ---
    logger.info("Loading %d modalities for %s res%d year=%d", len(modalities), args.study_area, args.resolution, args.year)
    frames: list[pd.DataFrame] = []
    for mod in modalities:
        df = _load_modality(paths, mod, args.resolution, args.year)
        logger.info("  %-12s  %d hexagons  %d cols", mod, len(df), len(df.columns))
        frames.append(df)

    fused = frames[0]
    for df in frames[1:]:
        fused = fused.join(df, how="inner")

    logger.info("After inner join: %d hexagons, %d columns", len(fused), len(fused.columns))
    if fused.empty:
        logger.error("Inner join produced 0 hexagons -- check that modalities share region_ids")
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
