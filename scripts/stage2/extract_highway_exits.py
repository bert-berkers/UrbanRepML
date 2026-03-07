"""
Extract multi-scale embeddings from a trained FullAreaUNet checkpoint.

The FullAreaUNet produces embeddings at 3 resolutions (res9, res8, res7) but
the training script only saves res9. This script loads the checkpoint, runs a
forward pass, and saves ALL resolution embeddings plus upsampled/blended
variants at the finest resolution.

Outputs (all in .../unet/embeddings/):
  - netherlands_res9_2022.parquet           (247K x 128D, verification)
  - netherlands_res8_2022.parquet           (NEW, ~37K x 128D)
  - netherlands_res7_2022.parquet           (NEW, ~5.6K x 128D)
  - netherlands_res9_multiscale_avg_2022.parquet   (247K x 128D)
  - netherlands_res9_multiscale_concat_2022.parquet (247K x 384D)

Lifetime: durable
Stage: stage2 (fusion)
"""

import argparse
import logging
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import torch

from stage2_fusion.data.multi_resolution_loader import MultiResolutionLoader
from stage2_fusion.models.full_area_unet import FullAreaUNet
from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract multi-scale embeddings from trained FullAreaUNet"
    )
    parser.add_argument(
        "--study-area", type=str, default="netherlands",
        help="Study area name (default: netherlands)"
    )
    parser.add_argument(
        "--year", type=int, default=2022,
        help="Data year (default: 2022)"
    )
    parser.add_argument(
        "--resolutions", type=str, default="9,8,7",
        help="Comma-separated resolutions finest-to-coarsest (default: 9,8,7)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Hidden / output embedding dimension (default: 128)"
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Verify res9 output matches existing file (default: True)"
    )
    parser.add_argument(
        "--no-verify", dest="verify", action="store_false",
        help="Skip verification against existing res9 file"
    )
    return parser.parse_args()


def load_model_and_data(args, resolutions, device):
    """Load the trained model and input data."""
    paths = StudyAreaPaths(args.study_area)

    # Load data
    logger.info("Loading multi-resolution data...")
    loader = MultiResolutionLoader(
        study_area=args.study_area,
        resolutions=resolutions,
        year=args.year,
    )
    data = loader.load()

    # Detect feature dim
    feature_dim = next(iter(data["features_dict"].values())).shape[1]
    logger.info(f"Input feature dim: {feature_dim}")

    # Build model with same config as training
    model_config = {
        "feature_dims": {"fused": feature_dim},
        "hidden_dim": args.hidden_dim,
        "output_dim": args.hidden_dim,
        "num_convs": 4,
        "resolutions": resolutions,
    }
    model = FullAreaUNet(**model_config, device=str(device))
    model.to(device)

    # Load checkpoint
    checkpoint_path = paths.checkpoints("unet") / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    logger.info(
        f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
        f"loss {checkpoint['loss']:.6f}"
    )

    # Move data to device
    features_dict = {k: v.to(device) for k, v in data["features_dict"].items()}
    edge_indices = {k: v.to(device) for k, v in data["edge_indices"].items()}
    edge_weights = {k: v.to(device) for k, v in data["edge_weights"].items()}
    mappings = {k: v.to(device) for k, v in data["mappings"].items()}

    return model, features_dict, edge_indices, edge_weights, mappings, data["hex_ids"]


def extract_embeddings(model, features_dict, edge_indices, edge_weights, mappings):
    """Run forward pass and return embeddings at all resolutions."""
    model.eval()
    with torch.no_grad():
        embeddings, _ = model(features_dict, edge_indices, edge_weights, mappings)

    # Move to CPU
    return {res: emb.detach().cpu().numpy() for res, emb in embeddings.items()}


def save_per_resolution(embeddings, hex_ids, paths, study_area, year, resolutions):
    """Save per-resolution embedding parquet files."""
    out_dir = paths.stage2("unet") / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for res in resolutions:
        emb = embeddings[res]
        hids = hex_ids[res]

        df = pd.DataFrame(
            emb,
            index=pd.Index(hids, name="region_id"),
            columns=[f"unet_{i}" for i in range(emb.shape[1])],
        )

        out_path = paths.fused_embedding_file("unet", res, year)
        df.to_parquet(out_path)
        saved_paths[res] = out_path
        logger.info(f"Saved res{res}: {df.shape} -> {out_path}")

    return saved_paths


def create_upsampled_blended(embeddings, hex_ids, paths, study_area, year, resolutions):
    """Create upsampled and blended variants at finest resolution."""
    finest_res = max(resolutions)
    sorted_res = sorted(resolutions, reverse=True)
    finest_hexes = hex_ids[finest_res]
    emb_dim = embeddings[finest_res].shape[1]
    n_hexes = len(finest_hexes)

    out_dir = paths.stage2("unet") / "embeddings"

    # Build upsampled arrays: for each coarser res, map parent embedding to children
    upsampled = {}
    upsampled[finest_res] = embeddings[finest_res]  # identity

    for res in sorted_res[1:]:  # skip finest
        # Build hex -> embedding lookup for this resolution
        res_hexes = hex_ids[res]
        res_emb = embeddings[res]
        hex_to_emb = {h: res_emb[i] for i, h in enumerate(res_hexes)}

        # For each finest-res hex, find its ancestor at this resolution
        upsamp = np.zeros((n_hexes, emb_dim), dtype=np.float32)
        for i, hex_id in enumerate(finest_hexes):
            parent = h3.cell_to_parent(hex_id, res)
            if parent in hex_to_emb:
                upsamp[i] = hex_to_emb[parent]
            # else: stays zero (edge hexes without parent in study area)

        upsampled[res] = upsamp
        logger.info(
            f"Upsampled res{res} -> res{finest_res}: "
            f"{n_hexes} hexagons, {emb_dim}D"
        )

    # Average blend: (res9 + res8_up + res7_up) / 3
    avg_emb = np.mean(
        [upsampled[res] for res in sorted_res],
        axis=0,
    )
    avg_df = pd.DataFrame(
        avg_emb,
        index=pd.Index(finest_hexes, name="region_id"),
        columns=[f"unet_{i}" for i in range(emb_dim)],
    )
    avg_path = out_dir / f"{study_area}_res{finest_res}_multiscale_avg_{year}.parquet"
    avg_df.to_parquet(avg_path)
    logger.info(f"Saved multiscale avg: {avg_df.shape} -> {avg_path}")

    # Concat blend: [res9; res8_up; res7_up]
    concat_parts = [upsampled[res] for res in sorted_res]
    concat_emb = np.concatenate(concat_parts, axis=1)

    # Column names: unet_res9_0..127, unet_res8_0..127, unet_res7_0..127
    concat_cols = []
    for res in sorted_res:
        concat_cols.extend([f"unet_res{res}_{i}" for i in range(emb_dim)])

    concat_df = pd.DataFrame(
        concat_emb,
        index=pd.Index(finest_hexes, name="region_id"),
        columns=concat_cols,
    )
    concat_path = out_dir / f"{study_area}_res{finest_res}_multiscale_concat_{year}.parquet"
    concat_df.to_parquet(concat_path)
    logger.info(f"Saved multiscale concat: {concat_df.shape} -> {concat_path}")

    return avg_path, concat_path


def verify_res9(embeddings, hex_ids, paths, year):
    """Verify that extracted res9 matches existing saved file."""
    finest_res = max(embeddings.keys())
    existing_path = paths.fused_embedding_file("unet", finest_res, year)

    if not existing_path.exists():
        logger.warning(f"No existing file to verify against: {existing_path}")
        return False

    existing_df = pd.read_parquet(existing_path)
    new_emb = embeddings[finest_res]
    new_hexes = hex_ids[finest_res]

    # Check shape
    if existing_df.shape[0] != len(new_hexes):
        logger.error(
            f"Shape mismatch: existing {existing_df.shape[0]} vs new {len(new_hexes)}"
        )
        return False

    if existing_df.shape[1] != new_emb.shape[1]:
        logger.error(
            f"Dim mismatch: existing {existing_df.shape[1]} vs new {new_emb.shape[1]}"
        )
        return False

    # Check index alignment
    if list(existing_df.index) != new_hexes:
        logger.warning("Hex ordering differs — comparing aligned values")
        # Align by index
        new_df = pd.DataFrame(
            new_emb,
            index=pd.Index(new_hexes, name="region_id"),
            columns=existing_df.columns,
        )
        common = existing_df.index.intersection(new_df.index)
        max_diff = np.abs(existing_df.loc[common].values - new_df.loc[common].values).max()
    else:
        max_diff = np.abs(existing_df.values - new_emb).max()

    logger.info(f"Max absolute difference from existing res{finest_res}: {max_diff:.2e}")

    if max_diff < 1e-4:
        logger.info("PASS: Extracted res9 matches existing file")
        return True
    else:
        logger.warning(
            f"WARN: Max diff {max_diff:.2e} exceeds 1e-4 threshold. "
            "This may be due to floating-point differences between training "
            "best_embeddings cache vs checkpoint reload + forward pass."
        )
        return False


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    resolutions = [int(r) for r in args.resolutions.split(",")]
    finest_res = max(resolutions)
    paths = StudyAreaPaths(args.study_area)

    print("=" * 60)
    print(f"Extract Highway Exits — {args.study_area}")
    print(f"  Resolutions: {resolutions}")
    print(f"  Hidden dim:  {args.hidden_dim}")
    print("=" * 60)

    # 1. Load model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    t0 = time.time()
    model, features_dict, edge_indices, edge_weights, mappings, hex_ids = \
        load_model_and_data(args, resolutions, device)
    logger.info(f"Load time: {time.time() - t0:.1f}s")

    # 2. Forward pass — extract all resolution embeddings
    t0 = time.time()
    embeddings = extract_embeddings(
        model, features_dict, edge_indices, edge_weights, mappings
    )
    logger.info(f"Forward pass time: {time.time() - t0:.1f}s")

    for res in sorted(resolutions, reverse=True):
        logger.info(f"  res{res}: {embeddings[res].shape}")

    # 3. Verify res9 matches existing (before overwriting)
    if args.verify:
        verify_res9(embeddings, hex_ids, paths, args.year)

    # 4. Save per-resolution embeddings
    saved_paths = save_per_resolution(
        embeddings, hex_ids, paths, args.study_area, args.year, resolutions
    )

    # 5. Create upsampled + blended variants
    avg_path, concat_path = create_upsampled_blended(
        embeddings, hex_ids, paths, args.study_area, args.year, resolutions
    )

    # Summary
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    for res in sorted(resolutions, reverse=True):
        shape = embeddings[res].shape
        print(f"  res{res}: {shape[0]:>8,} hexagons x {shape[1]}D -> {saved_paths[res].name}")
    print(f"  avg:    {embeddings[finest_res].shape[0]:>8,} hexagons x {embeddings[finest_res].shape[1]}D -> {avg_path.name}")
    concat_dim = embeddings[finest_res].shape[1] * len(resolutions)
    print(f"  concat: {embeddings[finest_res].shape[0]:>8,} hexagons x {concat_dim}D -> {concat_path.name}")
    print(f"\n  Output dir: {paths.stage2('unet') / 'embeddings'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
