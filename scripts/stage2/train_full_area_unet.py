"""
Train FullAreaUNet on multi-resolution H3 graph data.

Self-supervised training with reconstruction + cross-scale consistency loss.
Uses MultiResolutionLoader to build graph inputs from raw concatenated
Stage 1 embeddings (781D = 64 AE + 687 POI + 30 roads).

Produces 128D fused embeddings at res9 (finest), saved as parquet.

Lifetime: durable
Stage: stage2 (fusion)
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import torch

from stage2_fusion.data.multi_resolution_loader import MultiResolutionLoader
from stage2_fusion.models.full_area_unet import FullAreaModelTrainer
from utils.paths import StudyAreaPaths, write_run_info

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FullAreaUNet on multi-resolution H3 graph"
    )
    parser.add_argument(
        "--study-area", type=str, default="netherlands",
        help="Study area name (default: netherlands)"
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Maximum training epochs (default: 500)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Hidden / output embedding dimension (default: 128)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--patience", type=int, default=100,
        help="Early stopping patience (default: 100)"
    )
    parser.add_argument(
        "--year", type=str, default="2022",
        help="Data year label (default: 2022). Use '20mix' for mixed-year runs."
    )
    parser.add_argument(
        "--resolutions", type=str, default="9,8,7",
        help="Comma-separated resolutions finest-to-coarsest (default: 9,8,7)"
    )
    parser.add_argument(
        "--feature-source", type=str, default=None,
        help="Explicit path to raw concat parquet. Overrides auto-resolution."
    )
    return parser.parse_args()


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
    print(f"FullAreaUNet Training — {args.study_area}")
    print(f"  Resolutions: {resolutions}")
    print(f"  Hidden dim:  {args.hidden_dim}")
    print(f"  LR:          {args.lr}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Patience:    {args.patience}")
    print("=" * 60)

    # ---- 1. Load multi-resolution data ----
    logger.info("Loading multi-resolution data...")
    t0 = time.time()

    loader = MultiResolutionLoader(
        study_area=args.study_area,
        resolutions=resolutions,
        year=args.year,
        feature_source=args.feature_source,
    )
    data = loader.load()
    hex_ids = data["hex_ids"]

    load_time = time.time() - t0
    logger.info(f"Data loaded in {load_time:.1f}s")

    # Detect input feature dimension from loaded data
    feature_dim = next(iter(data["features_dict"].values())).shape[1]
    logger.info(f"Input feature dim: {feature_dim}")

    # ---- 2. Move tensors to device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    features_dict = {
        k: v.to(device) for k, v in data["features_dict"].items()
    }
    edge_indices = {
        k: v.to(device) for k, v in data["edge_indices"].items()
    }
    edge_weights = {
        k: v.to(device) for k, v in data["edge_weights"].items()
    }
    mappings = {
        k: v.to(device) for k, v in data["mappings"].items()
    }

    # ---- 3. Configure and create trainer ----
    model_config = {
        "feature_dims": {"fused": feature_dim},
        "hidden_dim": args.hidden_dim,
        "output_dim": args.hidden_dim,  # output_dim == hidden_dim for 128D embeddings
        "num_convs": 4,
        "resolutions": resolutions,
    }

    checkpoint_dir = paths.checkpoints("unet")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = FullAreaModelTrainer(
        model_config=model_config,
        city_name=args.study_area,
        checkpoint_dir=checkpoint_dir,
    )

    # Log model parameter count
    n_params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # ---- 4. Train ----
    logger.info("Starting training...")
    t0 = time.time()

    best_embeddings, best_state = trainer.train(
        features_dict=features_dict,
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        mappings=mappings,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
    )

    train_time = time.time() - t0

    if best_state is None:
        logger.error("Training failed — no best state found")
        return

    logger.info(
        f"Training complete in {train_time:.1f}s "
        f"({train_time/60:.1f} min)"
    )
    logger.info(
        f"Best epoch: {best_state['epoch']}, "
        f"Best loss: {best_state['loss']:.6f}"
    )

    # ---- 5. Extract and save embeddings at ALL resolutions ----
    saved_paths = {}
    for res in sorted(resolutions, reverse=True):
        emb_tensor = best_embeddings[res].detach().cpu().numpy()
        res_hex_ids = hex_ids[res]

        logger.info(
            f"Res{res} embeddings: {emb_tensor.shape} "
            f"({len(res_hex_ids)} hexagons, {emb_tensor.shape[1]}D)"
        )

        emb_df = pd.DataFrame(
            emb_tensor,
            index=pd.Index(res_hex_ids, name="region_id"),
            columns=[f"unet_{i}" for i in range(emb_tensor.shape[1])],
        )

        out_path = paths.fused_embedding_file("unet", res, args.year)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        emb_df.to_parquet(out_path)
        saved_paths[res] = out_path
        logger.info(f"Saved res{res} embeddings to {out_path}")

    # Also save the model checkpoint
    torch.save(best_state, checkpoint_dir / "best_model.pt")
    logger.info(f"Saved best model checkpoint to {checkpoint_dir / 'best_model.pt'}")

    # ---- 6. Write run info ----
    run_dir = paths.stage2("unet")
    write_run_info(
        run_dir,
        stage="stage2",
        study_area=args.study_area,
        config={
            "model": "FullAreaUNet",
            "resolutions": resolutions,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.hidden_dim,
            "feature_dim": feature_dim,
            "num_convs": 4,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "best_epoch": best_state["epoch"],
            "best_loss": best_state["loss"],
            "train_time_sec": round(train_time, 1),
            "n_params": n_params,
        },
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Model:          FullAreaUNet")
    print(f"  Resolutions:    {resolutions}")
    print(f"  Feature dim:    {feature_dim} -> {args.hidden_dim}D embeddings")
    print(f"  Parameters:     {n_params:,}")
    print(f"  Best epoch:     {best_state['epoch']}")
    print(f"  Best loss:      {best_state['loss']:.6f}")
    print(f"  Training time:  {train_time:.1f}s ({train_time/60:.1f} min)")
    for res, p in saved_paths.items():
        n_hex = len(hex_ids[res])
        print(f"  Embeddings res{res}: ({n_hex:,}, {args.hidden_dim})")
        print(f"    Saved to:     {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
