"""
Train FullAreaUNet on multi-resolution H3 graph data.

Self-supervised training with reconstruction (weight 1.0) + cross-scale
consistency loss (weight 0.3). Uses MultiResolutionLoader to build graph
inputs from raw concatenated Stage 1 embeddings.

Pyramid U-Net with dims [64, 128, 256] (fine->mid->coarse).
CosineAnnealingWarmRestarts schedule (3 restarts, eta_min = lr/50).
Produces dim_fine-D fused embeddings at all resolutions, saved as parquet.
Checkpoints versioned as best_model_{year}_{dim}D_{date}.pt.

Lifetime: durable
Stage: stage2 (fusion)
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
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
        "--dims", type=str, default="64,128,256",
        help="Pyramid dims fine,mid,coarse (default: 64,128,256)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2,
        help="Learning rate / max_lr for scheduler (default: 1e-2)"
    )
    parser.add_argument(
        "--patience", type=int, default=100,
        help="Early stopping patience (default: 100)"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=50,
        help="Linear LR warmup epochs (default: 50)"
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
    parser.add_argument(
        "--accessibility-graph", type=str, default=None,
        help="Path to accessibility Parquet for finest resolution edges "
             "(e.g. data/study_areas/netherlands/accessibility/walk_res9.parquet)"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Enable Weights & Biases logging (default: off)"
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None,
        help="Custom wandb run name. Auto-generated if not set."
    )
    parser.add_argument(
        "--supervised", action="store_true", default=False,
        help="Enable supervised leefbaarometer prediction heads "
             "(Levering semantic bottleneck + Kendall loss weighting)"
    )
    parser.add_argument(
        "--target-year", type=int, default=2022,
        help="Leefbaarometer target year (default: 2022). "
             "Only used when --supervised is set."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    resolutions = [int(r) for r in args.resolutions.split(",")]
    dims = [int(d) for d in args.dims.split(",")]
    finest_res = max(resolutions)
    paths = StudyAreaPaths(args.study_area)

    print("=" * 60)
    print(f"FullAreaUNet Training — {args.study_area}")
    print(f"  Resolutions: {resolutions}")
    print(f"  Pyramid dims: {dims} (fine -> mid -> coarse)")
    print(f"  LR:          {args.lr}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Patience:    {args.patience}")
    if args.accessibility_graph:
        print(f"  Acc. graph:  {args.accessibility_graph}")
    if args.supervised:
        print(f"  Supervised:  YES (target year {args.target_year})")
    print("=" * 60)

    # ---- 1. Load multi-resolution data ----
    logger.info("Loading multi-resolution data...")
    t0 = time.time()

    loader = MultiResolutionLoader(
        study_area=args.study_area,
        resolutions=resolutions,
        year=args.year,
        feature_source=args.feature_source,
        accessibility_graph=args.accessibility_graph,
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

    # ---- 2b. Load supervised targets (optional) ----
    target_data = None
    if args.supervised:
        target_path = paths.target_file("leefbaarometer", finest_res, args.target_year)
        logger.info(f"Loading leefbaarometer targets from {target_path}")

        target_df = pd.read_parquet(target_path)

        # Domain columns: fys, onv, soc, vrz, won (NOT lbm -- that's derived via semantic bottleneck)
        domain_cols = ["fys", "onv", "soc", "vrz", "won"]
        lbm_col = "lbm"

        # Align targets to hex_ids[finest_res] via index matching, preserving hex_ids order
        hex_ids_finest = hex_ids[finest_res]
        hex_series = pd.Series(range(len(hex_ids_finest)), index=hex_ids_finest, name="idx")

        # Build mask: True for hexes that have leefbaarometer data
        target_mask = torch.zeros(len(hex_ids_finest), dtype=torch.bool)
        matched = hex_series.index.intersection(target_df.index)
        matched_indices = hex_series.loc[matched].values
        target_mask[matched_indices] = True

        # Extract target values for masked hexes, in the order they appear in hex_ids
        target_values = target_df.reindex([h for h in hex_ids_finest if h in target_df.index])

        domain_tensor = torch.tensor(
            target_values[domain_cols].values, dtype=torch.float32
        ).to(device)
        lbm_tensor = torch.tensor(
            target_values[[lbm_col]].values, dtype=torch.float32
        ).to(device)

        target_data = {
            'domain': domain_tensor,
            'lbm': lbm_tensor,
            'mask': target_mask.to(device),
        }

        n_valid = target_mask.sum().item()
        assert target_mask.sum() == len(target_data['domain']), (
            f"Mask count ({target_mask.sum()}) != target rows ({len(target_data['domain'])})"
        )
        logger.info(
            f"Supervised targets loaded: {n_valid:,} / {len(hex_ids_finest):,} "
            f"hexagons have leefbaarometer data ({100*n_valid/len(hex_ids_finest):.1f}%)"
        )
        print(f"  Supervised:  {n_valid:,} / {len(hex_ids_finest):,} hexagons "
              f"({100*n_valid/len(hex_ids_finest):.1f}%)")

    # ---- 3. Configure and create trainer ----
    model_config = {
        "feature_dims": {"fused": feature_dim},
        "dims": dims,
        "num_convs": 10,
        "resolutions": resolutions,
    }

    checkpoint_dir = paths.checkpoints("unet")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = FullAreaModelTrainer(
        model_config=model_config,
        city_name=args.study_area,
        checkpoint_dir=checkpoint_dir,
        year=args.year,
        supervised=args.supervised,
    )

    # Log model parameter count
    n_params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # ---- 4. Train ----
    logger.info("Starting training...")
    t0 = time.time()

    # Build wandb config if --wandb is set
    wandb_config = None
    if args.wandb:
        # Auto-generate run name: unet-{dims}-{year}[-accessibility]
        if args.wandb_name:
            run_name = args.wandb_name
        else:
            dims_str = "-".join(str(d) for d in dims)
            run_name = f"unet-{dims_str}-{args.year}"
            if args.accessibility_graph:
                run_name += "-accessibility"
            if args.supervised:
                run_name += "-supervised"
        wandb_config = {
            "run_name": run_name,
            "study_area": args.study_area,
            "year": args.year,
            "accessibility_graph": args.accessibility_graph,
            "feature_dim": feature_dim,
            "n_params": n_params,
            "supervised": args.supervised,
            "target_year": args.target_year if args.supervised else None,
        }

    train_result = trainer.train(
        features_dict=features_dict,
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        mappings=mappings,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        wandb_config=wandb_config,
        target_data=target_data,
    )

    train_time = time.time() - t0

    best_embeddings = train_result['best_embeddings']
    best_state = train_result['best_state']
    loss_history = train_result['loss_history']
    best_epoch = train_result['best_epoch']

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

    # ---- 5a. Save loss history CSV ----
    if loss_history:
        loss_df = pd.DataFrame(loss_history)
        csv_path = checkpoint_dir / "training_loss_history.csv"
        loss_df.to_csv(csv_path, index=False)
        logger.info(f"Loss history saved to {csv_path}")

        # ---- 5b. Generate loss curve plot ----
        epochs_completed = loss_df["epoch"].max() + 1
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Primary axis: losses (log scale)
        ax1.semilogy(loss_df["epoch"], loss_df["total_loss"],
                     color="#1f77b4", linewidth=1.5, label="Total loss")
        ax1.semilogy(loss_df["epoch"], loss_df["reconstruction_loss"],
                     color="#ff7f0e", linewidth=1.2, linestyle="--", label="Reconstruction loss")
        ax1.semilogy(loss_df["epoch"], loss_df["consistency_loss"],
                     color="#2ca02c", linewidth=1.2, linestyle=":", label="Consistency loss")

        # Mark best epoch
        if best_epoch >= 0:
            ax1.axvline(x=best_epoch, color="red", linewidth=1.0,
                        linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (log scale)")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, which="both", alpha=0.3)

        # Secondary axis: learning rate
        ax2 = ax1.twinx()
        ax2.plot(loss_df["epoch"], loss_df["lr"],
                 color="#9467bd", linewidth=1.0, alpha=0.6, label="LR")
        ax2.set_ylabel("Learning rate", color="#9467bd")
        ax2.tick_params(axis="y", labelcolor="#9467bd")
        ax2.legend(loc="center right", fontsize=9)

        title = (
            f"FullAreaUNet — {args.study_area} {args.year}  |  "
            f"dims {dims}  |  "
            f"{epochs_completed} epochs  |  "
            f"best epoch {best_epoch}  |  "
            f"best loss {best_state['loss']:.4e}"
        )
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()

        plot_path = checkpoint_dir / "training_loss_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Loss curve saved to {plot_path}")

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

    # Checkpoint already saved by trainer._save_checkpoint() during training
    # (versioned + best_model.pt copy)

    # ---- 6. Write run info ----
    run_dir = paths.stage2("unet")
    write_run_info(
        run_dir,
        stage="stage2",
        study_area=args.study_area,
        config={
            "model": "FullAreaUNet",
            "resolutions": resolutions,
            "dims": dims,
            "feature_dim": feature_dim,
            "num_convs": 10,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "best_epoch": best_state["epoch"],
            "best_loss": best_state["loss"],
            "train_time_sec": round(train_time, 1),
            "n_params": n_params,
            "accessibility_graph": args.accessibility_graph,
            "supervised": args.supervised,
            "target_year": args.target_year if args.supervised else None,
        },
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Model:          FullAreaUNet")
    print(f"  Resolutions:    {resolutions}")
    print(f"  Feature dim:    {feature_dim} -> {dims[0]}D embeddings (pyramid {dims})")
    print(f"  Parameters:     {n_params:,}")
    if args.supervised:
        lc_params = sum(p.numel() for p in trainer.loss_computer.parameters())
        print(f"  Supervised:     YES ({lc_params} Kendall log_sigma params)")
    print(f"  Best epoch:     {best_state['epoch']}")
    print(f"  Best loss:      {best_state['loss']:.6f}")
    print(f"  Training time:  {train_time:.1f}s ({train_time/60:.1f} min)")
    for res, p in saved_paths.items():
        n_hex = len(hex_ids[res])
        print(f"  Embeddings res{res}: ({n_hex:,}, {dims[0]})")
        print(f"    Saved to:     {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
