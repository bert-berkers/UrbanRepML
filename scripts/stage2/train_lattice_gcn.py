"""
Train a simple GCN on the H3 hexagonal lattice for self-supervised embedding fusion.

Uses reconstruction loss (encode -> decode -> MSE) to learn graph-aware embeddings.
The encoder output is saved as the fused embedding. This validates whether explicit
message-passing graph convolutions improve over ring aggregation's k-ring averaging.

Lifetime: durable
Stage: Stage 2 (fusion)

Usage:
    python scripts/stage2/train_lattice_gcn.py --study-area netherlands --resolution 9
    python scripts/stage2/train_lattice_gcn.py --study-area netherlands --epochs 200 --hidden-dim 256
"""

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from stage2_fusion.models.lattice_gcn import LatticeGCN
from utils.paths import StudyAreaPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_edge_index_from_neighbourhood(
    neighbourhood, hex_ids: list[str]
) -> torch.Tensor:
    """Build PyG edge_index from an SRAI H3Neighbourhood object.

    Iterates over all hexagons, queries their direct neighbours via SRAI,
    and builds a bidirectional edge_index tensor. Filters to only include
    edges between hexagons in the provided hex_ids set.

    Args:
        neighbourhood: SRAI H3Neighbourhood instance (loaded from pickle).
        hex_ids: List of H3 hex IDs to include (defines node ordering).

    Returns:
        edge_index tensor of shape [2, num_edges].
    """
    hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
    hex_set = set(hex_ids)

    sources = []
    targets = []

    for hex_id in hex_ids:
        neighbors = neighbourhood.get_neighbours(hex_id)
        src_idx = hex_to_idx[hex_id]
        for nb in neighbors:
            if nb in hex_set:
                sources.append(src_idx)
                targets.append(hex_to_idx[nb])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return edge_index


def main():
    parser = argparse.ArgumentParser(
        description="Train LatticeGCN for self-supervised embedding fusion"
    )
    parser.add_argument("--study-area", default="netherlands")
    parser.add_argument("--resolution", type=int, default=9)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'auto', 'cuda', or 'cpu'",
    )
    args = parser.parse_args()

    paths = StudyAreaPaths(args.study_area)

    # ---- Resolve device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # ---- Load concat embeddings ----
    emb_path = paths.fused_embedding_file("concat", args.resolution, args.year)
    logger.info(f"Loading embeddings from {emb_path}")
    emb_df = pd.read_parquet(emb_path)
    logger.info(f"Embeddings shape: {emb_df.shape}")

    hex_ids = list(emb_df.index)
    input_dim = emb_df.shape[1]

    # ---- Load neighbourhood and build graph ----
    nb_path = (
        paths.neighbourhood_dir()
        / f"{args.study_area}_res{args.resolution}_neighbourhood.pkl"
    )
    logger.info(f"Loading neighbourhood from {nb_path}")
    with open(nb_path, "rb") as f:
        neighbourhood = pickle.load(f)

    logger.info(f"Building edge_index for {len(hex_ids)} hexagons...")
    t0 = time.time()
    edge_index = build_edge_index_from_neighbourhood(neighbourhood, hex_ids)
    build_time = time.time() - t0
    logger.info(
        f"Edge index built: {edge_index.shape[1]} directed edges in {build_time:.1f}s"
    )
    logger.info(
        f"Average degree: {edge_index.shape[1] / len(hex_ids):.1f}"
    )

    # ---- Create PyG Data object ----
    x = torch.tensor(emb_df.values, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index).to(device)
    logger.info(f"PyG Data: {data}")

    # ---- Create model ----
    model = LatticeGCN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Architecture: {input_dim} -> {args.hidden_dim} (x{args.num_layers - 1}) -> {args.embedding_dim}")

    # ---- Training loop ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    logger.info(f"Starting training for {args.epochs} epochs...")
    t_train_start = time.time()

    best_loss = float("inf")
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        z, x_hat = model(data.x, data.edge_index)
        loss = loss_fn(x_hat, data.x)

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:4d}/{args.epochs}  loss={loss_val:.6f}  best={best_loss:.6f}")

    train_time = time.time() - t_train_start
    logger.info(f"Training complete in {train_time:.1f}s ({train_time / args.epochs:.2f}s/epoch)")
    logger.info(f"Final loss: {loss_history[-1]:.6f}  Best loss: {best_loss:.6f}")

    # ---- Extract and save embeddings ----
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index).cpu().numpy()

    emb_columns = [f"gcn_{i:03d}" for i in range(embeddings.shape[1])]
    out_df = pd.DataFrame(embeddings, index=emb_df.index, columns=emb_columns)
    out_df.index.name = "region_id"

    out_path = paths.fused_embedding_file("gcn", args.resolution, args.year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)

    logger.info(f"Saved embeddings: {out_path}")
    logger.info(f"Output shape: {out_df.shape}")

    # ---- Save training metadata ----
    meta = {
        "study_area": args.study_area,
        "resolution": args.resolution,
        "year": args.year,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "epochs": args.epochs,
        "final_loss": float(loss_history[-1]),
        "best_loss": float(best_loss),
        "train_time_s": round(train_time, 1),
        "num_nodes": len(hex_ids),
        "num_edges": int(edge_index.shape[1]),
        "num_params": num_params,
        "device": str(device),
        "loss_history_sample": [float(loss_history[i]) for i in range(0, len(loss_history), max(1, len(loss_history) // 20))],
    }
    import json
    meta_path = out_path.parent / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Saved training metadata: {meta_path}")


if __name__ == "__main__":
    main()
