# DNN Probe: Graph Neural Network Probe for Liveability Prediction

## Status: Draft

## Context

The existing linear probe (`stage3_analysis/linear_probe.py`) evaluates whether AlphaEarth embeddings encode Dutch liveability signals by fitting per-hexagon ElasticNet regressions with spatial block CV. It treats each hexagon independently -- the 64-dim embedding at hexagon `i` is mapped to a scalar target at hexagon `i` with no information flow from neighbors.

This misses a key hypothesis: liveability is spatially structured. A hexagon's liveability depends not just on its own characteristics but on its neighborhood context. The linear probe cannot test this hypothesis because it has no mechanism for spatial information exchange.

A Graph Neural Network (GNN) probe fills this gap. By operating on the H3 adjacency graph, a GNN can learn spatially-aware transformations where each hexagon's prediction is informed by its local neighborhood. If the GNN significantly outperforms the linear probe, it provides evidence that:
1. Spatial context in embeddings matters for liveability prediction
2. The embedding space has neighborhood-level structure worth exploiting
3. Stage 2 fusion models (which operate on graphs) have room to improve over per-hexagon approaches

This probe should remain small -- 2-3 GCN/GAT layers, not a full U-Net. The goal is to measure the *marginal value of spatial context*, not to build the best possible predictor.

### Current Linear Probe Results (reference)

The linear probe produces `TargetResult` objects consumed by `LinearProbeVisualizer`. The DNN probe must produce output in the same format so all existing visualizations work without modification.

## Decision

Build a GCN/GAT-based graph neural network probe in `stage3_analysis/dnn_probe.py` that:
- Constructs an H3 adjacency graph using SRAI `H3Neighbourhood`
- Trains a shallow GNN (2-3 layers) per Leefbaarometer target
- Uses the same spatial block CV as the linear probe (node masking, not graph splitting)
- Outputs `TargetResult`-compatible results for direct comparison

## Architecture

### Data Flow

```
AlphaEarth Embeddings (parquet)     Leefbaarometer Targets (parquet)
       |                                      |
       v                                      v
  load_and_join_data() -- inner join on region_id/h3_index
       |
       v
  GeoDataFrame: [N hexagons x (64 features + 6 targets + geometry)]
       |
       +----------> create_spatial_blocks() --> fold assignments [N]
       |
       +----------> build_h3_graph() --> PyG Data object
       |                |
       |                +-- node features: [N, 64] (standardized per fold)
       |                +-- edge_index: [2, E] (H3 1-ring adjacency)
       |                +-- y: [N] (one target at a time)
       |
       v
  For each target:
       |
       +-- Optuna HPO (minimize mean val RMSE across folds)
       |     |
       |     +-- For each fold: mask nodes into train/val sets
       |     +-- Train GNN with early stopping on val loss
       |     +-- Return mean val RMSE
       |
       +-- Final CV evaluation with best hyperparameters
       |     |
       |     +-- For each fold: train -> predict on held-out nodes
       |     +-- Collect out-of-fold predictions
       |
       +-- Assemble TargetResult (compatible with LinearProbeVisualizer)
       |
       v
  Save results + comparison table (linear vs DNN)
```

### Model Architecture (ASCII)

```
Input: x_i in R^64 for each hexagon i
       edge_index from H3 1-ring adjacency

  x_i [N, 64]
    |
    v
  Linear(64, hidden_dim)  +  LayerNorm  +  GELU        -- input projection
    |
    v
  h_i^(0) [N, hidden_dim]
    |
    v
  +--[ GCN/GAT Layer 1 ]--+
  |  Conv(hidden_dim, hidden_dim)                       -- spatial message passing
  |  LayerNorm + GELU + Dropout                         -- 1-hop neighborhood
  |                        |
  +------- residual -------+                            -- skip connection
    |
    v
  h_i^(1) [N, hidden_dim]
    |
    v
  +--[ GCN/GAT Layer 2 ]--+
  |  Conv(hidden_dim, hidden_dim)                       -- 2-hop effective RF
  |  LayerNorm + GELU + Dropout
  |                        |
  +------- residual -------+
    |
    v
  h_i^(2) [N, hidden_dim]
    |
    v
  (optional: GCN/GAT Layer 3, same pattern)             -- 3-hop effective RF
    |
    v
  h_i^(L) [N, hidden_dim]
    |
    v
  Linear(hidden_dim, 1)                                 -- per-node regression head
    |
    v
  y_hat_i [N, 1]                                        -- predicted liveability
```

With 2 GCN layers, each hexagon's prediction is informed by its 2-hop neighborhood (~18 hexagons at res10). With 3 layers, ~36 hexagons. This is a local probe, not a global one.

### Spatial Block CV with Node Masking

The linear probe splits *rows* into train/test. The GNN probe cannot do this because the graph must remain intact for message passing. Instead, we mask *nodes*:

```
Full graph: all N nodes, all E edges remain connected always

Fold k:
  train_mask: nodes NOT in spatial block k   (used for loss computation)
  val_mask:   nodes IN spatial block k       (used for early stopping + OOF predictions)

Training loop:
  1. Forward pass on FULL graph (all nodes participate in message passing)
  2. Compute loss ONLY on train_mask nodes
  3. Evaluate ONLY on val_mask nodes for early stopping
  4. Store predictions for val_mask nodes as OOF predictions
```

This is the standard transductive GNN evaluation protocol. Validation nodes still receive messages from training nodes (and vice versa), which is acceptable because:
- The spatial blocks are 25km wide, much larger than the 2-3 hop receptive field (~1-2km at res10)
- Only nodes deep inside a validation block are truly "unseen"; border nodes get some train signal
- This matches how GNN probes work in the literature (e.g., Kipf & Welling 2017, node classification benchmarks)

The alternative (inductive: remove val edges entirely) would require subgraph extraction per fold and would break the graph structure at block boundaries, creating artificial discontinuities worse than the minor leakage from transductive evaluation.

## Key Classes and Signatures

### `DNNProbeConfig` (dataclass)

```python
@dataclass
class DNNProbeConfig:
    """Configuration for DNN graph probe regression."""

    # Study area
    study_area: str = "netherlands"
    year: int = 2022
    h3_resolution: int = 10
    target_cols: List[str] = field(default_factory=lambda: list(TARGET_COLS))

    # Spatial block CV (same defaults as LinearProbeConfig)
    n_folds: int = 5
    block_width: int = 25_000   # meters
    block_height: int = 25_000  # meters
    random_state: int = 42

    # GNN architecture
    conv_type: str = "gcn"          # "gcn" or "gat"
    hidden_dim: int = 128           # Hidden dimension
    num_layers: int = 2             # Number of GCN/GAT layers (2 or 3)
    num_heads: int = 4              # GAT attention heads (ignored for GCN)
    dropout: float = 0.1
    use_residual: bool = True       # Residual connections
    use_layer_norm: bool = True     # LayerNorm after each conv

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 30              # Early stopping patience
    min_delta: float = 1e-5         # Minimum improvement for early stopping
    batch_norm_graph: bool = False  # Use GraphNorm (alternative to LayerNorm)

    # Optuna HPO
    n_trials: int = 30

    # Optuna search ranges
    hidden_dim_choices: List[int] = field(default_factory=lambda: [64, 128, 256])
    lr_low: float = 1e-4
    lr_high: float = 1e-2
    dropout_low: float = 0.0
    dropout_high: float = 0.3
    weight_decay_low: float = 1e-6
    weight_decay_high: float = 1e-3
    num_layers_choices: List[int] = field(default_factory=lambda: [2, 3])

    # LR scheduler
    scheduler: str = "cosine"       # "cosine" or "plateau"

    # Data paths (same pattern as LinearProbeConfig)
    embeddings_path: Optional[str] = None
    target_path: Optional[str] = None
    output_dir: Optional[str] = None

    # Device
    device: str = "auto"            # "auto", "cuda", "cpu"

    def __post_init__(self):
        # Same path logic as LinearProbeConfig
        if self.embeddings_path is None:
            self.embeddings_path = (
                f"data/study_areas/{self.study_area}/embeddings/alphaearth/"
                f"{self.study_area}_res{self.h3_resolution}_{self.year}.parquet"
            )
        if self.target_path is None:
            self.target_path = (
                f"data/study_areas/{self.study_area}/target/leefbaarometer/"
                f"leefbaarometer_h3res{self.h3_resolution}_{self.year}.parquet"
            )
        if self.output_dir is None:
            self.output_dir = (
                f"data/study_areas/{self.study_area}/analysis/dnn_probe"
            )
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
```

### `GNNProbeModel` (torch.nn.Module)

```python
class GNNProbeModel(nn.Module):
    """
    Shallow GCN/GAT probe for per-hexagon regression.

    Not a full U-Net -- just 2-3 message-passing layers with residual
    connections on a single-resolution H3 adjacency graph.
    """

    def __init__(
        self,
        input_dim: int,         # 64 for AlphaEarth
        hidden_dim: int,        # e.g. 128
        num_layers: int,        # 2 or 3
        conv_type: str,         # "gcn" or "gat"
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        ...

    def forward(
        self,
        x: torch.Tensor,           # [N, input_dim]
        edge_index: torch.LongTensor,  # [2, E]
    ) -> torch.Tensor:              # [N, 1]
        ...
```

### `DNNProbeRegressor` (main class)

```python
class DNNProbeRegressor:
    """
    GNN probe with spatial block CV and Optuna optimization.

    Evaluates whether spatially-aware (graph-based) transformations of
    AlphaEarth embeddings improve liveability prediction over linear probes.
    """

    def __init__(self, config: DNNProbeConfig, project_root: Optional[Path] = None):
        ...

    def load_and_join_data(self) -> gpd.GeoDataFrame:
        """Load embeddings + targets, inner join. Same as LinearProbeRegressor."""
        ...

    def build_h3_graph(self, region_ids: np.ndarray) -> torch_geometric.data.Data:
        """
        Build H3 adjacency graph using SRAI H3Neighbourhood.

        Uses H3Neighbourhood to get 1-ring neighbors for each hexagon,
        then constructs a PyG Data object with edge_index.

        SRAI compliance: uses H3Neighbourhood, not h3.grid_disk.
        """
        ...

    def create_spatial_blocks(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Create spatial block fold assignments. Same as LinearProbeRegressor."""
        ...

    def _train_one_fold(
        self,
        data: torch_geometric.data.Data,
        train_mask: torch.BoolTensor,
        val_mask: torch.BoolTensor,
        config: DNNProbeConfig,  # or specific HP values
    ) -> Tuple[GNNProbeModel, List[float]]:
        """
        Train GNN for one fold with early stopping.

        Returns trained model and validation loss history.
        """
        ...

    def run_for_target(
        self,
        target_col: str,
        X: np.ndarray,
        y_all: pd.DataFrame,
        folds: np.ndarray,
        region_ids: np.ndarray,
        graph_data: torch_geometric.data.Data,
    ) -> TargetResult:
        """Run full HPO + CV pipeline for one target. Returns TargetResult."""
        ...

    def run(self) -> Dict[str, TargetResult]:
        """Run DNN probe for all targets."""
        ...

    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """Save results in same format as linear probe."""
        ...

    def compare_with_linear(
        self,
        linear_results: Dict[str, TargetResult],
    ) -> pd.DataFrame:
        """
        Produce comparison table: linear vs DNN R2/RMSE/MAE per target.

        Returns DataFrame with columns:
            target, target_name,
            linear_r2, linear_rmse, linear_mae,
            dnn_r2, dnn_rmse, dnn_mae,
            r2_delta, rmse_delta
        """
        ...
```

### Graph Construction Detail: `build_h3_graph`

```python
def build_h3_graph(self, region_ids: np.ndarray) -> torch_geometric.data.Data:
    """
    Build PyG Data from H3 adjacency using SRAI.

    Steps:
    1. Create GeoDataFrame from region_ids with H3 polygon geometry
    2. Use SRAI H3Neighbourhood to get 1-ring neighbors
    3. Build edge_index tensor (bidirectional edges)
    4. Return PyG Data with edge_index (node features added later per fold)

    SRAI compliance note:
        Uses srai.neighbourhoods.H3Neighbourhood for neighbor lookup.
        Uses srai.h3.h3_to_geoseries for geometry (if needed).
        Does NOT use h3.grid_disk or h3.grid_ring directly.
    """
    from srai.neighbourhoods import H3Neighbourhood

    # Build regions GeoDataFrame (required by SRAI neighbourhood API)
    # Option A: use h3_to_geoseries for full polygons
    # Option B: minimal approach -- H3Neighbourhood can work from index alone
    #           if the GeoDataFrame has region_id index

    neighbourhood = H3Neighbourhood()

    # Create hex_id -> integer index mapping
    hex_to_idx = {hex_id: i for i, hex_id in enumerate(region_ids)}

    # Collect edges
    src_list, dst_list = [], []
    for hex_id in region_ids:
        # SRAI H3Neighbourhood.get_neighbours returns neighbor hex IDs
        neighbors = neighbourhood.get_neighbours(hex_id)
        for nbr in neighbors:
            if nbr in hex_to_idx:
                src_list.append(hex_to_idx[hex_id])
                dst_list.append(hex_to_idx[nbr])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    return Data(edge_index=edge_index, num_nodes=len(region_ids))
```

**Expected graph statistics** (Netherlands, res10, ~500k hexagons):
- Nodes: ~500,000
- Edges: ~3,000,000 (bidirectional, ~6 neighbors per hex)
- Average degree: ~6
- Graph construction time: ~30-60 seconds (SRAI neighbourhood lookup per hex)

**Optimization note**: The per-hexagon SRAI loop may be slow for 500k hexagons. An acceptable optimization is to use `h3.grid_disk(hex_id, 1)` for the inner loop since this is hierarchy traversal (permitted by CLAUDE.md). However, the spec prefers SRAI first. If SRAI proves too slow, the implementer should profile and document the decision to use h3-py for this specific operation.

## Hyperparameters (Optuna Search Space)

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| hidden_dim | categorical | [64, 128, 256] | 128 |
| num_layers | categorical | [2, 3] | 2 |
| learning_rate | log-uniform | [1e-4, 1e-2] | 1e-3 |
| dropout | uniform | [0.0, 0.3] | 0.1 |
| weight_decay | log-uniform | [1e-6, 1e-3] | 1e-4 |
| conv_type | categorical | ["gcn", "gat"] | "gcn" |

Fixed (not searched):
- num_heads: 4 (only used for GAT)
- max_epochs: 300
- patience: 30
- scheduler: cosine annealing

The search space is deliberately smaller than a typical GNN HPO because this is a probe, not a production model. 30 Optuna trials should suffice.

## Training Loop Detail

```python
def _train_one_fold(self, data, train_mask, val_mask, hparams):
    model = GNNProbeModel(**hparams).to(self.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams['learning_rate'],
        weight_decay=hparams['weight_decay'],
    )

    # Cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=self.config.max_epochs
    )

    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0

    data = data.to(self.device)

    for epoch in range(self.config.max_epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # [N, 1]
        loss = criterion(out[train_mask].squeeze(), data.y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = criterion(out[val_mask].squeeze(), data.y[val_mask]).item()

        # Early stopping
        if val_loss < best_val_loss - self.config.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= self.config.patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_loss
```

### Feature Standardization

Per-fold standardization (same as linear probe):
- Compute mean/std on train_mask nodes only
- Apply to all nodes before setting `data.x`
- This prevents data leakage from validation nodes into normalization stats

```python
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[train_mask] = scaler.fit_transform(X[train_mask])
X_scaled[val_mask] = scaler.transform(X[val_mask])
data.x = torch.tensor(X_scaled, dtype=torch.float32)
```

## TargetResult Compatibility

The `TargetResult` dataclass has these fields that need special handling for the GNN:

| Field | Linear Probe | DNN Probe |
|-------|-------------|-----------|
| `best_alpha` | ElasticNet alpha | Set to 0.0 (not applicable) |
| `best_l1_ratio` | ElasticNet l1_ratio | Set to 0.0 (not applicable) |
| `coefficients` | model.coef_ (64-dim) | `np.zeros(n_features)` -- GNN has no linear coefficients |
| `intercept` | model.intercept_ | 0.0 |
| `fold_metrics` | List[FoldMetrics] | Same format, populated from GNN CV |
| `oof_predictions` | OOF array [N] | Same format, from node masking |
| `actual_values` | target array [N] | Same |
| `region_ids` | hex ID array [N] | Same |
| `feature_names` | ["A00", ..., "A63"] | Same |

The `LinearProbeVisualizer` methods that read `coefficients` (bar charts, heatmap) will show all zeros for the DNN probe. This is acceptable -- the coefficient plots are meaningless for a GNN. The visualizer methods that work on `oof_predictions` and `actual_values` (scatter plots, spatial maps, fold metrics) will work correctly.

To make the comparison cleaner, the DNN probe should also save GNN-specific metadata (best hyperparameters, training curves) in a separate file, not in the `TargetResult`.

## CLI Interface

```
python -m stage3_analysis.dnn_probe --study-area netherlands [options]

Options:
  --study-area STR      Study area name (default: netherlands)
  --conv-type STR       GCN or GAT (default: gcn)
  --n-trials INT        Optuna trials (default: 30)
  --n-folds INT         Spatial CV folds (default: 5)
  --max-epochs INT      Max training epochs (default: 300)
  --hidden-dim INT      Hidden dimension, skips HPO for this param (default: search)
  --num-layers INT      Number of GCN layers, skips HPO for this param (default: search)
  --device STR          cuda/cpu/auto (default: auto)
  --compare PATH        Path to linear probe results dir for comparison table
  --skip-hpo            Use default hyperparameters, skip Optuna (for quick testing)
```

## Output Files

```
data/study_areas/netherlands/analysis/dnn_probe/
    metrics_summary.csv          # Same format as linear probe
    predictions_{target}.parquet # Same format as linear probe
    config.json                  # Config + best hyperparameters per target
    training_curves/             # Optional: loss curves per target per fold
        {target}_fold{k}.json
    comparison_linear_vs_dnn.csv # If --compare provided
```

## Memory and Runtime Estimates

For Netherlands res10 (~500k hexagons, 64 features):

| Component | Memory | Time |
|-----------|--------|------|
| Graph construction (SRAI) | ~200 MB (edge_index) | ~30-60s |
| Node features (float32) | ~125 MB (500k x 64) | - |
| GNN forward pass (128 hidden) | ~500 MB GPU | ~0.5s/epoch |
| Full training (300 epochs, early stop ~100) | ~500 MB GPU | ~1-2 min/fold |
| One target (5 folds x 30 trials) | ~500 MB GPU peak | ~30-60 min |
| All 6 targets | ~500 MB GPU peak | ~3-6 hours |

The graph fits in memory because res10 hexagons have ~6 neighbors each, so edge_index is [2, ~3M] which is ~24 MB. The bottleneck is the number of Optuna trials, not the graph size.

**CPU fallback**: If no GPU is available, each epoch will be ~5-10x slower. Consider reducing `n_trials` to 15 and `max_epochs` to 150 for CPU runs.

## Alternatives Considered

### Alternative 1: MLP (no graph structure)

A multi-layer perceptron with the same hidden dims but no graph convolutions. This would test "is non-linearity enough?" without spatial context.

**Why not primary**: Does not test the spatial hypothesis. However, it would make a useful additional baseline. The implementer may add an `--mlp-baseline` flag that runs an MLP alongside the GNN for a 3-way comparison (linear / MLP / GNN).

### Alternative 2: Full multi-resolution U-Net probe

Use the ConeBatchingUNet or FullAreaUNet architecture with frozen encoders as a probe.

**Why not**: Too heavy for a probe. The U-Net has 6 resolution levels, millions of parameters, and requires cone preprocessing. It answers a different question ("can our production model predict liveability?") rather than "does spatial context help?"

### Alternative 3: Spatial lag features + linear model

Add mean-of-neighbors features to the linear probe: for each hexagon, compute the mean embedding of its H3 neighbors and concatenate.

**Why not primary**: This is a reasonable alternative and much simpler. It tests spatial context without requiring a GNN. However, it cannot learn *which* neighbor information matters or how to combine it -- the GNN's attention (GAT) or learned message weights do this. The spatial lag approach could be a useful additional baseline.

## Consequences

### Positive
- Directly answers the question "does spatial context improve liveability prediction from embeddings?"
- Reuses existing visualization infrastructure via `TargetResult` compatibility
- Follows project conventions: SRAI for neighborhoods, PyG for graph operations, Optuna for HPO
- Small model size makes it fast to iterate
- Comparison table provides clear evidence for/against spatial approaches

### Negative
- Transductive node masking allows minor information leakage at spatial block boundaries (nodes within 2-3 hops of the block edge see some validation signal during training)
- Graph construction with SRAI may be slow for 500k hexagons; may need h3-py fallback
- GNN adds PyTorch training complexity (device management, early stopping, LR scheduling) compared to sklearn's clean API
- Zero coefficients in `TargetResult` mean some visualizer methods produce empty/meaningless plots

### Neutral
- 30 Optuna trials per target is fewer than the linear probe's 50, but GNN training is more expensive per trial so wall-clock time is similar
- The probe is intentionally underpowered relative to the stage 2 models; a negative result (GNN no better than linear) is still informative

## Implementation Notes

### Ordering
1. Implement `GNNProbeModel` (torch module, no dependencies on data loading)
2. Implement `build_h3_graph` (SRAI graph construction, test with small region set)
3. Implement `_train_one_fold` (training loop with early stopping)
4. Implement `run_for_target` (Optuna HPO + CV, producing TargetResult)
5. Implement `run` and `save_results` (full pipeline, CLI entry point)
6. Implement `compare_with_linear` (comparison table)
7. Add to `stage3_analysis/__init__.py` exports
8. Test on a single target with `--skip-hpo` before running full HPO

### Dependencies
- `torch` and `torch_geometric` (already in project dependencies)
- `srai.neighbourhoods.H3Neighbourhood` (already used in project)
- `optuna` (already used by linear probe)
- `spatialkfold` (already used by linear probe)
- No new dependencies required

### Integration with Existing Code
- Import `TargetResult`, `FoldMetrics`, `TARGET_COLS`, `TARGET_NAMES` from `stage3_analysis.linear_probe`
- Reuse `create_spatial_blocks` logic (copy or factor out into shared utility)
- Reuse `load_and_join_data` logic (copy or factor out)
- `LinearProbeVisualizer` works on `Dict[str, TargetResult]` -- pass DNN results directly

### Code Reuse Consideration
The `load_and_join_data` and `create_spatial_blocks` methods are duplicated between `LinearProbeRegressor` and `DNNProbeRegressor`. These could be factored into a shared base class or utility module. However, for a first implementation, duplication is acceptable and simpler than refactoring the existing linear probe. If a third probe type is added, refactoring becomes warranted.
