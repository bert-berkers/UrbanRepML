# DNN Probe: MLP Probe for Liveability Prediction

## Status: Implemented

## Context

The linear probe (`stage3_analysis/linear_probe.py`) evaluates whether AlphaEarth embeddings encode Dutch liveability signals by fitting per-hexagon ElasticNet regressions with spatial block CV. It treats each hexagon independently -- the 64-dim embedding at hexagon `i` is mapped to a scalar target with no non-linear transformation.

The DNN probe tests the next hypothesis: **does non-linearity help?** A multi-layer perceptron (MLP) with residual connections, LayerNorm, and GELU activations can learn non-linear feature combinations that ElasticNet cannot. If the MLP significantly outperforms the linear probe, it provides evidence that the embedding space contains non-linear structure relevant to liveability prediction.

This probe intentionally does NOT use graph structure. Each hexagon is an independent sample, identical to the linear probe. Graph-based spatial reasoning belongs in stage 2 (fusion), not stage 3 (analysis). The earlier GNN-based design was refactored to MLP for this reason, and because graph construction was the pipeline bottleneck (~10 min for 250k hexagons) with no clear benefit for a probe.

## Decision

Build an MLP probe in `stage3_analysis/dnn_probe.py` that:
- Trains a shallow MLP (2-3 hidden layers) per Leefbaarometer target
- Uses the same spatial block CV as the linear probe (simple train/val splitting)
- Uses fixed hyperparameters from config (no Optuna HPO -- over-engineered for a probe)
- Outputs `TargetResult`-compatible results for direct comparison with the linear probe

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
       v
  For each target:
       |
       +-- For each fold:
       |     +-- Per-fold StandardScaler (fit on train, transform both)
       |     +-- Train MLP with early stopping on val loss
       |     +-- Collect out-of-fold predictions
       |
       +-- Assemble TargetResult (compatible with LinearProbeVisualizer)
       |
       v
  Save results + optional comparison table (linear vs DNN)
```

### MLP Model

```
Input: x_i in R^64 for each hexagon i

  x_i [N, 64]
    |
    v
  Linear(64, hidden_dim)  +  LayerNorm  +  GELU        -- input projection
    |
    v
  h^(0) [N, hidden_dim]
    |
    v
  +--[ MLP Layer 1 ]------+
  |  Linear(hidden_dim, hidden_dim)                     -- non-linear transform
  |  LayerNorm + GELU + Dropout
  |                        |
  +------- residual -------+                            -- skip connection
    |
    v
  +--[ MLP Layer 2 ]------+
  |  (same pattern)
  |                        |
  +------- residual -------+
    |
    v
  Linear(hidden_dim, 1)                                 -- regression head
    |
    v
  y_hat_i [N, 1]
```

With default settings (2 layers, 128 hidden dim), the model has ~25k parameters -- intentionally small for a probe.

### Spatial Block CV

Spatial block cross-validation prevents spatial autocorrelation leakage. The `spatialkfold` library creates rectangular blocks (default 25km x 25km) projected in EPSG:28992 (RD New), randomly assigned to `n_folds` folds.

For each fold `k`:
- **train set**: all hexagons NOT in spatial block `k`
- **val set**: all hexagons IN spatial block `k`

This is standard train/val splitting -- no graph masking or transductive evaluation needed. The 25km block width is much larger than the spatial autocorrelation range of liveability scores (~5km), ensuring held-out blocks are genuinely unseen.

### Feature Standardization

Per-fold standardization prevents data leakage from validation hexagons into normalization statistics:
1. Fit `StandardScaler` on training samples only
2. Transform both training and validation samples
3. New scaler per fold (no information leaks across folds)

## TargetResult Compatibility

The `TargetResult` dataclass (defined in `linear_probe.py`) is shared between probes. Some fields have different semantics for the MLP:

| Field | Linear Probe | MLP Probe |
|-------|-------------|-----------|
| `best_alpha` | ElasticNet alpha | `0.0` (not applicable) |
| `best_l1_ratio` | ElasticNet l1_ratio | `0.0` (not applicable) |
| `coefficients` | `model.coef_` (64-dim) | `np.zeros(n_features)` -- MLP has no linear coefficients |
| `intercept` | `model.intercept_` | `0.0` |
| `fold_metrics` | List[FoldMetrics] | Same format, populated from MLP CV |
| `oof_predictions` | OOF array [N] | Same format, from fold splitting |
| `actual_values` | target array [N] | Same |
| `region_ids` | hex ID array [N] | Same |
| `feature_names` | ["A00", ..., "A63"] or ["emb_0", ...] | Same |

Visualizer methods that work correctly with MLP results:
- `plot_scatter_predicted_vs_actual` (uses `oof_predictions`)
- `plot_spatial_residuals` (uses `oof_predictions` + `region_ids`)
- `plot_fold_metrics` (uses `fold_metrics`)
- `plot_metrics_comparison` (uses `overall_r2`)

Not meaningful (coefficients are zeros):
- `plot_coefficient_bars` / `plot_coefficient_bars_faceted`
- `plot_coefficient_heatmap`
- `plot_rgb_top3_map`
- `plot_cross_target_correlation`

MLP-specific metadata (hyperparameters, training curves) is saved separately in `config.json` and `training_curves/`, not in `TargetResult`.

## Output Files

```
data/study_areas/{study_area}/stage3_analysis/dnn_probe/{run_id}/
    metrics_summary.csv              # Same format as linear probe
    predictions_{target}.parquet     # Same format as linear probe
    config.json                      # Config + hyperparameters per target
    run_info.json                    # Provenance (when run_descriptor is set)
    training_curves/                 # Val loss history per target per fold
        {target}_fold{k}.json
    comparison_linear_vs_dnn.csv     # If --compare provided
    plots/                           # If --visualize provided
```

## CLI Interface

```
python -m stage3_analysis.dnn_probe --study-area netherlands [options]

Options:
  --study-area STR      Study area name (default: netherlands)
  --n-folds INT         Spatial CV folds (default: 5)
  --max-epochs INT      Max training epochs (default: 300)
  --hidden-dim INT      Hidden dimension (default: 128)
  --num-layers INT      Number of MLP layers (default: 2)
  --device STR          cuda/cpu/auto (default: auto)
  --compare PATH        Path to linear probe results dir for comparison table
  --visualize           Generate DNN probe visualizations after saving results
```

## Memory and Runtime Estimates

For Netherlands res10 (~250k hexagons after join, 64 features):

| Component | Memory | Time |
|-----------|--------|------|
| Data loading + join | ~200 MB | ~10s |
| Spatial block creation | ~50 MB | ~5s |
| MLP forward pass (128 hidden) | ~50 MB | ~0.01s/epoch |
| One fold training (300 epochs, early stop ~80-120) | ~100 MB peak | ~10-20s |
| One target (5 folds) | ~100 MB peak | ~1-2 min |
| All 6 targets | ~100 MB peak | ~6-12 min |

CPU is fine. No GPU needed -- the MLP is small and the data fits in memory as plain tensors. This is a major improvement over the GNN design, which required ~500 MB GPU and 3-6 hours for all targets due to graph construction and Optuna HPO.

## Code Reuse Consideration

`load_and_join_data()` and `create_spatial_blocks()` are duplicated between `LinearProbeRegressor` and `DNNProbeRegressor`. These could be factored into a shared base class or utility module. For now, duplication is acceptable and simpler than refactoring the existing linear probe. If a third probe type is added, refactoring becomes warranted.

## Alternatives Considered

### Alternative 1: GNN probe (original design, rejected)

A GCN/GAT probe on the H3 adjacency graph with Optuna HPO. This was the original spec (now superseded). Rejected because:
- Graph construction belongs in stage 2 (fusion), not stage 3 (analysis)
- Graph construction was the bottleneck (~10 min for 250k hexagons)
- Transductive node masking added unnecessary complexity
- Optuna HPO was over-engineered for a probe (30 trials x 5 folds x 6 targets)
- The spatial hypothesis ("does neighborhood context help?") is better tested by stage 2 models

### Alternative 2: Spatial lag features + linear model

Add mean-of-neighbors features to the linear probe for each hexagon. Simpler than a GNN but still tests spatial context. Reasonable but tests a different hypothesis than what the MLP probe targets.

## Consequences

### Positive
- Simple, fast, CPU-friendly -- runs in minutes, not hours
- Directly answers "does non-linearity improve liveability prediction from embeddings?"
- Reuses existing visualization infrastructure via `TargetResult` compatibility
- Clean separation of concerns: per-hexagon analysis in stage 3, spatial reasoning in stage 2
- No new dependencies (dropped `torch_geometric` and `optuna` requirements)

### Negative
- Zero coefficients in `TargetResult` mean some visualizer methods produce empty plots
- Cannot answer the spatial context question (that responsibility moves to stage 2)
- Fixed hyperparameters may underfit or overfit for some targets (mitigated by early stopping)

### Neutral
- The probe is intentionally underpowered relative to stage 2 models; a negative result (MLP no better than linear) is still informative -- it means embeddings are already linearly separable for liveability
- Duplicated data loading code between linear and DNN probes is technical debt but manageable

## Implementation Notes

### Files
- `stage3_analysis/dnn_probe.py` -- main module (MLPProbeModel, DNNProbeConfig, DNNProbeRegressor)
- `stage3_analysis/dnn_probe_viz.py` -- visualization (DNNProbeVisualizer)

### Dependencies
- `torch` (already in project dependencies)
- `spatialkfold` (already used by linear probe)
- `sklearn.preprocessing.StandardScaler` (already used)
- No `torch_geometric` or `optuna` required
