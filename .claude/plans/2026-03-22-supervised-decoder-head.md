# Supervised Decoder Head for FullAreaUNet

**Goal**: Add leefbaarometer supervised prediction as a bonus loss signal to FullAreaUNet,
using Kendall uncertainty weighting to balance self-supervised and supervised objectives.
Beat ring_agg mean R^2=0.534 on leefbaarometer probes.

**Strategic context**: Wave 6 showed the accessibility UNet architecture works
(vrz=0.795 beats ring_agg 0.786) but the self-supervised reconstruction objective
discards liveability signal (mean R^2=0.476 vs ring_agg 0.534). The remaining gap
is an objective function problem, not an architecture problem. Levering et al. 2023
showed that adding supervised domain prediction heads with a semantic bottleneck forces
the model to preserve liveability-relevant features. Kendall et al. 2018 uncertainty
weighting replaces fragile fixed loss weights with learnable scalars.

---

## Architecture Overview

```
FullAreaUNet forward() produces:
  embeddings[res9]  = [N_res9, dim_fine]    <-- supervised heads attach HERE
  embeddings[res8]  = [N_res8, dim_fine]
  embeddings[res7]  = [N_res7, dim_fine]

New supervised pathway (res9 only):
  embeddings[res9]          [N_res9, dim_fine]
       |
  SupervisedHead.domain     Linear(dim_fine, 5) + Sigmoid
       |
  domain_preds              [N_res9, 5]  (fys, onv, soc, vrz, won)
       |
  SupervisedHead.lbm        Linear(5, 1)
       |
  lbm_pred                  [N_res9, 1]

Loss masking: Only ~130K of ~399K res9 hexagons have leefbaarometer targets.
Supervised loss is computed on the masked subset only.
```

---

## Data Shapes Reference

| Tensor | Shape | Notes |
|--------|-------|-------|
| `embeddings[res9]` | `[398931, 74]` | With dims=[74,37,18], dim_fine=74 |
| `targets_domain` | `[N_valid, 5]` | fys, onv, soc, vrz, won (order from TARGET_COLS[1:]) |
| `targets_lbm` | `[N_valid, 1]` | Overall liveability score |
| `target_mask` | `[398931]` | Boolean, True for ~130K hexes with LBM data |
| `domain_preds[target_mask]` | `[N_valid, 5]` | Predicted domain scores |
| `lbm_pred[target_mask]` | `[N_valid, 1]` | Predicted overall liveability |

Where N_valid = number of res9 hexes with leefbaarometer targets (~130,792).

Target value ranges (from res9 2022 parquet):
- lbm: mean=4.20, std=0.12, range=[3.42, 5.04]
- fys: mean=-0.003, std=0.067, range=[-0.74, 0.46]
- onv: mean=0.078, std=0.065, range=[-0.47, 0.16]
- soc: mean=0.075, std=0.059, range=[-0.42, 0.44]
- vrz: mean=-0.115, std=0.102, range=[-0.33, 0.69]
- won: mean=0.062, std=0.052, range=[-0.53, 0.89]

Note: lbm is on a different scale (~4.2) than domains (~0.0). The Kendall
weighting handles this automatically (each loss gets its own log_sigma), but
z-scoring the targets before training is also an option.

---

## Wave 1: SupervisedHead module + LossComputer rewrite

### 1a. New class: `SupervisedHead` in `full_area_unet.py`

```python
class SupervisedHead(nn.Module):
    """Levering-style semantic bottleneck: embedding -> 5 domains -> 1 lbm.

    The lbm prediction MUST pass through the 5 domain scores (semantic bottleneck).
    This forces the embedding to encode interpretable liveability dimensions.
    """
    def __init__(self, input_dim: int, n_domains: int = 5):
        super().__init__()
        # Domain prediction: embedding -> 5 domain scores
        self.domain_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, n_domains),
        )
        # LBM prediction: 5 domain scores -> 1 overall score (semantic bottleneck)
        self.lbm_head = nn.Linear(n_domains, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Embedding tensor [N, input_dim]
        Returns:
            domain_preds: [N, 5] predicted domain scores
            lbm_pred: [N, 1] predicted overall liveability
        """
        domain_preds = self.domain_head(x)        # [N, 5]
        lbm_pred = self.lbm_head(domain_preds)    # [N, 1]
        return domain_preds, lbm_pred
```

Design decisions:
- Hidden layer (input_dim // 2) before domain output: prevents the domain head from
  being a rank-5 bottleneck on the embedding gradient. With dim_fine=74, the hidden
  layer is 37D -- matches the mid-resolution dim in the [74,37,18] pyramid.
- No sigmoid/activation on outputs: targets are continuous (not bounded [0,1]).
  The domain scores are z-scored-ish (centered near 0, range roughly [-0.7, 0.9]).
  lbm is on a different scale (~4.2) but linear output handles this fine with MSE.
- Semantic bottleneck: lbm_head takes 5D domain_preds as input, NOT the raw embedding.
  This is the key design from Levering -- forces interpretability.

### 1b. Modify `FullAreaUNet.__init__()` to optionally create SupervisedHead

Add `n_targets: int = 0` parameter. When n_targets > 0, create the supervised head.

```python
class FullAreaUNet(nn.Module):
    def __init__(
            self,
            feature_dims: Dict[str, int],
            dims: List[int] = None,
            num_convs: int = 10,
            device: str = "cuda",
            resolutions: Optional[list] = None,
            n_targets: int = 0,          # NEW: 0=self-supervised only, 5=leefbaarometer
            # Legacy args
            hidden_dim: int = None,
            output_dim: int = None,
    ):
        # ... existing init code ...

        # Supervised head (optional)
        self.supervised = n_targets > 0
        if self.supervised:
            self.supervised_head = SupervisedHead(
                input_dim=dim_fine,
                n_domains=n_targets,
            )
```

### 1c. Modify `FullAreaUNet.forward()` return signature

**Backward compatibility contract**: When `self.supervised` is False, forward()
returns exactly as before: `(embeddings, reconstructed)`. When True, it returns
`(embeddings, reconstructed, predictions)` where predictions is a dict.

```python
def forward(
        self,
        features_dict: Dict[str, torch.Tensor],
        edge_indices: Dict[int, torch.Tensor],
        edge_weights: Dict[int, torch.Tensor],
        mappings: Dict[Tuple[int, int], torch.Tensor]
) -> Union[
    Tuple[Dict[int, torch.Tensor], Dict[str, torch.Tensor]],
    Tuple[Dict[int, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
]:
    # ... existing encoder/decoder/output/reconstruction code unchanged ...

    if self.supervised:
        domain_preds, lbm_pred = self.supervised_head(embeddings[rf])
        predictions = {
            'domain_preds': domain_preds,   # [N_res9, 5]
            'lbm_pred': lbm_pred,           # [N_res9, 1]
        }
        return embeddings, reconstructed, predictions

    return embeddings, reconstructed
```

### 1d. Rewrite `LossComputer` with Kendall uncertainty weighting

The current LossComputer takes fixed `loss_weights: Dict[str, float]`. Replace
with learnable Kendall log-sigma parameters.

**Kendall uncertainty weighting formula** (Kendall, Gal, Cipolla 2018):

For K loss components L_1, ..., L_K with learnable parameters s_1, ..., s_K
(where s_i = log(sigma_i)):

```
total_loss = sum_i [ L_i * exp(-2 * s_i) + s_i ]
```

- When L_i is large: s_i grows -> exp(-2*s_i) shrinks -> L_i contributes less
- When L_i is small: s_i shrinks -> exp(-2*s_i) grows -> L_i contributes more
- The +s_i term prevents collapse (sigma -> infinity would kill all losses)
- Self-balancing: no manual weight tuning needed

```python
class LossComputer(nn.Module):
    """Loss computation with optional Kendall uncertainty weighting.

    When n_supervised_components > 0, uses learnable log_sigma per loss component
    (Kendall et al. 2018). Otherwise, uses fixed weights (backward compatible).
    """
    def __init__(
        self,
        n_supervised_components: int = 0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.use_kendall = n_supervised_components > 0

        if self.use_kendall:
            # 2 base components (reconstruction, consistency) + supervised components
            # When supervised: recon, consist, domain, lbm = 4 total
            n_total = 2 + n_supervised_components  # n_supervised_components = 2 (domain + lbm)
            # Initialize log_sigma to 0 => sigma=1 => initial weight = exp(0) = 1
            self.log_sigmas = nn.ParameterDict({
                'recon': nn.Parameter(torch.tensor(0.0)),
                'consist': nn.Parameter(torch.tensor(0.0)),
            })
            if n_supervised_components >= 1:
                self.log_sigmas['domain'] = nn.Parameter(torch.tensor(0.0))
            if n_supervised_components >= 2:
                self.log_sigmas['lbm'] = nn.Parameter(torch.tensor(0.0))

    def _kendall_weight(self, loss: torch.Tensor, log_sigma: nn.Parameter) -> torch.Tensor:
        """Apply Kendall uncertainty weighting to a single loss component.

        Returns: loss * exp(-2 * log_sigma) + log_sigma
        """
        return loss * torch.exp(-2 * log_sigma) + log_sigma

    def compute_losses(
            self,
            embeddings: Dict[int, torch.Tensor],
            reconstructed: Dict[str, torch.Tensor],
            features_dict: Dict[str, torch.Tensor],
            mappings: Dict[Tuple[int, int], torch.Tensor],
            loss_weights: Dict[str, float],
            predictions: Optional[Dict[str, torch.Tensor]] = None,
            targets: Optional[Dict[str, torch.Tensor]] = None,
            target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses with optional supervised components.

        New parameters (all optional, backward compatible):
            predictions : dict with 'domain_preds' [N, 5] and 'lbm_pred' [N, 1]
            targets     : dict with 'domain' [N_valid, 5] and 'lbm' [N_valid, 1]
            target_mask : boolean tensor [N_res9], True where targets exist

        Returns dict of named losses including 'total_loss'.
        """
        # 1. Reconstruction loss (unchanged)
        recon_losses = {}
        for name, pred in reconstructed.items():
            target_feat = features_dict[name]
            recon_losses[name] = F.mse_loss(
                F.normalize(pred, p=2, dim=1, eps=1e-8),
                F.normalize(target_feat, p=2, dim=1, eps=1e-8)
            )
        raw_recon = sum(recon_losses.values())

        # 2. Consistency loss (unchanged)
        consistency_losses = {}
        for (res_fine, res_coarse), mapping in mappings.items():
            fine_mapped = torch.sparse.mm(mapping.t(), embeddings[res_fine])
            fine_mapped = F.normalize(fine_mapped, p=2, dim=1, eps=1e-8)
            coarse_emb = F.normalize(embeddings[res_coarse], p=2, dim=1, eps=1e-8)
            consistency_losses[(res_fine, res_coarse)] = F.mse_loss(fine_mapped, coarse_emb)
        raw_consist = sum(consistency_losses.values()) / len(consistency_losses)

        # 3. Supervised losses (only when predictions + targets provided)
        raw_domain = None
        raw_lbm = None
        has_supervised = (
            predictions is not None
            and targets is not None
            and target_mask is not None
        )

        if has_supervised:
            masked_domain_preds = predictions['domain_preds'][target_mask]  # [N_valid, 5]
            masked_lbm_pred = predictions['lbm_pred'][target_mask]          # [N_valid, 1]
            raw_domain = F.mse_loss(masked_domain_preds, targets['domain'])
            raw_lbm = F.mse_loss(masked_lbm_pred, targets['lbm'])

        # 4. Combine losses
        if self.use_kendall:
            total = self._kendall_weight(raw_recon, self.log_sigmas['recon'])
            total = total + self._kendall_weight(raw_consist, self.log_sigmas['consist'])

            if has_supervised and raw_domain is not None:
                total = total + self._kendall_weight(raw_domain, self.log_sigmas['domain'])
                total = total + self._kendall_weight(raw_lbm, self.log_sigmas['lbm'])
        else:
            # Fixed weights (backward compatible path)
            total = raw_recon * loss_weights['reconstruction']
            total = total + raw_consist * loss_weights['consistency']

        result = {
            'total_loss': total,
            'reconstruction_loss': raw_recon,
            'consistency_loss': raw_consist,
            **{f'recon_loss_{name}': loss for name, loss in recon_losses.items()},
            **{f'consistency_loss_{k[0]}_{k[1]}': v for k, v in consistency_losses.items()},
        }

        if has_supervised:
            result['domain_loss'] = raw_domain
            result['lbm_loss'] = raw_lbm

        if self.use_kendall:
            for name, log_sigma in self.log_sigmas.items():
                result[f'log_sigma_{name}'] = log_sigma
                result[f'effective_weight_{name}'] = torch.exp(-2 * log_sigma)

        return result
```

**Critical change**: `LossComputer` is now `nn.Module` (not a plain class) because it
owns `log_sigmas` as `nn.ParameterDict`. This means:
- Its parameters must be included in the optimizer
- The trainer must pass `list(model.parameters()) + list(loss_computer.parameters())`
  to the optimizer

### Gate 1: Code review

**Go/no-go criteria**:
- SupervisedHead produces correct shapes: [N, 5] and [N, 1]
- LossComputer backward-compatible: when n_supervised_components=0, behaves exactly as before
- Forward() return type changes only when supervised=True
- log_sigma parameters are in the optimizer's param groups

---

## Wave 2: Trainer modifications

### 2a. Modify `FullAreaModelTrainer.__init__()`

```python
class FullAreaModelTrainer:
    def __init__(
            self,
            model_config: dict,
            loss_weights: Dict[str, float] = None,
            city_name: str = "south_holland_threshold80",
            wandb_project: str = "urbanrepml",
            checkpoint_dir: Optional[Path] = None,
            year: str = "2022",
            supervised: bool = False,          # NEW
            n_targets: int = 5,                # NEW: number of domain targets
    ):
        # ... existing setup ...

        # Add n_targets to model_config when supervised
        if supervised:
            model_config = {**model_config, 'n_targets': n_targets}

        self.model = FullAreaUNet(**model_config, device=self.device)
        self.model.to(self.device)

        # LossComputer: nn.Module when supervised (owns log_sigma params)
        if supervised:
            self.loss_computer = LossComputer(n_supervised_components=2)
            self.loss_computer.to(self.device)
        else:
            self.loss_computer = LossComputer()

        self.supervised = supervised
        # ... rest unchanged ...
```

### 2b. Modify `FullAreaModelTrainer.train()`

New parameter: `target_data: Optional[Dict[str, torch.Tensor]] = None`

```python
def train(self, features_dict, edge_indices, edge_weights, mappings,
          num_epochs=100, learning_rate=1e-4, warmup_epochs=10,
          patience=100, gradient_clip=1.0, wandb_config=None,
          target_data=None,                                       # NEW
          ) -> dict:
```

`target_data` is a dict with:
- `'domain'`: Tensor [N_valid, 5] -- the 5 domain scores for valid hexagons
- `'lbm'`: Tensor [N_valid, 1] -- the overall lbm score for valid hexagons
- `'mask'`: Tensor [N_res9] bool -- True for hexagons with valid targets

Changes inside the training loop:

```python
# Optimizer includes loss_computer params when supervised
if self.supervised:
    all_params = list(self.model.parameters()) + list(self.loss_computer.parameters())
else:
    all_params = list(self.model.parameters())

optimizer = torch.optim.AdamW(
    all_params,
    lr=learning_rate,
    weight_decay=0.01,
    eps=1e-8
)

# ... in training loop ...

# Forward pass
if self.supervised:
    embeddings, reconstructed, predictions = self.model(
        features_dict, edge_indices, edge_weights, mappings
    )
else:
    embeddings, reconstructed = self.model(
        features_dict, edge_indices, edge_weights, mappings
    )
    predictions = None

# Loss computation
losses = self.loss_computer.compute_losses(
    embeddings=embeddings,
    reconstructed=reconstructed,
    features_dict=features_dict,
    mappings=mappings,
    loss_weights=self.loss_weights,
    predictions=predictions,
    targets=target_data if self.supervised else None,
    target_mask=target_data['mask'] if self.supervised and target_data else None,
)

# wandb logging: add supervised metrics when available
if wb_run is not None:
    log_dict = {
        "epoch": epoch,
        "total_loss": total_loss.item(),
        "reconstruction_loss": losses['reconstruction_loss'].item(),
        "consistency_loss": losses['consistency_loss'].item(),
        "lr": current_lr,
        "grad_norm": grad_norm,
        "patience_counter": patience_counter,
    }
    if self.supervised and 'domain_loss' in losses:
        log_dict["domain_loss"] = losses['domain_loss'].item()
        log_dict["lbm_loss"] = losses['lbm_loss'].item()
        for name in self.loss_computer.log_sigmas:
            log_dict[f"log_sigma_{name}"] = losses[f'log_sigma_{name}'].item()
            log_dict[f"effective_weight_{name}"] = losses[f'effective_weight_{name}'].item()
    wandb.log(log_dict, step=epoch)
```

### Gate 2: Trainer smoke test

**Go/no-go criteria**:
- Without `--supervised`, training behaves identically to before
- With `--supervised`, forward pass produces predictions dict, loss includes domain + lbm
- log_sigma values are being logged to wandb
- Optimizer param count increases by 4 (the log_sigma scalars) + SupervisedHead params

---

## Wave 3: Training script modifications

### 3a. Modify `scripts/stage2/train_full_area_unet.py`

Add `--supervised` flag and target loading logic.

```python
def parse_args():
    # ... existing args ...
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
```

Target loading in `main()` -- after data loading, before trainer creation:

```python
# ---- 2b. Load supervised targets (optional) ----
target_data = None
if args.supervised:
    target_path = paths.target_file("leefbaarometer", finest_res, args.target_year)
    logger.info(f"Loading leefbaarometer targets from {target_path}")

    target_df = pd.read_parquet(target_path)

    # Domain columns: fys, onv, soc, vrz, won (NOT lbm -- that's derived)
    domain_cols = ["fys", "onv", "soc", "vrz", "won"]
    lbm_col = "lbm"

    # Align targets to hex_ids[finest_res] via inner join preserving order
    hex_ids_finest = hex_ids[finest_res]
    hex_series = pd.Series(range(len(hex_ids_finest)), index=hex_ids_finest, name="idx")

    # Build mask: True for hexes that have leefbaarometer data
    target_mask = torch.zeros(len(hex_ids_finest), dtype=torch.bool)
    matched = hex_series.index.intersection(target_df.index)
    matched_indices = hex_series.loc[matched].values
    target_mask[matched_indices] = True

    # Extract aligned target values (ordered by hex_ids position)
    target_aligned = target_df.loc[matched]  # ordered by target_df index
    # Re-order to match hex_ids ordering
    target_ordered = target_df.reindex(hex_series.index).dropna()
    # But we only need the values for masked hexes, in the order they appear in hex_ids
    target_values = target_df.reindex([h for h in hex_ids_finest if h in target_df.index])

    domain_tensor = torch.tensor(
        target_values[domain_cols].values, dtype=torch.float32
    ).to(device)
    lbm_tensor = torch.tensor(
        target_values[[lbm_col]].values, dtype=torch.float32
    ).to(device)

    target_data = {
        'domain': domain_tensor,        # [N_valid, 5]
        'lbm': lbm_tensor,              # [N_valid, 1]
        'mask': target_mask.to(device),  # [N_res9] bool
    }

    n_valid = target_mask.sum().item()
    logger.info(
        f"Supervised targets loaded: {n_valid:,} / {len(hex_ids_finest):,} "
        f"hexagons have leefbaarometer data ({100*n_valid/len(hex_ids_finest):.1f}%)"
    )
    print(f"  Supervised:  {n_valid:,} / {len(hex_ids_finest):,} hexagons "
          f"({100*n_valid/len(hex_ids_finest):.1f}%)")
```

Trainer creation with supervised flag:

```python
trainer = FullAreaModelTrainer(
    model_config=model_config,
    city_name=args.study_area,
    checkpoint_dir=checkpoint_dir,
    year=args.year,
    supervised=args.supervised,    # NEW
)
```

Pass target_data to train():

```python
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
    target_data=target_data,          # NEW
)
```

### Gate 3: End-to-end smoke test

Run with small epoch count to verify the full pipeline:
```bash
python scripts/stage2/train_full_area_unet.py \
    --study-area netherlands --epochs 5 --supervised \
    --dims 74,37,18 --resolutions 9,8,7 --year 2022 \
    --accessibility-graph data/study_areas/netherlands/accessibility/walk_res9.parquet
```

**Go/no-go**: Completes 5 epochs without errors. Loss dict contains domain_loss and lbm_loss.

---

## Wave 4: Train, probe, compare

### 4a. Full training run

```bash
python scripts/stage2/train_full_area_unet.py \
    --study-area netherlands --epochs 1500 --supervised \
    --dims 74,37,18 --resolutions 9,8,7 --year 2022 \
    --lr 1e-2 --patience 200 --warmup-epochs 50 \
    --accessibility-graph data/study_areas/netherlands/accessibility/walk_res9.parquet \
    --wandb --wandb-name "unet-74-37-18-supervised-kendall"
```

Training budget: 1500 epochs (was 1000 for self-supervised). Patience 200 (was 100).
The supervised loss adds a direct gradient signal that should accelerate convergence,
but the combined loss landscape may be more complex.

### 4b. Extract embeddings + probe

Same pipeline as Wave 6:
1. Extract highway exits at all resolutions
2. Run leefbaarometer probes on res9 embeddings
3. Run leefbaarometer probes on multiscale concatenated embeddings
4. Write to probe_results using `ProbeResultsWriter`

### 4c. Comparison table

Expected approaches to compare:
| Approach | Dim | Expected R^2 |
|----------|-----|-------------|
| raw_concat | 74D | 0.517 |
| ring_agg_k10 | 74D | 0.534 |
| unet_uniform | 74D | 0.324 |
| unet_accessibility | 74D | 0.441 |
| unet_accessibility_multiscale | 222D | 0.476 |
| **unet_supervised** | 74D | target: >0.534 |
| **unet_supervised_multiscale** | 222D | target: >0.534 |

### Gate 4: Did it beat ring_agg?

Success criteria:
- **Primary**: Mean R^2 across 6 targets > 0.534 (ring_agg baseline)
- **Secondary**: Individual target R^2 improvements, especially on targets where UNet already showed strength (vrz)
- **Diagnostic**: Check that Kendall weights converged to sensible values (reconstruction weight should be < supervised weight by end of training -- the model should "care more" about predicting liveability than reconstruction)

---

## Wave 5: Verification and documentation

- QAQC pass on code changes
- Verify checkpoint contains SupervisedHead weights
- Verify backward compatibility (training without --supervised still works)
- Update report with supervised results
- Librarian update

---

## Backward Compatibility Contract

| Component | Without --supervised | With --supervised |
|-----------|---------------------|-------------------|
| `FullAreaUNet.__init__()` | n_targets=0, no SupervisedHead created | n_targets=5, SupervisedHead created |
| `FullAreaUNet.forward()` | Returns `(embeddings, reconstructed)` | Returns `(embeddings, reconstructed, predictions)` |
| `LossComputer.__init__()` | n_supervised_components=0, no log_sigmas | n_supervised_components=2, log_sigmas created |
| `LossComputer.compute_losses()` | Fixed weights, ignores predictions/targets/mask args | Kendall weighting, uses predictions/targets/mask |
| `FullAreaModelTrainer.__init__()` | supervised=False, plain LossComputer | supervised=True, LossComputer as nn.Module |
| `FullAreaModelTrainer.train()` | target_data=None, no supervised loss | target_data provided, supervised loss computed |
| Training script | `--supervised` absent, no target loading | `--supervised` present, targets loaded and aligned |
| Checkpoint | state_dict has no supervised_head keys | state_dict has supervised_head.* keys |
| Existing checkpoints | Load fine (no new keys) | Load with `strict=False` if loading old checkpoint into supervised model |

**Key invariant**: When `--supervised` is NOT passed, the entire codebase behaves
identically to the current version. Zero behavioral changes to the non-supervised path.

---

## Files Modified

1. `stage2_fusion/models/full_area_unet.py`:
   - New class: `SupervisedHead`
   - Modified: `FullAreaUNet.__init__()` (add n_targets param)
   - Modified: `FullAreaUNet.forward()` (return predictions when supervised)
   - Rewritten: `LossComputer` (now nn.Module, Kendall weighting)
   - Modified: `FullAreaModelTrainer.__init__()` (add supervised, n_targets params)
   - Modified: `FullAreaModelTrainer.train()` (add target_data param, supervised loss path)

2. `scripts/stage2/train_full_area_unet.py`:
   - New args: `--supervised`, `--target-year`
   - New section: target loading, alignment, mask construction
   - Modified: trainer creation and train() call

No other files are modified. No new files are created.

---

## Risk Mitigation

1. **Target alignment correctness**: The mask construction is the most error-prone part.
   The ordering must be: `target_mask[i] == True` means `hex_ids[finest_res][i]` has a
   leefbaarometer target, AND the j-th True in target_mask corresponds to `targets['domain'][j]`.
   Add an assertion: `assert target_mask.sum() == len(target_data['domain'])`.

2. **Kendall instability early in training**: log_sigma initialized to 0 means initial
   effective_weight = exp(0) = 1 for all components. During warmup (50 epochs), all losses
   are large and noisy. The log_sigma values will adjust. If instability occurs, consider
   clamping log_sigma to [-4, 4] range (effective weight range [exp(-8), exp(8)] = [0.0003, 2981]).

3. **Gradient scale mismatch**: The 5 domain targets are small (std ~0.06-0.10) while lbm
   is larger (mean ~4.2, std ~0.12). MSE on lbm will be ~1000x larger than MSE on domains
   in raw scale. Kendall weighting will handle this, but monitor the log_sigma values -- if
   log_sigma_lbm is much larger than log_sigma_domain, consider z-scoring targets.

4. **Memory**: SupervisedHead adds ~dim_fine * (dim_fine/2) + (dim_fine/2) * 5 + 5 * 1
   = 74*37 + 37*5 + 5 = 2923 parameters. Negligible relative to the model (~3M params).
   No memory concern.

---

## Execution Status

- **Waves 1-3**: DONE (OODA 1, commit 06b085d). SupervisedHead + LossComputer Kendall rewrite + training script --supervised flag. QAQC 14/14 PASS.
- **Wave 4**: OODA 2 (train + probe + compare)
- **Wave 5**: OODA 2 or 3 (verification + documentation)

### OODA 2: Train + Compare

```
/clear
```
Then:
```
/niche OODA 2 of 3: Train supervised UNet + probe + compare. Run: python scripts/stage2/train_full_area_unet.py --study-area netherlands --epochs 1500 --supervised --dims 74,37,18 --resolutions 9,8,7 --year 2022 --lr 1e-2 --patience 200 --warmup-epochs 50 --accessibility-graph data/study_areas/netherlands/accessibility/walk_res9.parquet --wandb --wandb-name "unet-74-37-18-supervised-kendall". Then extract embeddings, run leefbaarometer probes (res9 + multiscale), write via ProbeResultsWriter, generate comparison chart across all approaches (concat 0.517, ring_agg 0.534, uniform 0.324, accessibility 0.441, accessibility_multiscale 0.476, supervised, supervised_multiscale). Target: beat ring_agg mean R^2=0.534.
```

### OODA 3: Weekend Close-Off

```
/clear
```
Then:
```
/niche OODA 3 of 3: Weekend close-off. Mark Wave 6 DONE in accessibility plan. Plan GTFS as public transport modality. Housekeeping: review scripts/one_off for stale items, ensure data/results organized. Write forward-look for next Friday.
```
