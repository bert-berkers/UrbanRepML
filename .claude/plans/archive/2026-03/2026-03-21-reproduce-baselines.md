# Plan: Reproduce Baselines in Standardized Probe Format

**Session**: coral-falling-snow (2026-03-21)
**Compound state**: reproduction-run
**Dependency**: jade-falling-wind shipping `probe_results_writer.py` — wait for "ready" message

## Data decision

**2-modality, temporally aligned, 2022 only:**
- AlphaEarth 64D (GEE 2022)
- Roads PCA-10 (OSM PBF 2022, 99.6% variance retained)
- Total: **74D** concat, z-scored per modality block
- Dropped: hex2vec/POI (2026 Overpass), GTFS (2026 OVapi, 97% background)

Rationale: clean temporal alignment over mixed-year mess. Prototyping is done; baselines should be honest.

## Baselines to reproduce

| Slug | Source | Notes |
|------|--------|-------|
| `concat_74d_2022` | AE 64 + Roads PCA-10, z-scored | Must regenerate — old 20mix concat won't work |
| `ring_agg_k10_2022` | Ring agg k=10 exponential on 74D | Must regenerate from new concat |
| `unet_2022` | UNet trained on 74D, 1337ep patience=100 | New training run |

---

## Wave 1 — Prep (current)

Goal: understand all moving parts before touching anything.

- [ ] **1a.** Read `concat.py` — does it support per-modality PCA? Or do we PCA roads separately?
- [ ] **1b.** Read `train_full_area_unet.py` — verify `--feature-source` override, LR schedule, checkpoint naming
- [ ] **1c.** Read `run_probe_comparison.py` — can we pass custom embedding paths? Or need a new config?
- [ ] **1d.** Read `run_simple_ring_aggregation.py` — does it take a custom input or always reads from concat output?
- [ ] **1e.** Reply to jade: linear AND DNN, or just DNN?

### GATE 1 — Human decision
> Do we need to modify `concat.py` to add per-modality PCA, or is a separate PCA step cleaner?
> Does the training script accept `--feature-source` for the new 74D concat path?
> **Sync window**: present findings, get go/no-go before writing any code.

---

## Wave 2 — Build embeddings

Blocked on: Gate 1 approval + any code changes from prep findings.

**Step 2a** — Generate 74D concat (sequential):
```bash
python -m stage2_fusion.concat --modalities alphaearth,roads --year 2022 [+ PCA flags TBD]
```
Verify: output is 74D, z-scored, ~247K rows (inner join of AE ∩ Roads).

**Step 2b** — Generate ring_agg (depends on 2a):
```bash
python scripts/stage2/run_simple_ring_aggregation.py --study-area netherlands --year 2022 --K 10 --weighting exponential
```
Verify: same row count, 74D, spatially smoothed.

**Step 2c** — Start UNet training (depends on 2a, runs in background):
```bash
python scripts/stage2/train_full_area_unet.py --study-area netherlands --epochs 1337 --year 2022 --patience 100 --feature-source <concat_74d_path>
```
Expected runtime: ~1-2 hours. Runs in background.

### GATE 2 — Embedding sanity + jade sync

**Embedding checklist** (must all pass):
- [ ] Concat parquet: exactly 74 columns (64 AE + 10 Roads PCA), z-scored, no NaNs
- [ ] Row count: ~247K (AE ∩ Roads inner join)
- [ ] Ring agg parquet: same shape as concat, values spatially smoother
- [ ] UNet training: started, first 5-10 epochs logged, loss decreasing, no CUDA errors

**Jade readiness check** (blocks Wave 3):
- [ ] Check lateral messages for jade's "ready" signal
- [ ] If not ready: UNet keeps training, we wait — no probe runs without the writer
- [ ] If ready: verify `probe_results_writer.py` exists and importable, confirm slug naming convention still `concat_74d_2022` / `ring_agg_k10_2022` / `unet_2022`

**Sync window**: Present embedding shapes + UNet training status to human. Confirm jade status. Go/no-go for probes.

---

## Wave 3 — Probes

Blocked on: Gate 2 jade readiness check passed.
UNet probe additionally blocked on: UNet training completion.

**Step 3a** — Probe concat_74d_2022 (can start immediately after Gate 2 + jade ready):
```bash
python scripts/stage3/run_probe_comparison.py --embeddings "concat_74d_2022:<concat_path>"
```

**Step 3b** — Probe ring_agg_k10_2022 (parallel with 3a):
```bash
python scripts/stage3/run_probe_comparison.py --embeddings "ring_agg_k10_2022:<ring_agg_path>"
```

**Step 3c** — Probe unet_2022 (after UNet training completes):
```bash
python scripts/stage3/run_probe_comparison.py --embeddings "unet_2022:<unet_embedding_path>"
```

Each probe: pipe through `ProbeResultsWriter.write_from_regressor()` → `probe_results/{slug}/predictions.parquet` + `metrics.parquet`.

### GATE 3 — Results review
> All 3 probes complete? R² values for each?
> **Sync window**: present comparison table. Does ring_agg still beat UNet? Did 1337 epochs help?
> Human decides: are these the baselines we keep, or do we adjust?

---

## Wave 4 — Close-out

- [ ] Coordinator scratchpad (coral-falling-snow)
- [ ] Librarian update (new 2022 concat pipeline, 74D)
- [ ] Ego check

---

## Decision log

| # | Question | Answer | Decided |
|---|----------|--------|---------|
| 1 | Which modalities? | AE + Roads only, both 2022 | coral + human |
| 2 | Roads dimensionality? | PCA-10 (99.6% variance) | human |
| 3 | Epochs? | 1337, patience=100 | human |
| 4 | Wait for jade's writer? | Yes — hold probes until "ready" | jade msg |
| 5 | PCA in concat.py or separate? | TBD (Wave 1 finding) | — |
| 6 | Linear AND DNN probes? | TBD (reply to jade) | — |