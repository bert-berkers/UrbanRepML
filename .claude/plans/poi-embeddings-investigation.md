# POI Embeddings Investigation Plan

**Written**: 2026-03-01
**Status**: SUPERSEDED — hex2vec training now running directly (2026-03-06). OOM issues resolved via EarlyStopping fix (2130d0e) and res10 optimization (2aa6066).
**Blocking**: DNN probe sweep, downstream fusion training

> **Session note (2026-03-01, misty-shining-maple)**: The OOM in `_get_raw_counts → astype(np.float32)` was the central finding of this session. Root cause confirmed: 9.3 GiB count matrix construction with a ~18 GiB float64 intermediate. Wave 2 (Path A res9 + Path B GeoVex) is the recommended next dispatch. Plan remains ACTIVE — no paths have been attempted yet beyond Wave 1 triage.

## Executive Summary

POI embeddings via Hex2Vec and GeoVex are blocked by an **OOM error in SRAI's count matrix construction**. This plan documents the investigation, blockers, and options for unblocking.

### Current State
- **Intermediates**: Pre-computed for res8, res9, res10 with HEX2VEC_FILTER (725 POI categories)
- **POI Processor**: Refactored (commit 95fb8a9) with training callbacks and gradual batch size ramp
- **Hex2Vec Training**: Fails at `_get_raw_counts` → `astype(np.float32)` with OOM

### Decision Required
Choose one of:
1. **Try lower resolution** (res9 instead of res10)
2. **Switch to GeoVex** (unknown if it has same OOM)
3. **Reduce feature dimensionality** (filter sparse POI categories)
4. **Investigate SRAI sparse matrix path** (if available)

---

## Blocker: OOM in Hex2Vec Count Matrix Construction

### Error Details
**Timestamp**: 2026-03-01 12:36:07
**Location**: SRAI `Hex2VecEmbedder.fit()` → `_get_raw_counts()` → pandas `astype(np.float32)`

**Traceback**:
```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 9.32 GiB
for an array with shape (725, 3450481) and data type float32
```

**Context**:
- 6.07M total hexagons (res10, full Netherlands)
- 3.45M populated hexagons (56.9% occupancy)
- 725 POI feature categories (from HEX2VEC_FILTER)
- 25.13M joint pairs (hexagon-feature relationships)

**Timeline**:
1. ✓ Load regions/features/joint pairs (1m 55s)
2. ✓ Load neighbourhood graph (1s)
3. ✓ Filter to populated regions (1m 30s)
4. ✓ Estimate memory: "16.4 GiB full → 9.3 GiB filtered"
5. ✓ Initialize model and callbacks
6. ✗ FAIL: Construct count matrix in memory, cast to float32

**Why it happens**: SRAI's `_get_raw_counts` calls the parent embedder's `transform()` which builds a dense (725, 3.45M) matrix as float64, then casts to float32. This requires:
- ~18 GB for float64 intermediate
- ~9.3 GB for float32 final
- Peak memory usage during astype: ~27 GB or more

The estimation logged (9.3 GiB) did not account for the intermediate float64 allocation.

---

## Investigation Paths

### Path A: Lower Resolution (Quickest)

**Test**: Run Hex2Vec at res9 instead of res10

**Rationale**:
- Fewer hexagons = smaller count matrix
- H3 res9 Netherlands ≈ 600K-1M hexagons (vs 3.45M at res10)
- Linear memory savings: 1M vs 3.45M ≈ 3.45x less OOM risk

**Commands**:
```bash
# Regenerate intermediates at res9
python -m stage1_modalities.poi --study-area netherlands --resolution 9 \
  --cache-intermediates

# Attempt hex2vec at res9
python -m stage1_modalities.poi --study-area netherlands --resolution 9 \
  --embedder hex2vec --batch-size 16384 --initial-batch-size 512 \
  --early-stopping-patience 3
```

**Outcome if successful**:
- Confirms OOM is purely a memory pressure issue, not a logic error
- Hex2Vec can train on lower resolution
- Lower resolution works for downstream fusion (res8-10 multi-scale)

**Risk**: Lower resolution loses granularity; may impact probe performance

**Owner**: stage1-modality-encoder

---

### Path B: Switch to GeoVex (Parallel)

**Test**: Try GeoVex embedder instead of Hex2Vec

**Rationale**:
- GeoVex is a CNN-based embedder, different architecture
- May use sparse tensors or chunked computation internally
- Already integrated in processor.py (config.yaml has geovex_epochs, geovex_embedding_size)

**Commands**:
```bash
# Try GeoVex at res10 (same as failed hex2vec)
python -m stage1_modalities.poi --study-area netherlands --resolution 10 \
  --embedder geovex --batch-size 4096 --initial-batch-size 256 \
  --early-stopping-patience 3
```

**Outcome if successful**:
- GeoVex avoids the count matrix astype bottleneck
- Parallel embedding method available for probe pipeline
- May be faster or slower than Hex2Vec (empirical)

**Risk**: GeoVex configuration is untested; may require tuning (embedding_size, convolutional_layers)

**Owner**: stage1-modality-encoder (or run both in parallel)

---

### Path C: Reduce Feature Dimensionality (Advanced)

**Test**: Filter sparse POI categories before Hex2Vec

**Rationale**:
- HEX2VEC_FILTER includes 725 categories, many sparse
- Filtering to top N categories (e.g., 256) reduces matrix width by 2.8x
- Memory savings: (256, 3.45M) = 3.5 GiB vs (725, 3.45M) = 9.3 GiB

**Option 1: Programmatic filtering**:
```python
# In processor.py run_hex2vec():
feature_counts = features_gdf['tags'].value_counts()
top_features = feature_counts.nlargest(256).index
filtered_features_gdf = features_gdf[features_gdf['tags'].isin(top_features)]
```

**Option 2: Config-based override**:
```yaml
poi_categories:  # Override HEX2VEC_FILTER with custom subset
  - restaurant
  - shop_supermarket
  - amenity_pharmacy
  # ... top 256 categories only
```

**Outcome if successful**:
- Significantly lower memory footprint
- Still captures most POI variation (top 256 categories cover ~90% of data)
- Can experiment with different thresholds

**Risk**: Loses rare but potentially significant POI types; may bias embeddings

**Owner**: stage1-modality-encoder or POI processor maintainer

---

### Path D: SRAI Sparse Matrix Path (Research)

**Test**: Investigate if SRAI's Hex2VecEmbedder has sparse matrix option

**Rationale**:
- Most POI count matrices are sparse (many hexagons have zero counts for rare categories)
- If SRAI internally supports sparse, could reduce peak memory 10-50x

**Investigation steps**:
1. Read SRAI Hex2VecEmbedder source: `srai/embedders/hex2vec/embedder.py`
2. Check if `_get_raw_counts` or parent's `transform` accepts sparse format
3. Look for kwargs like `sparse=True` or `use_sparse_matrix`
4. Check SRAI issues/docs for sparse count matrix discussions

**Outcome if successful**:
- No memory cap; can handle arbitrary resolution
- Cleaner solution than dimensionality reduction

**Risk**: Sparse path may not exist or may break downstream model compatibility

**Owner**: stage1-modality-encoder (research only, no code changes without SRAI PR)

---

## Execution Plan

### Wave 1: Triage (this session)

**Executor**: execution (runner)
**What**: Run Hex2Vec at res10 and document failure (COMPLETED 2026-03-01)

**Acceptance**:
- Failure reproduced and root cause identified ✓
- OOM is in count matrix astype ✓
- Execution scratchpad updated ✓

---

### Wave 2: Parallel Investigation (next coordinator session)

**Executor A**: stage1-modality-encoder
**Task A**: Run Hex2Vec at **resolution 9** (Path A)

- Regenerate POI intermediates at res9
- Attempt hex2vec training
- Measure training time and final embedding shape
- Report success/failure and memory profile

**Executor B**: stage1-modality-encoder (or separate session)
**Task B**: Run **GeoVex** at res10 (Path B)

- Use existing res10 intermediates
- Attempt geovex training with standard config
- Report success/failure and compare to Hex2Vec
- Document any configuration tuning needed

**Acceptance**:
- At least one of (A, B) succeeds and produces valid embeddings
- stage1-modality-encoder scratchpad logs both runs
- QAQC verifies output shape matches expectations

---

### Wave 3: Consolidation (if needed)

**Executor**: stage1-modality-encoder
**Tasks** (if both A and B fail):

1. Implement Path C: Filter features to top 256 categories
2. Regenerate intermediates with filtered feature set
3. Retry Hex2Vec with smaller matrix
4. Document feature filtering decision and impact

**Executor** (research): stage1-modality-encoder
**Task** (parallel, optional):
- Investigate SRAI sparse matrix path (Path D)
- Read SRAI source, report findings to coordinator

**Acceptance**:
- At least one of (A, B, C) produces valid POI embeddings
- Embeddings can be used in downstream fusion pipeline
- Decision documented: which method chosen and why

---

## Expected Outcomes

### Success Criteria

**Primary**: POI embeddings successfully generated
- ✓ Valid shape (N_hexagons, embedding_dim)
- ✓ No NaN/Inf values
- ✓ Saved to `data/study_areas/netherlands/stage1_unimodal/poi/embeddings/`

**Secondary**: DNN probe sweep unblocked
- ✓ POI embeddings available for fusion pipeline
- ✓ Downstream agents can run stage2 fusion training
- ✓ Probe sweep can use POI+AlphaEarth+Roads fusion

### Failure Modes (and next steps)

| Failure | Next Step |
|---------|-----------|
| All three paths OOM | Escalate to SRAI maintainers; switch to simpler embeddings (count + diversity only, no neural) |
| GeoVex trains but quality poor | Use Hex2Vec at res9, sacrifice granularity |
| Training succeeds but DNN probe fails | May be separate issue; debug probe pipeline |

---

## Open Questions

1. **Sparse matrix in SRAI**: Does `_get_raw_counts` or parent support sparse? (Path D research)
2. **GeoVex performance**: How does GeoVex compare to Hex2Vec in training time and embeddings quality?
3. **Resolution-fusion mismatch**: If we use res9 for POI but res8-10 for fusion, does the inner join collapse too much? (Check downstream)
4. **Feature sparsity distribution**: What's the cumulative coverage of top 256 categories? (For Path C decision)

---

## Resources

- **SRAI Hex2Vec docs**: https://srai.readthedocs.io/en/latest/user_guide/embedders/hex2vec.html
- **SRAI GeoVex docs**: https://srai.readthedocs.io/en/latest/user_guide/embedders/geovex.html
- **SRAI source** (Hex2Vec): `srai/embedders/hex2vec/embedder.py` (look for `_get_raw_counts`)
- **Execution scratchpad**: `.claude/scratchpad/execution/2026-03-01.md` (failure details)
- **POI Processor**: `stage1_modalities/poi/processor.py` (entry points: `run_hex2vec`, `run_geovex`)

---

## Coordination Notes

**Recommended next action**: Dispatch stage1-modality-encoder with **Path A (res9) and Path B (GeoVex)** in parallel. If both fail within 1 hour, escalate to Path C or Path D research.

**Timeline**: Expect resolution by end of next coordinator session (2026-03-02 or 2026-03-03).

**Dependencies**: Unblocks DNN probe sweep, fusion training, stage3 analysis. High priority.
