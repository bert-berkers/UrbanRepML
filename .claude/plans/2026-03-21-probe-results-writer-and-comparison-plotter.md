# Plan: Probe Results Writer + Comparison Plotter

## Context

Probe results (linear, DNN) currently write per-regressor CSVs/parquets via `save_results()`.
No standardized cross-approach format exists, so comparing ring_agg vs concat vs UNet
requires ad-hoc loading. Terminal C (coral-falling-snow) is holding baseline runs until
this writer ships. Terminal A (amber-rising-tide) is building AccessibilityUNet — its
probe results will also flow through this writer.

**Compound state**: `pipeline-tooling` (code_quality 4, data_eng 4, model_arch 2, no exploration)

---

## Schema Contract

**Two parquets per approach** (different granularity, same directory):

### `predictions.parquet` — per-hex, per-target
| Column | Type | Example | Source field on TargetResult |
|--------|------|---------|---------------------------|
| approach | str | "ring_agg_k10" | constructor arg |
| target_variable | str | "lbm" | `.target` |
| region_id | str | "8944a1a06c3ffff" | `.region_ids[i]` |
| y_true | float64 | 5.2 | `.actual_values[i]` |
| y_pred | float64 | 4.9 | `.oof_predictions[i]` |

### `metrics.parquet` — per-target, per-metric
| Column | Type | Example | Source field on TargetResult |
|--------|------|---------|---------------------------|
| approach | str | "ring_agg_k10" | constructor arg |
| target_variable | str | "lbm" | `.target` |
| metric | str | "r2" | literal |
| value | float64 | 0.556 | `.overall_r2` / `.overall_rmse` / `.overall_mae` |

**Output path**: `data/study_areas/{area}/probe_results/{approach}/predictions.parquet` + `metrics.parquet`

**Approach naming convention** (agreed with coral-falling-snow):
`concat_208d`, `ring_agg_k10`, `unet_128d_1000ep`, `accessibility_unet_64d`, etc.

---

## Files to Create/Modify

### 1. `utils/paths.py` — add `probe_results()` + `probe_results_root()` to StudyAreaPaths

Insert after `stage3_run()` (line 248), before `latest_run()`:

```python
# -----------------------------------------------------------------
# Probe results (cross-approach comparison store)
# -----------------------------------------------------------------

def probe_results_root(self) -> Path:
    """Base directory for all probe result approaches."""
    return self.root / "probe_results"

def probe_results(self, approach: str) -> Path:
    """Directory for a specific approach's probe results.

    Layout::
        data/study_areas/{area}/probe_results/{approach}/
        ├── predictions.parquet
        └── metrics.parquet
    """
    return self.probe_results_root() / approach
```

Note: follows existing convention — `StudyAreaPaths` does NOT mkdir (docstring line 31-32).

### 2. `stage3_analysis/probe_results_writer.py` — NEW

```python
class ProbeResultsWriter:
    """Standardized probe results writer.

    Converts Dict[str, TargetResult] → two parquets with fixed schema.
    """

    def __init__(self, study_area: str, approach: str):
        self.paths = StudyAreaPaths(study_area)
        self.approach = approach

    def write(self, results: Dict[str, TargetResult]) -> Path:
        """Write predictions.parquet + metrics.parquet.

        Returns the approach directory path.
        """
        # Build predictions DataFrame: one row per (target, region_id)
        # Build metrics DataFrame: one row per (target, metric)
        # Write both to self.paths.probe_results(self.approach)/

    @classmethod
    def write_from_regressor(cls, regressor, approach: str, study_area: str) -> Path:
        """Convenience: regressor.results → write()."""
        writer = cls(study_area, approach)
        return writer.write(regressor.results)
```

Implementation details:
- Import `TargetResult` from `stage3_analysis.linear_probe`
- `approach` baked as column value (redundant with path, but enables concat across approaches)
- Metrics extracted: `overall_r2`, `overall_rmse`, `overall_mae` (3 rows per target)
- For classification TargetResults (`task_type == "classification"`): also emit `accuracy`, `f1_macro`
- No `run_info.json` — this is a derivative store, not a primary run

### 3. `stage3_analysis/comparison_plotter.py` — NEW

```python
class ProbeComparisonPlotter:
    """Cross-approach probe comparison visualization.

    Globs probe_results/*/predictions.parquet to build side-by-side views.
    """

    def __init__(self, study_area: str, output_dir: Optional[Path] = None,
                 h3_resolution: int = 9, dpi: int = 150):
        self.paths = StudyAreaPaths(study_area)
        self.study_area = study_area
        self.db = SpatialDB.for_study_area(study_area)
        self.h3_resolution = h3_resolution
        self.dpi = dpi
        self.output_dir = output_dir or (self.paths.root / "probe_results" / "comparison")

    def load_all(self, approaches: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Glob and concat all approach parquets. Filter to `approaches` if given."""

    def plot_r2_bars(self, metrics_df: pd.DataFrame) -> Path:
        """Grouped bar chart: x=target_variable, grouped bars=approach, y=R².
        Sorted by best R² per target. Color per approach."""

    def plot_scatter(self, predictions_df: pd.DataFrame, target: str) -> Path:
        """Faceted y_true vs y_pred scatter. One subplot per approach.
        1:1 reference line, R² annotation, shared axes."""

    def plot_residual_maps(self, predictions_df: pd.DataFrame, target: str) -> Path:
        """Side-by-side spatial residual maps. One panel per approach.
        residual = y_pred - y_true. RdBu_r symmetric colormap.
        SpatialDB.centroids() for positions, load_boundary() for overlay.
        Shared colorbar norm across panels."""

    def plot_all(self, target: Optional[str] = None) -> List[Path]:
        """Generate all plots. If target=None, use first target_variable found."""
```

Implementation details:
- Grid layout: `ncols = min(4, n_approaches)`, `nrows = ceil(n / ncols)`
- Spatial rendering: `rasterize_continuous()` from `utils/visualization.py:130`
- Map overlay: `plot_spatial_map()` from `utils/visualization.py:366`
- Boundary: `load_boundary()` from `utils/visualization.py:56`
- Stamp: `stamp = max(1, 11 - resolution)` (res9→2, res10→1)
- Centroids in EPSG:28992 via `SpatialDB.centroids(hex_ids, resolution, crs=28992)`
- Extent via `SpatialDB.extent(hex_ids, resolution, crs=28992)`
- Date-keyed output: `output_dir / YYYY-MM-DD / {plot_name}.png`
- Matplotlib setup: `sns.set_style("whitegrid")`, DPI 150
- Colormaps: `RdBu_r` diverging (symmetric, `TwoSlopeNorm`), `viridis` continuous
- CLI via `if __name__ == "__main__"` with argparse: `--study-area`, `--approaches`, `--target`, `--output-dir`, `--resolution`

### 4. `stage3_analysis/__init__.py` — add exports

Add `ProbeResultsWriter` and `ProbeComparisonPlotter` to imports + `__all__`.

---

## Wave Structure

### Wave 1 — Implementation (parallel, 2 agents)

| # | Agent | Task | Files | Acceptance criteria |
|---|-------|------|-------|-------------------|
| 1a | stage3-analyst | Build `probe_results_writer.py` + add `probe_results()`/`probe_results_root()` to `utils/paths.py` | `stage3_analysis/probe_results_writer.py` (new), `utils/paths.py` (edit) | Schema matches contract above. `write()` returns Path. Handles both regression and classification TargetResult. |
| 1b | stage3-analyst | Build `comparison_plotter.py` | `stage3_analysis/comparison_plotter.py` (new) | `load_all()` returns correct DataFrame shapes. All 3 plot methods produce PNGs. CLI works. Uses SpatialDB, not raw h3. |

**__init__.py ownership**: Agent 1a owns `stage3_analysis/__init__.py` edits (prevents merge conflict).

### SYNC 1 — Human decision point

> **Go/no-go**: Do both agents' outputs match the schema contract?
> Check: `predictions.parquet` columns, `metrics.parquet` columns, path construction.
> If yes → Wave 2. If no → fix agent re-dispatch.

### Wave 2 — Verification (1 agent)

| # | Agent | Task | Acceptance criteria |
|---|-------|------|-------------------|
| 2 | qaqc | Light verification: imports resolve, TargetResult type consistency, StudyAreaPaths usage, no hardcoded paths, CLI arg parsing | Verdict: committable or list of fixes |

**Depth**: code_quality=4, test_coverage=2 → check types and imports, don't write tests.

### SYNC 2 — Human decision point

> **Go/no-go**: Is QAQC verdict "committable"?
> If yes → Wave 3. If QAQC found issues → fix wave, then re-verify.
> If issues are minor (cosmetic) → human can override to Wave 3.

### Wave 3 — Commit (1 agent)

| # | Agent | Task |
|---|-------|------|
| 3 | devops | `git add` the 3 new/modified files, commit with descriptive message |

### SYNC 3 — Human decision point

> **Go/no-go**: Is the commit clean? Ready to notify coral-falling-snow?
> If yes → send "ready" message to coral, proceed to Final Wave.

### Wave 4 — Notify coral-falling-snow

Send coordinator message:
```
from: jade-falling-wind
to: coral-falling-snow
type: info
text: |
  Writer shipped. You can now use:
    from stage3_analysis.probe_results_writer import ProbeResultsWriter
    ProbeResultsWriter.write_from_regressor(regressor, "concat_208d", "netherlands")
  Output: probe_results/{approach}/predictions.parquet + metrics.parquet
  Go ahead with your baseline runs.
```

### Final Wave — Close-out (mandatory)

1. Write coordinator scratchpad (`.claude/scratchpad/coordinator/2026-03-21.md`)
2. `/librarian-update` (parallel with 3)
3. `/ego-check` (parallel with 2)

---

## Key References

| What | Where |
|------|-------|
| `TargetResult` dataclass | `stage3_analysis/linear_probe.py:132-154` |
| `FoldMetrics` dataclass | `stage3_analysis/linear_probe.py:119-130` |
| `StudyAreaPaths` class | `utils/paths.py:19-305` |
| `write_run_info()` | `utils/paths.py:307-350` |
| `rasterize_continuous()` | `utils/visualization.py:130` |
| `plot_spatial_map()` | `utils/visualization.py:366` |
| `load_boundary()` | `utils/visualization.py:56` |
| `SpatialDB` | `utils/spatial_db.py` |
| `LinearProbeVisualizer.plot_spatial_residuals()` | `stage3_analysis/linear_probe_viz.py:454-635` (template for residual maps) |
| `DNNProbeVisualizer.plot_comparison_bars()` | `stage3_analysis/dnn_probe_viz.py:271` (template for bar charts) |
| Existing `__init__.py` | `stage3_analysis/__init__.py` (19 exports, ~39 lines) |

---

## Verification (post-commit)

```python
# Quick smoke test — run in Python REPL
from stage3_analysis.probe_results_writer import ProbeResultsWriter
from stage3_analysis.comparison_plotter import ProbeComparisonPlotter
from stage3_analysis.linear_probe import TargetResult
import numpy as np

# Mock a TargetResult
tr = TargetResult(
    target="lbm", target_name="Overall Liveability",
    best_alpha=0.0, best_l1_ratio=0.0, fold_metrics=[],
    overall_r2=0.55, overall_rmse=0.8, overall_mae=0.6,
    coefficients=np.zeros(10), intercept=0.0,
    feature_names=[f"f{i}" for i in range(10)],
    oof_predictions=np.random.randn(100),
    actual_values=np.random.randn(100),
    region_ids=np.array([f"89283{i:010d}fff" for i in range(100)]),
)

# Write
path = ProbeResultsWriter.write_from_regressor(
    type("R", (), {"results": {"lbm": tr}})(), "test_approach", "netherlands"
)
print(f"Written to: {path}")

# Verify schema
import pandas as pd
pred = pd.read_parquet(path / "predictions.parquet")
metr = pd.read_parquet(path / "metrics.parquet")
assert list(pred.columns) == ["approach", "target_variable", "region_id", "y_true", "y_pred"]
assert list(metr.columns) == ["approach", "target_variable", "metric", "value"]
print("Schema OK")
```

## Execution
Invoke: `/niche .claude/plans/2026-03-21-probe-results-writer-and-comparison-plotter.md`
