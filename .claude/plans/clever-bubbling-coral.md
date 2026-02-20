# Plan: Strip Classification Branches from Regression Probes + Commit

## Context
The classification probe suite (4 new files) is complete. The old regression files (`linear_probe.py`, `dnn_probe.py`, `linear_probe_viz.py`, `dnn_probe_viz.py`) still have classification branches that are now redundant. Strip them to keep regression files pure regression.

## Critical Constraint: Shared Dataclasses
The new classification probes **import** `FoldMetrics` and `TargetResult` from `linear_probe.py`. These fields MUST STAY in the dataclasses:
- `FoldMetrics.accuracy`, `FoldMetrics.f1_macro`
- `TargetResult.overall_accuracy`, `overall_f1_macro`, `n_classes`, `task_type`
- `TAXONOMY_TARGET_COLS`, `TAXONOMY_TARGET_NAMES` (imported by classification probes)

Only remove classification CODE PATHS (branching, imports used only by those paths, CLI args).

## File 1: `stage3_analysis/linear_probe.py` (~14 changes)

### Remove imports (only those that become unused):
- `LogisticRegression` from sklearn.linear_model (line 27)
- `accuracy_score`, `f1_score` from sklearn.metrics (lines 29-30)

### Remove config fields + auto-detection:
- `task_type` field from `LinearProbeConfig` (line 78)
- Auto-detect block: `if self.task_type == "regression": self.task_type = "classification"` (lines 106-107)
- Keep the `if self.target_cols == list(TARGET_COLS): self.target_cols = list(TAXONOMY_TARGET_COLS)` part (lines 104-105) — regression can still target taxonomy cols

### Keep dataclass fields (used by classification probes):
- `FoldMetrics.accuracy`, `FoldMetrics.f1_macro` — KEEP
- `TargetResult.overall_accuracy`, `overall_f1_macro`, `n_classes`, `task_type` — KEEP

### Strip `_train_and_evaluate_cv`:
- Remove `task_type` parameter
- Remove entire `if task_type == "classification":` block (lines 357-384)
- Dedent the regression `else` block to be the only code path

### Strip `run`:
- Remove `task_type` variable usage, hard-code "Regression" for mode_label
- Remove classification logging branch (lines 542-545)
- Remove classification result-building block (lines 555-584)
- Remove classification summary logging (lines 621-624)
- Remove `task_type=task_type` from `_train_and_evaluate_cv` call

### Strip `save_results`:
- Remove classification metrics from metrics_summary (lines 649-652)
- Remove classification fold metrics branching (lines 657-659)
- Keep `task_type` in row dict (for format compatibility)

### Strip CLI:
- Remove `--task-type` argument (lines 742-744)
- Remove `task_type=args.task_type` from config creation

## File 2: `stage3_analysis/dnn_probe.py` (~24 changes)

### Remove imports:
- `accuracy_score`, `f1_score` from sklearn.metrics (lines 37-38)

### Remove config fields:
- `task_type` field (line 73)
- `n_classes` field (line 74)
- Auto-detect: `if self.task_type == "regression": self.task_type = "classification"` (lines 117-118)

### Keep `MLPProbeModel.output_dim` parameter (used by `dnn_classification_probe.py`)

### Strip `_train_one_fold`:
- Remove `task_type`, `n_classes`, `label_offset` parameters
- Remove `is_clf` variable
- Remove classification label prep (lines 455-459), keep only regression target standardization
- Remove classification tensor creation (lines 473-475), keep float32 tensors
- Hard-code `output_dim = 1`
- Hard-code `criterion = nn.MSELoss()`
- Remove classification loss branch in training loop
- Remove classification val_loss branch

### Strip `run_for_target`:
- Remove `task_type`/`is_clf` variables, `label_offset`/`n_classes` computation
- Remove classification kwargs from `_train_one_fold` call
- Remove classification prediction branch (argmax)
- Remove classification fold metrics
- Remove classification overall metrics + TargetResult block

### Strip `run`:
- Hard-code "Regression" for mode_label
- Remove classification summary logging

### Strip `save_results`:
- Remove classification metrics from metrics_summary

### Strip CLI:
- Remove `--task-type` argument
- Remove `task_type=args.task_type` from config creation

## File 3: `stage3_analysis/linear_probe_viz.py` (~5 changes)

### Remove:
- `from sklearn.metrics import confusion_matrix` import (line 36)
- Classification branch in `plot_fold_metrics` (lines 1077-1095)
- Entire `plot_confusion_matrix` method (lines 1124-1185)
- Entire `plot_classification_metrics_comparison` method (lines 1187-1253)
- Classification branching in `plot_all` (lines 1284, 1297-1299, 1311-1312)

## File 4: `stage3_analysis/dnn_probe_viz.py` (~3 changes)

### Remove:
- `any_clf` variable in `plot_all` (line 786)
- Classification metrics comparison call (lines 811-812)

## Delegation Plan (Waves)

### Wave 1 (parallel, 2x stage3-analyst):
- **Agent A**: Strip `linear_probe.py` + `linear_probe_viz.py`
- **Agent B**: Strip `dnn_probe.py` + `dnn_probe_viz.py`

### Wave 2 (parallel, 2 agents):
- **qaqc**: Verify no regression behavior broke, classification imports from linear_probe still work
- **librarian**: Update codebase graph

### Wave 3 (sequential):
- **execution**: Smoke test imports + `--help` for all 4 probe modules

### Wave 4 (parallel + coordinator):
- **devops**: Commit all changes (4 new + 6 edited files)
- **ego**: Process health check
- Coordinator scratchpad

## Verification
1. `python -c "from stage3_analysis import ClassificationProber, DNNClassificationProber"` — classification probes still import OK
2. `python -m stage3_analysis.linear_probe --help` — no --task-type arg
3. `python -m stage3_analysis.dnn_probe --help` — no --task-type arg
4. `python -m stage3_analysis.classification_probe --help` — works
5. `python -m stage3_analysis.dnn_classification_probe --help` — works
6. `grep -r "task_type.*classification" stage3_analysis/linear_probe.py` — no hits
7. `grep -r "task_type.*classification" stage3_analysis/dnn_probe.py` — no hits
