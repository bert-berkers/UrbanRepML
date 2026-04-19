# Plan: Script Cleanup & Coordinator Hello Broadcast

**Created**: 2026-03-01
**Task**: Clean up one-off script bloat (especially run/plot scripts), establish guardrails to prevent recurrence, and add mandatory coordinator startup announcements.

## Context

Audit from previous session found:
- **63 total scripts**, **26 in one_off/** or ad-hoc locations
- **8 roads scripts** doing essentially the same thing (`run_roads_*.py` variants)
- **4 one-off plot scripts** (3 in `one_off/`, 1 at top-level `plot_targets.py`)
- **6 superseded training scripts** in `scripts/netherlands/` (replaced by cone-based training)
- **1 broken script**: `scripts/processing_modalities/multimodal integration/test_multimodal_pipeline.py` (space in path)
- `dnn_probe_sweep.py` is in `one_off/` but is a real reusable tool — should be promoted

Coordinator gap: coordinators check for incoming messages at Wave 0 but never **send** one. If a coordinator starts and nobody's there, the next coordinator has no context about what happened.

## Wave 1: Script Cleanup (parallel)

### 1a. spec-writer: Define script hygiene conventions
- Draft a "Script Hygiene" section for CLAUDE.md with:
  - **Three-tier organization**: `scripts/` (durable tools) → `scripts/archive/` (historical, read-only) → `scripts/one_off/` (temporary, expected to be cleaned)
  - **New-script gate**: every new script must have a docstring explaining purpose, expected lifetime (durable/temporary), and which module it exercises
  - **Periodic cleanup rule**: coordinator Final Wave should flag scripts in `one_off/` older than 30 days
  - **No hardcoded paths** — use `utils/paths.py` `StudyAreaPaths`
  - **No tests in scripts/** — tests belong in `tests/`
- Write to `specs/script-hygiene.md` first, then the CLAUDE.md addition
- Acceptance: spec file + CLAUDE.md patch ready for review

### 1b. general-purpose: Identify exact archive/delete targets
- Read every script in the categories below and confirm each is truly dead/superseded:
  - Roads scripts: `run_roads_duckdb_direct.py`, `run_roads_full_netherlands.py`, `run_roads_full_netherlands_no_progress.py`, `run_roads_processor.py`, `run_roads_processor_online.py` — check if any has unique logic not in `stage1_modalities/roads/`
  - Plot scripts: `plot_classification_comparison.py`, `plot_dnn_vs_linear_scatter.py`, `plot_linear_vs_dnn_comparison.py`, `plot_targets.py` — check if any is still imported or referenced
  - Superseded training: `run_experiment.py`, `run_experiment_fsi95.py`, `run_hexagonal.py`, `run_netherlands_lattice_unet.py` — replaced by `train_lattice_unet_res10_cones.py` and `train_cone_alphaearth.py`
  - Broken: `scripts/processing_modalities/multimodal integration/test_multimodal_pipeline.py`
  - `bridge_archive_embeddings.py` — one-time data migration, no longer needed
- Produce a categorized list: ARCHIVE (move to `scripts/archive/`), DELETE (truly dead), KEEP (still needed)
- Acceptance: categorized list with justification for each

### 1c. spec-writer: Draft coordinator hello broadcast spec
- Read `.claude/skills/coordinate/skill.md` Wave 0 section
- Design a mandatory "hello" broadcast step added to Wave 0 (after step 6, before proceeding to OODA)
- Every coordinator announces on startup: session ID, task summary, intent, risk areas, claimed paths
- Fires even if no other coordinators are active — the message is for the NEXT coordinator, not just current ones
- Format: use existing `coordinator_registry` message system with level `info`
- Write to `specs/coordinator-hello.md`
- Acceptance: spec file with exact text to add to skill.md

## Wave 2: Implementation (parallel, after Wave 1)

### 2a. devops: Execute script moves/deletes
- Follow the categorized list from 1b
- Move ARCHIVE targets to `scripts/archive/` (with appropriate subdirectory)
- Delete DELETE targets
- Promote `dnn_probe_sweep.py` from `one_off/` to `scripts/tools/` or `scripts/analysis/`
- Fix the space-in-path issue for `multimodal integration/`
- Acceptance: all moves/deletes done, no broken imports

### 2b. spec-writer: Apply CLAUDE.md script hygiene section
- Take the spec from 1a and apply the CLAUDE.md edit
- Keep it concise — this is a conventions section, not a tutorial
- Acceptance: CLAUDE.md updated with script hygiene section

### 2c. spec-writer: Apply coordinator hello broadcast
- Take the spec from 1c and edit `.claude/skills/coordinate/skill.md`
- Add hello broadcast step to Wave 0 section
- Acceptance: skill.md updated, hello fires on next `/coordinate` invocation

## Wave 3: Verification

### 3a. qaqc: Verify cleanup and conventions
- Verify all archive/delete moves were executed correctly
- Check no broken imports reference deleted scripts
- Verify CLAUDE.md script hygiene section is clear and actionable
- Verify coordinator skill.md hello broadcast addition is well-formed
- Produce commit-readiness verdict
- Acceptance: PASS/FAIL verdict with findings

## Wave 4: Commit

### 4a. devops: Commit in logical chunks
- Commit 1: `chore: archive/remove superseded and dead scripts`
- Commit 2: `chore: add script hygiene conventions to CLAUDE.md`
- Commit 3: `feat: add mandatory hello broadcast to coordinator Wave 0`
- Acceptance: 3 clean commits, all pushed

## Final Wave: Close-out (mandatory)

1. Write coordinator scratchpad at `.claude/scratchpad/coordinator/2026-03-01.md`
2. Invoke `/librarian-update` (parallel with 3)
3. Invoke `/ego-check` (parallel with 2)

## Execution

Invoke: `/coordinate .claude/plans/script-cleanup-and-coordinator-hello.md`
