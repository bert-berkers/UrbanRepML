# Clean House — Scripts, Plots, SpatialDB, Output Organization, Docs

**Intent**: Get the project back in shape. Consolidate drifting one-off scripts, fix plots, verify SpatialDB, organize outputs into date-keyed dirs, fix stale docs.

## Wave 0: Clean State
- `git status`, commit any dirty state
- `/summarize-scratchpads` for orientation

## Wave 0.5: Fix Terminal Identity (PPID → Terminal Shell PID)

**Problem discovered during Wave 0**: The PPID-keyed session isolation system is fundamentally broken. The documented workflow — `/valuate` → `/clear` → `/niche` — crosses a process boundary that invalidates the key.

### What went wrong

We assumed `os.getppid()` returns a PID that's stable per terminal. It doesn't. It returns the PID of the **Claude Code process** (node.exe), which is a child of the terminal shell. When the user runs `/clear`, Claude Code restarts as a new process with a new PID. So:

- During `/valuate`: `os.getppid()` = `32684` (Claude Code v1)
- After `/clear` + `/niche`: `os.getppid()` = `33768` (Claude Code v2)
- Session/supra files keyed to `32684` are invisible to the new process

This means the supra identity set during `/valuate` is lost every time the user follows the standard workflow. Multi-terminal isolation also breaks unpredictably.

### The process tree (observed on this machine)

```
explorer.exe (8476)
  └── pycharm64.exe (19260)        ← IDE (shared across all terminals)
        └── powershell.exe (3224)   ← TERMINAL SHELL — stable per terminal tab ✓
              └── node.exe (32604)  ← Claude Code — changes on /clear ✗
                    └── bash.exe...
                          └── python.exe (hooks)
```

The **terminal shell** (powershell.exe PID 3224) is the correct stable key. It's spawned when the user opens a terminal tab and persists across all `/clear` restarts because Claude Code is its child, not its parent. Each terminal tab has its own shell PID, so multi-terminal isolation is naturally preserved.

### The fix

Replace `os.getppid()` with a new `get_terminal_pid()` function that walks up the process tree via `psutil` to find `node.exe`, then returns its parent's PID (the terminal shell). This is a drop-in replacement across all 6 call sites in 3 files.

### Alternatives considered

| Approach | `/clear` survives? | Multi-terminal? | Why rejected |
|---|---|---|---|
| **PPID** (current) | No | Yes (while it lasts) | Breaks at the exact seam we always use |
| **Env var in shell** | Can't set from child | N/A | Child processes can't export to parent shell |
| **"Most recent today"** | Yes | No | Terminal B's `/valuate` overwrites Terminal A |
| **Session name as arg** | Yes | Yes | Manual — user must remember/type the name |
| **Shell wrapper function** | Yes | Yes | Requires `.bashrc` modification, extra setup |
| **Terminal shell PID** ✓ | Yes | Yes | Requires `psutil`, but already available |

### Tasks

1. **devops**: Add `get_terminal_pid()` to `coordinator_registry.py`. Walk up process tree via `psutil`, find `node.exe`, return its parent PID. Fallback to `os.getppid()` if psutil unavailable or tree walk fails. Replace all 6 `os.getppid()` call sites across `coordinator_registry.py`, `supra_reader.py`, `stop.py`.

2. **devops**: Re-key existing session/supra files from old PPID (`32684`) to new terminal PID so this session's identity is immediately usable. Files: `.claude/coordinators/sessions/`, `.claude/coordinators/supra/`, `.claude/supra/sessions/`.

## Wave 1: Audit (parallel)

1. **Librarian**: Audit `scripts/one_off/` — for each script: (a) does it have a docstring? (b) is it older than 30 days (shelf life per CLAUDE.md)? (c) should it be promoted to `scripts/stage3/` or `scripts/stage2/`, archived to `scripts/archive/`, or deleted? Produce a triage list with recommended action per file. Also check `scripts/` root for loose scripts that should be in subdirs.

2. **Librarian**: Audit `scripts/stage3/` — identify scripts with overlapping functionality (e.g. multiple probe comparison scripts, multiple plot scripts). Flag candidates for consolidation. Check all scripts for docstrings and `StudyAreaPaths` usage (hardcoded paths = violation).

3. **srai-spatial**: Audit SpatialDB usage — check `utils/spatial_db.py` implementation, then grep all call sites. Verify: (a) is SpatialDB actually used in production code or just referenced? (b) does the SedonaDB backend work or do all sites fall back to GeoPandas? (c) is `tests/test_spatial_db.py` passing? Report on actual usage vs aspirational references.

4. **Explore**: Scan output directories — find where scripts write results (plots, CSVs, embeddings). Check if any use date-keyed subdirs already. Map the current output structure so Wave 2 knows what to organize.

5. **Librarian**: Audit CLAUDE.md for stale content. Ego flagged: (a) FullAreaUNet still says 128D output (actual: 64D since March 8), (b) skill names still say /attune and /coordinate (now /valuate and /niche), (c) probe infrastructure list incomplete. Also check stale plan files in `.claude/plans/` — ego flagged `2026-03-09-4modality-unet-and-interpretability.md` as referencing superseded UNet++ approach.

6. **Human**: Go through existing plot outputs (maps, probe comparisons, cluster visualizations, CE plots). Make a list of plots that look bad — especially maps with wrong extents, missing labels, ugly colormaps, clipped hexagons, or misaligned boundaries. This list feeds Wave 2 so we can fix the plotting code to never produce bad plots again.

## Wave 2: Fix (sequential decisions based on Wave 1)

7. **devops**: Execute the one-off triage — move/archive/delete scripts per Wave 1 recommendations (after coordinator confirms the list with user).

8. **devops**: Fix CLAUDE.md issues from audit: update FullAreaUNet description (64D), skill names (/valuate, /niche, /sync, /vitals), probe infrastructure list. Archive stale plan files.

9. **stage3-analyst**: Consolidate probe comparison scripts if Wave 1 found overlap. The goal: one flexible `compare_probes.py` (or similar) that handles the variations, not 5 separate scripts that each do a slightly different comparison.

10. **stage3-analyst**: Fix bad plots from human's list (Wave 1, item 6). Harden the plotting code — fix map extents, colormaps, labels, hex rendering. The goal is that our standard plotting functions produce good-looking output by default, so one-off plot scripts become unnecessary.

11. **devops**: Add date-keyed output organization — update `StudyAreaPaths` or a shared utility so scripts write to `{output_dir}/YYYY-MM-DD/` by default. Fix any scripts that write to flat dirs.

## Wave 3: Verify (parallel)

12. **qaqc**: Run `tests/test_spatial_db.py` and any other relevant tests. Check that moved/archived scripts don't break imports. Verify date-keyed output works. Fix QAQC chronic carry items: spatial_db CRS test and AE geometry leakage (both deferred 11+ sessions — fix them, don't WONTFIX).

13. **qaqc**: Review plot quality — rerun a fixed plot script and compare before/after. Verify the fixes from item 10 produce good output.

## Wave 4: Guardrails

14. **spec-writer**: Write a `.claude/rules/script-discipline.md` rule that agents pick up automatically. Should cover: (a) before creating a new script, check if an existing script or function already does what you need — extend it, don't duplicate; (b) one-off scripts MUST go in `scripts/one_off/` with a docstring and expiry date; (c) plotting must use the shared visualization functions (from Wave 2 fixes), not inline matplotlib; (d) all output goes to date-keyed dirs. Also update `CLAUDE.md` script organization section if needed.

15. **qaqc**: Verify the new rule is syntactically correct and will be auto-loaded by the rules engine. Spot-check that existing durable scripts comply.

## Final Wave: Close-out
- Write coordinator scratchpad
- Commit in logical chunks
- `/ego-check`

## Execution
Invoke: `/niche .claude/plans/2026-03-15-clean-house.md`
