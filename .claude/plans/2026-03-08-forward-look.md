## Ego Forward-Look -- 2026-03-08 (for coordinator)

### Recommended Focus

1. **Fix LR schedule and retrain UNet (P0).** The cosine schedule decays to near-zero by epoch 400, causing stalling. Two consecutive non-converged runs. The vrz=0.849 result is promising but all 6 probe numbers are unreliable until convergence. Options: (a) OneCycleLR with warmup, (b) flat LR with step decay at 80%/90%, (c) cosine with 2000 epochs. Pick one and run. This is a 15-minute code change + 6-minute training run.

2. **Implement checkpoint versioning (P0).** `best_model.pt` has been overwritten 3 times. Save as `best_model_{year}_{dims}_{date}.pt` with symlink to `best_model.pt`. Trivial implementation, prevents irreversible loss. Assign to stage2-fusion-architect.

3. **Re-probe with converged model.** Once the retrain completes, rerun `probe_20mix_multiscale.py` to get honest numbers. The current vrz result may hold or improve. The 5 targets where concat wins may flip if the UNet actually converges.

4. **CE report disclaimers (P1).** Two lines at the top of each CE report: "Note: R-squared numbers below are from an unconverged UNet (best epoch=max). Treat as directional only." Five-minute fix.

### Unresolved Tensions

- **GTFS prefix divergence.** Code says `G00..G63`, disk says `gtfs2vec_0..63`. Concat works because it reads disk. But if someone re-runs the GTFS processor, the output changes to `G00..G63`, and then concat breaks because MODALITY_PREFIXES was corrected to `"gtfs": "gtfs2vec_"`. The code path and the intended format are inconsistent. Either revert the code to `gtfs2vec_` naming or re-run the processor and update MODALITY_PREFIXES back to `"G"`.
- **QAQC chronic carry items.** spatial_db CRS test (10+ sessions), AE geometry leakage (9+ sessions). These need a WONTFIX decision or a scheduled fix. Carrying them forward costs attention budget every session.
- **One-off script proliferation.** 7 files in `scripts/one_off/`. The two new probe-rerun scripts overlap with the durable `probe_20mix_multiscale.py`. Recommend folding `spatial_maps_from_saved.py` into the durable script as a `--from-saved` flag, and deleting `clustering_probe_from_saved.py` once its logic is in the durable script.

### Agent Invocation Plan

1. **stage2-fusion-architect**: Fix LR schedule + add checkpoint versioning. Single focused delegation.
2. **Coordinator executes training**: Same pattern as today -- run the CLI command directly, no agent needed for a 6-minute GPU job.
3. **stage3-analyst**: Re-probe with converged model. Reuse `probe_20mix_multiscale.py`. Generate all 6 spatial maps (not just lbm/vrz).
4. **qaqc**: Verify converged model output shapes. Reconcile chronic carry items (CRS test: WONTFIX? AE geometry: WONTFIX?). Do NOT carry forward again without a decision.
5. **librarian**: Update CLAUDE.md FullAreaUNet description (64D output, named mappings, pyramid dims). Update probe infrastructure list.
6. **devops**: Message directory cleanup (delete messages older than 7 days). Currently 85 files.

### Risks and Concerns

- **Converged model may not beat concat on all targets.** The UNet's advantage on vrz (macro-scale amenity) is clear, but it may structurally lose on targets where local features dominate (fys, won). If convergence does not improve those 5 targets, the conclusion is that UNet adds value for spatial/hierarchical targets but not for local-feature targets. This is a valid scientific finding, not a failure.
- **GTFS 97.2% background vectors.** The fusion model sees 844K identical GTFS vectors. Without gating/masking, the UNet may learn to ignore GTFS entirely. This undermines the 4-modality story. Gating is P1 but will become P0 if probe results don't improve after convergence.
- **Stale plan files.** `2026-03-09-4modality-unet-and-interpretability.md` references UNet++ which was superseded. Anyone reading plans for orientation gets a false signal. Delete or archive stale plans.

### Process Improvements

- **Inline trivial fixes during training.** While the 6-minute training run executes, the coordinator could have implemented checkpoint versioning and CE disclaimers. Both are <5 minutes. Dead time during GPU jobs is an opportunity for housekeeping.
- **Message cleanup as OBSERVE step.** Add a 1-line check at the start of each coordinator session: delete messages older than 7 days. Prevents the 85-file accumulation.
- **One-off script discipline.** When a one-off script's logic is needed more than once, fold it into the durable script as a flag/mode immediately. Do not create a second one-off for the same domain.
- **QAQC carry-item escalation rule.** The 5-session rule exists but is not enforced. Next session, the coordinator should explicitly WONTFIX or schedule every carry item over 5 sessions. No more silent carries.
