# Implement /valuate + /niche Graph-Driven Orchestration & Temporal Priors

## Context
Spec: `specs/temporal-supra-profiles.md`
Graph: `deepresearch/liveability_approaches_graph.json`
Memory: `memory/liveability_graph_ontology.md`
Current skills: `/valuate`, `/niche` (renamed from `/attune`, `/coordinate` — 2026-03-13)

## Ontology (corrected 2026-03-13)
- **Indicators** = codebase/filesystem (objective)
- **Percepts** = sequential context windows of coordinator runs within one terminal (subjective)
- **Needs** = hard constraints (budget). **Desires** = soft constraints (utility function). Rational choice.
- **Per ↔ Per lateral coupling** = `/sync` message passing (homo narrans), NOT scratchpads
- **Scratchpads** = memory traces of dead percepts (intra-lineage inheritance)
- **Session_id** = narrator identity for homo narrans (persists across percept deaths)

## DONE (2026-03-13): Skill rename + spec rewrite
- `/attune` → `/valuate`, `/coordinate` → `/niche` — completed
- Frontmatter, cross-references, old dirs deleted
- Spec theoretical grounding rewritten with corrected ontology
- Graph JSON: N/D → Per direction fixed in dynamic section
- Graph renderer: re-rendered PNGs with correct arrows
- Memory file: `liveability_graph_ontology.md` written

## DONE (2026-03-13): Waves 1-3 implemented + committed

Commit `6c84004` — 10 files, 998 insertions. QAQC: all 8 test groups PASS.
- supra_reader.py: +13 functions (temporal segment, supra session, graph topology, EMA priors)
- session-start.py: supra session registration
- stop.py: temporal prior update on exit
- subagent-context.py: supra ID injection + lateral coupling gate
- coordinator_registry.py: write_lateral_message()
- valuate/niche/sync SKILL.md: temporal prior + graph framing
- temporal_priors.yaml: bootstrapped (sunday-morning: 3 obs, friday-evening: 2 obs)
- Bootstrap script: scripts/one_off/bootstrap_temporal_priors.py

### Known items for follow-up
- `.claude/sync_run.py` not yet updated to use lateral coupling gate (SKILL.md describes desired behavior)
- `friday-evening.yaml` profile has `lemode:` typo (cosmetic)
- `.current_supra_session_id` only created when SessionStart hook fires (not manually bootstrapped sessions)

---

## Wave 1: Temporal prior functions + graph integration (parallel)

1. **devops**: Add to `supra_reader.py`:
   - `_temporal_segment_key()` — returns e.g. `friday-evening` from local time
   - `_supra_session_id()` — returns `friday-evening-2026-03-13`
   - `get_active_graph()` — returns `'static'` or `'dynamic'`
   - `get_edge_topology()` — loads edges from JSON for active graph
   - `is_lateral_coupling_active()` — True during `/niche`, False during `/valuate`
   - `read_temporal_priors()` / `get_temporal_prior()`
   - `record_temporal_observation()` — EMA update with alpha=0.3
   - `temporal_prior_to_states()` — round floats to ints

2. **devops**: Create `supra/temporal_priors.yaml` with version header + empty segments

3. **devops**: Update `session-start.py`:
   - Compute supra session ID on start
   - Create/join session file at `supra/sessions/{supra_session_id}.yaml`
   - Write `.current_supra_session_id`
   - Register narrator identity for `/sync`

4. **devops**: Update `stop.py`:
   - Fire temporal prior update if session was valuated
   - Write scratchpad (percept death trace)

5. **devops**: Update `/valuate` SKILL.md:
   - Step 1: read temporal prior for current segment
   - Step 3: add `use prior` shorthand
   - Step 3.5: surface temporal prior in morning inread
   - Step 5: record temporal observation after writing states

6. **devops**: Update `/niche` SKILL.md:
   - Add graph-theoretical framing (percepts, indicators, N/D→Per)
   - Note that `/sync` IS the Per ↔ Per lateral coupling
   - Clarify: OODA checkpoints = course correction, not re-valuation

## Wave 2: Lateral coupling enforcement (parallel, after W1)

7. **devops**: Update `coordinator_registry.py`:
   - Messages use supra session_id as sender identity (narrator name)
   - Add `is_lateral_coupling_active()` gate

8. **devops**: Update `subagent-context.py`:
   - Inject supra session ID alongside coordinator session ID
   - Gate cross-agent scratchpad injection on `is_lateral_coupling_active()`

9. **devops**: Update `/sync` skill:
   - Use supra session_id as narrator identity
   - Only active during dynamic graph (gate on `is_lateral_coupling_active()`)

## Wave 3: Bootstrap + verify (sequential after W2)

10. **devops**: Bootstrap migration:
    - Convert 3 existing poetic session files to `{segment}-{date}.yaml` format
    - Seed temporal priors from existing profiles + sessions

11. **qaqc**: Verify:
    - `/valuate` skill loads and runs
    - `/niche` skill loads and runs
    - Temporal prior file updates on `/valuate`
    - Session identity persists (new process sees existing supra session file)
    - `/sync` uses supra session_id as narrator
    - Lateral coupling gated by active graph
    - All old `/attune` and `/coordinate` references gone

## Wave 4: Final

12. Coordinator scratchpad
13. `/librarian-update`
14. `/ego-check`

## Execution
Invoke: `/niche .claude/plans/2026-03-13-valuate-niche-implementation.md`
