# UrbanRepML Specs — Index

**17 specs.** This directory is the architectural decision trail: what we chose, why, what we rejected.

Code documentation lives in docstrings. Day-to-day runtime rules live in `.claude/rules/`. Specs are the *rationale* — the load-bearing "why" that survives successive refactors.

## How to read a status

| Status | Meaning |
|---|---|
| **Active** | Describes current load-bearing behaviour. Code references this spec or a rule derived from it. Safe to rely on. |
| **Implemented** | Was active, now describes realised behaviour. Read for rationale, not for spec-compliance. |
| **Superseded (→ path)** | Replaced by a newer spec or a `.claude/rule`. The pointer at the top of the file names the successor. Kept for historical context. |
| **Draft** | Proposal. May be partially implemented. Check the header blockquote for concrete status. |
| **Historical** | Narrative record of a completed refactor. Low reading value day-to-day. |

Status is determined by: (a) explicit implementation evidence in `CLAUDE.md`, `MEMORY.md`, or `utils/paths.py`-style back-pointers; (b) the spec's own stated status; (c) a dated successor pointer. The `spec-writer` agent curates this index.

---

## Pipeline specs

The three-stage UrbanRepML pipeline (stage1 modality encoders → stage2 fusion → stage3 analysis) plus the accessibility graph that bridges them.

| Spec | Status | Summary |
|---|---|---|
| [`3_stage_pipeline_restructure.md`](./3_stage_pipeline_restructure.md) | Implemented | The `modalities/` → `stage1_modalities/` + `urban_embedding/` → `stage2_fusion/` + new `stage3_analysis/` package split (Feb 2026). |
| [`experiment_paths.md`](./experiment_paths.md) | Active | `StudyAreaPaths` directory conventions; `utils/paths.py` docstring points here. |
| [`accessibility_graph_pipeline.md`](./accessibility_graph_pipeline.md) | Active | Four-step OSM + RUDIFUN accessibility graph (pyosmium → H3 adjacency → gravity weighting → percentile pruning). |
| [`dnn_probe.md`](./dnn_probe.md) | Active | MLP probe architecture for leefbaarometer prediction with spatial block CV. Matches `stage3_analysis/dnn_probe.py`. |
| [`hex2vec-poi-recovery.md`](./hex2vec-poi-recovery.md) | Implemented | Four-wave migration from raw POI counts (687D) to hex2vec spatial embeddings (50D). |
| [`skip-connection-fix.md`](./skip-connection-fix.md) | Implemented | FullAreaUNet collapse fix (cosine coherence loss, removed normalisations, gated skip). |
| [`poi-pipeline-pyosmium-sedona.md`](./poi-pipeline-pyosmium-sedona.md) | Draft (partial) | pyosmium + date-specific PBF extraction with SRAI IntersectionJoiner. Snapshot extractor not yet durable. |
| [`h3_index_vs_region_id.md`](./h3_index_vs_region_id.md) | Draft (decided, pending) | Decision to use `region_id` throughout was made; code-wide migration not executed. |
| [`run_provenance.md`](./run_provenance.md) | Draft | Dated run dirs + `run_info.json` manifest. This is the answer to MEMORY.md P0 #7 (checkpoint versioning). |

## Scripts / tooling specs

| Spec | Status | Summary |
|---|---|---|
| [`script-hygiene.md`](./script-hygiene.md) | Active | Three-tier `scripts/` organisation (durable / one_off / archive), docstring gate, 30-day shelf life for one-offs. Enforced by `.claude/rules/script-discipline.md`. |

## Claude Code infrastructure specs

The `.claude/` multi-agent coordination layer — hooks, rules, skills, session identity, valuate/niche theory.

| Spec | Status | Summary |
|---|---|---|
| [`claude_code_multi_agent_setup.md`](./claude_code_multi_agent_setup.md) | Implemented (v1) | Original architectural overview: hooks + rules + skills + settings. Partial drift — see successors below. Still the best single "how it all fits together" read. |
| [`session-identity-architecture.md`](./session-identity-architecture.md) | Active | Canonical PPID/terminal-PID identity, `/clear` lifecycle, two-layer (supra + coordinator) identity, graph modes. |
| [`temporal-supra-profiles.md`](./temporal-supra-profiles.md) | Active | Valuate/niche theoretical grounding (TNO narrative layers, active inference) + temporal prior EMA store. Largest spec (38 KB); theory-vs-implementation split is a candidate refactor. |
| [`coordinator_to_coordinator.md`](./coordinator_to_coordinator.md) | Superseded (→ `.claude/rules/coordinator-coordination.md`) | Full C2C advisory protocol: claims, messages, heartbeats, Levin principles. The runtime contract moved into the rule; this spec is the design rationale. |
| [`hooks_architecture.md`](./hooks_architecture.md) | Superseded (→ `claude_code_multi_agent_setup.md` §Phase 1) | Narrower retelling of the four lifecycle hooks. Covered in broader context by `claude_code_multi_agent_setup.md`. |
| [`coordinator-hello.md`](./coordinator-hello.md) | Superseded (→ `.claude/skills/niche/SKILL.md` Wave 0) | Proposed the HELLO broadcast step. Step is now codified in the niche skill and coordinator-coordination rule. |
| [`between-wave-pause-redesign.md`](./between-wave-pause-redesign.md) | Draft | Wave Results block between OODA waves. Absorption status into `niche/SKILL.md` is unverified as of 2026-04-18. |

---

## Conventions

**When to write a spec**: an architectural decision with alternatives considered and cross-session consequence. If the question "why did we do it this way?" would otherwise be lost, it deserves a spec. Reports (`reports/`) document results; scratchpads (`.claude/scratchpad/`) are in-session memory; specs are the permanent decision trail.

**Minimum sections**: `# Title`, `## Status: {status}` (line 2–3), `## Context`, `## Decision`, `## Alternatives Considered`, `## Consequences` (Positive / Negative / Neutral), `## Implementation Notes`.

**Superseded means preserved, not deleted**: when a spec is superseded, add a blockquote pointer at the top naming the successor. Never delete the content — the rationale is load-bearing for future sessions asking "why did we choose this shape?".

**Pattern**: *specs are rationale, rules are runtime.* A spec that describes a durable pattern that agents must follow in every session belongs in `.claude/rules/` (where it is auto-loaded by path globs). The original spec stays as the rationale trail behind the rule.

**Maintenance cadence**: `spec-writer` agent audits specs during organisational-flywheel waves. Drafts over 30 days old with no uptake should be either promoted to a plan (`.claude/plans/`) or closed explicitly with a dated note.

**Size guidance**: 3–12 KB typical. Anything over 20 KB is a candidate to split (theory → `memory/` or `deepresearch/`; implementation → the spec).

**When adding a new spec**: (1) create the file with the minimum sections above, (2) add a row to this index in the appropriate group, (3) if it supersedes an existing spec, add the pointer block to that spec and flip its status here. Do not auto-generate this index — curation is the point.
