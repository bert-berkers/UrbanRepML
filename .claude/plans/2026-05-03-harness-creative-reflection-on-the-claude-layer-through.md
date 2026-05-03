# Harness-creative Reflection on the .claude/ Layer — 2026-05-03

| Field | Value |
|---|---|
| **Status** | SHIPPED 2026-05-03 — 3-edit Levin reframe bundle landed (Markov item-7-as-lede, scratchpad-line-limit→salience-density, identity-as-record-vs-prompt spec section) + post-bundle hook surgery (PID-tree-walk drift FIXED in-session per user request: marker list extended `("node", "node.exe", "claude", "claude.exe")`, identity now resolves stably). Zero open carries. |
| **Shard** | `pale-listening-dew` |
| **Mode** | single-thread (user override of explore=5+creative auto-multi-thread; user said "one thread") |
| **Intent** | Harness-creative reflection on the .claude/ layer through the lens of `memory/feedback_stigmergic_traces_are_generative_model.md` (active inference + Levin 2024 self-improvising memory). Examine which conventions treat traces as faithful records vs. as prompts for future interpretive competencies — and shift the harness toward salience-density-as-prompt where it makes sense. |
| **Est** | 3–4 waves, weekend-evening pace; one-session-shippable. |

## Reference frame (echoed from supra session for cold-start resume)

```
mode=creative speed=3 explore=5 quality=3 tests=1 spatial=2 model=1 urgency=2 data_eng=2
intent="Harness-creative reflection on the .claude/ layer through the lens of feedback_stigmergic_traces_are_generative_model.md (active inference + Levin 2024). Examine which conventions still treat traces as faithful records (Markov regex strictness, scratchpad word-count discipline, forward-look format, /valuate inread, OODA Observe) and which should shift toward Levin's salience-density-as-prompt framing. One thread, weekend-evening pace, creative + high-explore."
focus=['traces-as-prompts, not records', 'salience-density over completeness', 'Markov Contract item 7 (gist paragraph) as the load-bearing summary; items 1-6 as scaffolding']
suppress=['voronoi-toolkit carry sweep', "Book of Netherlands quantitative work (cobalt-drifting-cove's domain)", 'data pipeline / model architecture work']
```

If `/clear` lands you here cold: the supra yaml is `.claude/supra/sessions/pale-listening-dew-2026-05-03.yaml`. The valuate scratchpad is `.claude/scratchpad/valuate/2026-05-03.md`. Read both for full state. The originating memory file is `memory/feedback_stigmergic_traces_are_generative_model.md` — read that FIRST; the rest is downstream.

## Frame — why one thread, not a menu

The auto-generator scaffolded a multi-thread kapstok because of the rule `explore≥4 + creative ⇒ multi-thread`. The user explicitly said "one thread". This kapstok is rewritten in single-thread mode — and that act is itself part of the reflection:

> **A rule that fires on dimension values (explore=5, mode=creative) is treating the supra YAML as a record. The user's "one thread" is a salience-bearing prompt. Honoring the prompt over the rule is exactly the shift this session is about.**

So the kapstok rewrites itself in light of its own subject matter. Single thread, but with sub-angles that the waves traverse. Below: the suspect conventions (the harness's "records-mode" surfaces) are listed once, then the waves visit them.

## The suspect conventions (read these as the lens, not the target list)

1. **Markov-Completeness Contract regex strictness** (`.claude/rules/multi-agent-protocol.md` "Markov-Completeness Contract"; `markov_check.py`). Items 1–6 are regex-checked. Item 7 (the gist paragraph) is the actually-load-bearing one — the only item that subsumes all others in prose. The harness validates the scaffolding (1–6) but not the prompt-quality of 7. This is records-mode at the scale of process: structural correctness ≠ salience-density.
2. **Scratchpad ≤30-line guideline** (same file, "Scratchpad Discipline"). The override mechanism `<!-- OVERRIDE: ... -->` already concedes the point: "Length is not the cost; loss-of-state is." But the default still privileges word-count over salience-density. A 12-line entry that prompts well > a 30-line entry that records dutifully.
3. **Forward-look conventions** (e.g., today's `coordinator/2026-05-04.md`): "Recommended Focus" + "Unresolved Tensions" + numbered priorities. This is high-fidelity record-keeping. It works — yesterday's session executed cleanly off it. But Levin's frame asks: would a single-paragraph "what would matter to whoever wakes up next" prompt better, with the structured list as appendix?
4. **`/valuate` morning inread** (`.claude/skills/valuate/SKILL.md` Step 3.5). Lists 3–5 items: forward-look, ego, git, messages, scratchpads-by-name. This is a reading list — a record of where to look. It assumes future-me has the patience to traverse. The Levin move: synthesize a prompt ("Yesterday you were doing X; the live tension is Y; one rope to grab is Z") and offer the reading list as drill-down.
5. **OODA Observe phase** (coordinator pattern). The phase name implies witness/record. Levin's frame: Observe is action-conditioned perception — fetch what current Decide will need, not what is "true about the world right now." Most coordinator OODA cycles I've seen do this implicitly; the question is whether making it explicit ("what does Decide need?") changes the framing usefully.
6. **The ledger / sidecar provenance system** (`utils/provenance.py`, `specs/run_provenance.md`, `specs/artifact_provenance.md`). At first glance this is the most records-mode artifact in the system. But provenance for *data artifacts* is genuinely about reproducibility — the world there really IS supposed to be a faithful record. So this one might be exempt: data wants records; process wants prompts. Worth checking that the boundary is drawn correctly.
7. **Terminal-PID identity binding** (`.claude/coordinators/terminals/{pid}.yaml`, `coordinator_registry.read_ppid_*()`). Surfaced live during this session: `/valuate` wrote the supra YAML correctly (intent, dimensions, focus) but never wrote the terminal-binding file, AND each `Bash` invocation returned a different PID (28440 → 33948 → 11848 — the Desktop App Quirk: fresh shell per call, PPID walking breaks). So the indirection `PID → terminal yaml → session_id + supra_session_id` was doubly broken: nobody wrote the file, and even if they had, the key it was indexed under wouldn't be stable. Meanwhile the SessionStart hook *already* injects `pale-listening-dew` (session) + `pale-listening-dew-2026-05-03` (supra) directly into the coordinator's context as a system-reminder. **The records layer is partially redundant where the prompt layer already exists.** This is the most pointed Levin example in the harness: identity-as-record (PID binding) vs identity-as-prompt (poetic name in the context injection). The poetic name is interpretable by *whatever future agent shows up*; the PID binding only works if a stable-PID world cooperates. Levin: "memories are messages between agents separated across time" — but a message is only a message if its addressing carries. Poetic names carry; PIDs don't (in this environment).

## Wave structure

### W0 — Audit + spec-freeze (coordinator-direct, ~20 min)

- Read the originating memory file in full. Done as part of `/valuate`; coordinator should re-read in W0 to fully internalize.
- Read the seven suspect surfaces above, lightly — enough to confirm the framing fits, not deep-dive yet:
  - `.claude/rules/multi-agent-protocol.md` (Markov contract, scratchpad discipline)
  - `.claude/scratchpad/coordinator/2026-05-04.md` (forward-look format)
  - `.claude/skills/valuate/SKILL.md` Step 3.5 (inread) AND Step 5 (note the missing terminal-binding write — surface 7 evidence)
  - `.claude/hooks/coordinator_registry.py` `read_ppid_*()` + `get_terminal_pid()` (surface 7: how identity is supposed to resolve)
  - One representative coordinator scratchpad close-out (Markov 7/7 in the wild) — pick from `scratchpad/coordinator/2026-05-03-*.md`
- **Acceptance**: coordinator can name, in one sentence each, what each suspect surface currently does AND what it would do under the Levin frame.
- **Anti-scope**: do NOT make edits in W0. Reading and orienting only.

### W1 — Categorize: where is records-mode load-bearing vs. residual?

- For each of the 7 suspect surfaces, place it in one of three buckets:
  - **Records-mode is correct** (provenance for data artifacts, schema contracts, reproducibility) — leave alone, write down WHY.
  - **Records-mode is residual** — was useful for an earlier need, now mostly cargo. Candidate for trimming.
  - **Mixed / shift the emphasis** — keep the record but make the prompt more prominent (e.g. Markov item 7 stays, items 1–6 stay, but item 7 becomes the document's lede rather than its tail).
- **Acceptance**: a 7-row table with bucket assignment and one-sentence reasoning per row. Likely outcome: provenance = correct; Markov + scratchpad-line-limit + forward-look + inread = mixed; OODA-Observe = nominal-only-rename; **terminal-PID identity = residual (the records layer is redundant where the prompt layer already works)**. (Pre-write expectation, not pre-decision — let the audit overrule.)
- **Output target**: a section in the W3 deliverable, OR a standalone short report at `reports/2026-05-03-traces-as-prompts-audit.md` if the categorization grows long.

### W2 — One concrete shift (coordinator-direct, ~30 min, EITHER/OR)

Pick ONE of the following based on W1 findings — do not try to do all of them. The choice is: which shift is small enough to ship tonight and visibly enacts the principle?

- **(2a) Markov Contract item-7 lede**: in `.claude/rules/multi-agent-protocol.md`, reorder so the gist paragraph (item 7) is named the *primary* deliverable and items 1–6 are framed as "scaffolding to ensure item 7 has the materials it needs." The hook regex doesn't change; the prose framing does. This communicates to future agents that filling in 1–6 mechanically without a strong 7 is the failure mode.
- **(2b) Scratchpad-line-limit reframe**: replace "≤30 lines" with "salience density: write what a cold-start reader needs to act, and no more — typically 10–25 lines, override permitted with rationale." Removes the false threshold; keeps the discipline.
- **(2c) /valuate inread synthesis**: amend Step 3.5 to produce a 2–3 sentence prompt-style synthesis FIRST, with the reading list below it. The synthesis says "yesterday X; today Y is open; Z is the rope to grab." Reading list becomes drill-down for the curious.
- **(2d) Identity-as-prompt over identity-as-record**: surface 7 fix. Two sub-options here, pick whichever the audit suggests is most surgical:
  - **(2d.i)** Amend `/valuate` SKILL.md Step 5 to write the terminal-binding file `.claude/coordinators/terminals/{pid}.yaml` when it doesn't exist. Closes the immediate bug. Still records-mode, but at least complete records-mode.
  - **(2d.ii)** Add a fallback path to `coordinator_registry.read_ppid_*()`: if no terminal file exists, parse the SessionStart-injected `supra_session_id` from a known location (e.g., a hook-written file at `.claude/coordinators/.last_session_start.yaml` per terminal-tab) and use that. Acknowledges that the prompt layer is the actually-working identity carrier and routes the records layer to fall back to it gracefully. More Levin-aligned: the poetic name *is* the address, the PID is just a (failing) attempt to reify it.
  - **(2d.iii)** Frame-only fix: add a `## Identity in this harness` section to `specs/session-identity-architecture.md` (or wherever) that names the records-mode/prompt-mode duality explicitly, so future maintainers stop reflexively reaching for PID-based fixes. No code change. The cheapest possible enactment.
- **(2e) Wildcard**: a fifth shift discovered during W1.
- **Acceptance**: ONE file edit, < 40 lines changed, with a paragraph in commit/scratchpad explaining why this shift enacts the principle.
- **Anti-scope**: no hook-code changes, no `markov_check.py` regex changes, no schema changes. Prose-and-framing only.

### W3 — Close-out (coordinator-direct, ~15 min)

- Write coordinator scratchpad with Markov 7/7. The gist paragraph (item 7) gets extra care — it should itself demonstrate the principle (a prompt for whoever picks this up next, not a record of what got done).
- Optionally write `reports/2026-05-03-traces-as-prompts-audit.md` if W1 grew enough material — short report, not exhaustive.
- Update this kapstok's Status line to SHIPPED.
- Decide whether to commit (`.claude/`-only, coordinator-direct ok) or stage and let user review at session end.

## Anti-scope (do NOT do these this session)

- ❌ voronoi-toolkit carry sweep — see forward-look at `coordinator/2026-05-04.md`; deferred per `/valuate` suppress directive.
- ❌ Book of Netherlands quantitative work — cobalt-drifting-cove's active domain; do not touch `stage3_analysis/` or `reports/book_of_netherlands/`.
- ❌ Data pipeline / model architecture work — wrong session for this.
- ❌ Bulk hook-code refactors — single prose-and-framing edit only in W2.
- ❌ Multi-file changes in W2 — pick one shift, one file.
- ❌ Re-litigating the Markov contract's existence. The contract works. We're refining its emphasis, not removing it.
- ❌ Treating this kapstok as a contract. It's a seed; deviation with rationale is fine.

## Peer-terminal pointer

Today's supra YAMLs (live terminal bindings not detected from this PID, but valuate scratchpad shows):

- **cobalt-drifting-cove-2026-05-03** — ACTIVE. Domain: `stage3_analysis/`, `reports/book_of_netherlands/`. Multi-score LBM cluster analysis + score-aware clustering loss. **No path overlap with this session**: pale-listening-dew claims `.claude/**` (process layer), cobalt claims data/model layer.
- russet-rolling-brook-2026-05-03 — ENDED. Book of Netherlands shipped (commit 2de38b6).
- muted-sliding-dune (yesterday) — ENDED. Voronoi toolkit shipped.

## If you only read this section

This plan is a kapstok — a structural framework `/niche` hangs work on, written by `/valuate` to crystallize today's characteristic state. **Single thread by user override** (auto-gen wanted multi-thread; user said "one thread" — and honoring the user's salience-bearing prompt over the rule's records-mode dimension match is itself the principle this session is about). The session is meta-philosophical: read seven harness conventions through Levin's "memories are prompts, not records" frame; categorize each as records-correct, records-residual, or mixed; ship ONE small prose-and-framing shift (candidates: Markov item-7-as-lede; scratchpad-line-limit reframe; /valuate inread synthesis; identity-as-prompt fix per surface 7) that visibly enacts the principle. Surface 7 (terminal-PID identity binding) was added live during W0 — surfaced as a real bug AND a clean Levin example: identity-as-record (PID file) is partially redundant where identity-as-prompt (SessionStart-injected poetic name `pale-listening-dew-2026-05-03`) already works. Weekend-evening pace, one-session-shippable, no hook-code or schema changes (small `.claude/`-prose edits permitted). Originating memory: `memory/feedback_stigmergic_traces_are_generative_model.md` — read first.
