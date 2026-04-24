# Session-Identity Hardening (warmstart, 2026-04-19)

| Field | Value |
|---|---|
| **Status** | PARTIALLY SHIPPED — W3 absorbed into cluster-2 commit [pending]; W1/W2 remain OPEN |
| **Cluster** | governance / hooks-architecture |
| **Trigger** | 2026-04-19 collision: Terminals B and D both wrote under `session_id=calm-glowing-rock` for ~2h before the human spotted it. Root cause: `/valuate` ran as a forked subagent whose PID-walk could not resolve the real terminal. |
| **Root-cause fix already shipped** | commit `d077c25 fix(skills/valuate): run inline like /niche` — removes `context: fork` from valuate's frontmatter so it executes in the coordinator's main context. |
| **Failure-mode log** | `.claude/scratchpad/coordinator/notes.md` §"2026-04-19 — Failure Mode: Identity Tagging Drift" |
| **Est** | 2–3h (W1+W2 are small; W3 is the meaty one) |

**2026-04-24**: W3a additions to `coordinator-coordination.md` absorbed into `.claude/plans/2026-04-18-cluster2-ledger-sidecars.md` W3 wave, shipped at commit [pending]. W1 (runtime identity check in `/niche`) and W2 (subagent guard on identity-bearing writes) still require fresh context with hook-testing focus.

## Why this plan exists

The `/valuate` fix patches the **specific** cause of the 2026-04-19 collision but leaves three categories of identity drift unguarded:

1. No runtime check that the three identity components agree (SessionStart name vs `terminals/{pid}.yaml` vs supra yaml).
2. Other skills could regress to subagent-style execution and reintroduce the same bug.
3. The coordinator-coordination rule file does not name SessionStart as the canonical identity source.

Each W below addresses one. They are independent and parallelisable.

## Three identity components that must agree

1. **SessionStart-injected `session_id`** (e.g. `willowy-leaning-maple`) — produced by the SessionStart hook. **Canonical.**
2. **Terminal yaml** `.claude/coordinators/terminals/{shell-pid}.yaml` — links shell PID to session_id + supra_session_id.
3. **Supra yaml** `.claude/supra/sessions/{supra_session_id}.yaml` — carries reference frame + intent.

When any drifts from another, identity confusion is in progress. SessionStart is the source of truth.

## Wave structure

### W1 — Wave-0 identity sanity check in `/niche` (skill edit + helper)

**Owner**: spec-writer for the SKILL.md edit; devops for any `coordinator_registry` helper additions.

Add a new step to `/niche` Wave 0 (after the existing supra read, before HELLO broadcast):

1. Read `session_id_from_session_start` from the SessionStart system-reminder injection (the coordinator already has this in its context — call it out explicitly so it's not forgotten).
2. Read `session_id_from_terminal_yaml` via `coordinator_registry.read_ppid_session()`.
3. Read `supra_session_id_from_supra_yaml` (derive from the loaded supra session file's `supra_session_id` field).
4. Check: `session_id_from_session_start == session_id_from_terminal_yaml` AND `supra_session_id_from_supra_yaml.startswith(session_id_from_session_start + '-')`.
5. If any mismatch:
   - Print a clear `## Identity Drift Detected` banner naming the three observed values.
   - Halt before HELLO broadcast.
   - Suggest: "Run `/valuate` with the SessionStart name explicitly, or move the wrong-id terminal/supra files to `archive/` and re-run /niche."

Acceptance: a deliberate id-mismatch test (rename terminal yaml's session_id to a wrong value, then run /niche) triggers the banner and halts. Test added to `.claude/hooks/test_identity_check.py` (new file) or extends `test_claim_narrowing.py` if simpler.

### W2 — Subagent guard on identity-bearing writes

**Owner**: devops.

Add a guard at the top of `coordinator_registry.write_ppid_session()` and `write_ppid_supra()` that detects subagent context and refuses the write:

```python
def _is_subagent_context() -> bool:
    # SubagentStart hook sets CLAUDE_AGENT_TYPE in the env; main coordinator does not.
    # Belt-and-suspenders: also check for absence of CLAUDE_SESSION_ID env (set in main only).
    return bool(os.environ.get("CLAUDE_AGENT_TYPE")) and not os.environ.get("CLAUDE_CODE_ENTRYPOINT", "").startswith("claude-")
```

If subagent context detected → log warning to stderr, return early without writing. Tests: 4 cases (main allowed, subagent blocked, missing env tolerated, malformed env tolerated). Verify the actual env vars Claude Code sets — if `CLAUDE_AGENT_TYPE` isn't reliably present, fall back to caller-stack inspection (uncommon but viable).

Acceptance: any future skill that regresses to `context: fork` and tries to write identity files gets blocked before the file is written. The `/valuate` fix in `d077c25` is preserved as the primary defense; this is the second line.

### W3 — Codify SessionStart as canonical in `coordinator-coordination.md`

**Owner**: spec-writer.

Edit `.claude/rules/coordinator-coordination.md`:

1. Add a top-level section `## Identity: SessionStart Is Canonical` immediately after the three-scale architecture section.
2. State the three identity components in priority order.
3. Cross-link to the failure-mode log in `coordinator/notes.md`.
4. Cross-link to the `d077c25` fix and W1's sanity check.
5. Add to anti-patterns: "Skills writing identity-bearing files from a forked subagent context. The SessionStart hook is the only legitimate writer of the session-side of the identity; `/valuate` is the only legitimate writer of the supra-side."

Acceptance: any future contributor reading the rule file sees the contract before writing a skill.

### Final Wave

- Coordinator scratchpad with all 7 Markov-completeness items.
- `done` message to peers naming `d077c25` (root-cause fix) + W1/W2/W3 commits.
- Update `.claude/scratchpad/coordinator/notes.md` failure-mode entry with a "Resolved" subsection naming the three landed waves.

## Out of scope

- Renaming `coordinator_registry.read_ppid_session()` to drop the `_ppid` infix (terminology cleanup — separate concern).
- Cross-platform PID-walk improvements (the desktop-app quirk in `memory/project_desktop_app_quirks.md`) — separate concern; today's failure was about **subagent** PID-walk, not OS PID-walk.
- Reworking the `/sync` lateral channel.

## Execution

```
/clear
/valuate focused, quality 5, tests 3, focus "session-identity hardening"
/niche follow plan .claude/plans/2026-04-19-session-identity-hardening.md
```
