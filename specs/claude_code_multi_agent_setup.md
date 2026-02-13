# Claude Code Multi-Agent Setup: Hooks, Rules, Skills, and Settings

## Status: Implemented

## Context

UrbanRepML uses a 12-agent stigmergic coordination system where specialist agents communicate across sessions through scratchpad entries in `.claude/scratchpad/{agent}/YYYY-MM-DD.md`. The ego agent's Feb 7 and Feb 8 assessments identified compounding process failures:

1. **Agents not writing scratchpads.** On Day 1 (Feb 7), the main agent did specialist work directly instead of delegating, producing no scratchpad trail. Seven agent directories existed but were empty. Cross-agent observations -- the field that enables stigmergic feedback -- had zero data points across all scratchpads.

2. **Scratchpad bloat.** By Feb 8, the librarian wrote 5 separate entries (300+ lines) for one day. The stage3-analyst had duplicate evening sections. The ego itself was 290+ lines with three separate assessments and contradictions. When scratchpads become walls of text, they stop functioning as coordination mechanisms.

3. **No automated session orientation.** Every new session started cold. The coordinator had to manually read scratchpads, check git log, and recall context. This wasted the first 10-15 minutes of every session and was error-prone (the ego cited stale "67 dims" data that the librarian had already corrected to 64).

4. **Conventions enforced only by CLAUDE.md.** SRAI-first rules, data-code separation, region_id contracts, and the scratchpad protocol all lived in CLAUDE.md. When that file was 236 lines, agents frequently missed or forgot specific rules. No mechanism existed to surface relevant rules at the right time.

The solution: wire Claude Code's hooks, rules, and skills features to make the multi-agent workflow self-enforcing rather than relying on agents remembering to follow CLAUDE.md.

## What Changed

Five phases restructured the Claude Code configuration layer:

### Phase 1: Hooks (automated context injection and enforcement)

Two command-type hooks inject context, two prompt-type hooks enforce scratchpad discipline.

**`.claude/hooks/session-start.py`** -- SessionStart hook, matcher: `startup` only. On fresh session start (not resume or compact), reads and injects: (a) most recent coordinator scratchpad (up to 25 lines), (b) most recent ego assessment (up to 20 lines), (c) `git log --oneline -5`. This eliminates cold-start orientation. The coordinator opens every session already knowing what happened yesterday, what the ego recommends, and what was committed.

**`.claude/hooks/subagent-context.py`** -- SubagentStart hook, no matcher (fires for all subagents). Reads stdin JSON to get `agent_type`, then injects: (a) today's date, (b) the agent's scratchpad path, (c) mandatory scratchpad protocol reminder with the three required sections, (d) last 3 lines of the ego's latest assessment, (e) last 3 lines of the coordinator's latest entry. Every specialist agent spawns with the protocol baked into its context window -- it cannot claim ignorance of the scratchpad requirement.

**SubagentStop prompt hook** -- Fires when any subagent completes. Checks whether the subagent wrote to `.claude/scratchpad/{agent_type}/YYYY-MM-DD.md` with the three required sections (What I did, Cross-agent observations, Unresolved). If not, responds with `decision: block` and a reason explaining the agent must write its scratchpad. This is the enforcement gate: agents cannot complete without logging.

**Stop prompt hook** -- Fires when the main session ends. Checks whether coordinator mode was active (the `/coordinate` skill was used or multi-agent delegation occurred) and, if so, whether a coordinator scratchpad exists for today. Blocks if coordinator mode was used but no scratchpad was written. Does not block non-coordinator sessions.

### Phase 2: Path-Scoped Rules

Four rule files in `.claude/rules/`, each with a YAML frontmatter `paths:` declaration that controls when they are auto-loaded into context.

**`srai-spatial.md`** -- Paths: `stage1_modalities/**`, `stage2_fusion/**`, `stage3_analysis/**`, `scripts/processing_modalities/**`, `scripts/accessibility/**`. Contains the SRAI-first policy: which h3-py functions are allowed (hierarchy traversal only), which are banned (tessellation, neighborhoods), correct import patterns, and a 4-item audit checklist for code reviews. Replaces the "SRAI-First" section previously in CLAUDE.md.

**`multi-agent-protocol.md`** -- Paths: `.claude/**`. Contains scratchpad format (three mandatory sections), scratchpad discipline (consolidate, do not append-only, 80-line max), coordination architecture (main agent IS coordinator, OODA, Task tool delegation), and cross-agent communication patterns. Replaces the "Multi-Agent Workflow" section previously in CLAUDE.md.

**`data-code-separation.md`** -- Paths: `stage1_modalities/**`, `stage2_fusion/**`, `stage3_analysis/**`, `scripts/**`. Contains the absolute data-code boundary: code directories vs `data/` (gitignored), study-area directory layout, and rules (every script needs `--study-area`, no hardcoded paths). Replaces the "Study Area Organization" section previously in CLAUDE.md.

**`index-contracts.md`** -- Paths: `stage1_modalities/**`, `stage2_fusion/**`, `stage3_analysis/**`, `scripts/**`. Contains the `region_id` standard, the stage boundary convention (Stage 1 outputs `h3_index` for backwards compatibility, Stage 2+ uses `region_id`, `MultiModalLoader` bridges), and rules for writing new code. Replaces the "SRAI region_id Index Standard" section previously in CLAUDE.md.

### Phase 3: Skills

Three new skills plus the existing `/coordinate` skill.

**`/summarize-scratchpads`** -- Context: fork. Agent: Explore. Tools: Read, Glob, Grep. Compresses today's and yesterday's scratchpad entries across all agents into a 20-30 line structured summary with sections for In Progress, Blocked, Tensions, Unresolved, and Notable. Flags agents that should have written scratchpads but did not (based on recent git activity) and notes scratchpad bloat (entries over 100 lines). Use at session start or mid-session when context is unclear.

**`/ego-check`** -- Context: fork. Agent: ego. `disable-model-invocation: true`. Full ego process health assessment following the ego's standard protocol: read all agent scratchpads, check recent git diffs, compare specs vs implementation, look for agent output patterns (confusion, failures, conflicting edits). Writes to `.claude/scratchpad/ego/YYYY-MM-DD.md` with Working/Strained/Forward-Look sections. Also seeds the next session by writing the forward-look to the coordinator's next-day scratchpad. Use at session end.

**`/librarian-update`** -- Context: fork. Agent: librarian. `disable-model-invocation: true`. Reads `git diff HEAD~3..HEAD --stat`, examines changed files, and updates `.claude/scratchpad/librarian/codebase_graph.md` with new modules, changed interfaces, removed components, and updated dependencies. Writes a daily scratchpad entry documenting what changed. Use after significant code changes.

### Phase 4: Settings Files

**`.claude/settings.json`** (project-level, committed to git) -- Contains all hook configurations (SessionStart, SubagentStart, SubagentStop, Stop) and shared permissions: `Bash(*)`, `Skill(coordinate)`, `Skill(summarize-scratchpads)`, `Skill(ego-check)`, `Skill(librarian-update)`, with `defaultMode: acceptEdits`. This is the canonical hook registry.

**`.claude/settings.local.json`** (gitignored, per-developer) -- Contains only personal preferences that should not be shared: WebFetch domain allowlist (github.com, docs.claude.com, kraina-ai.github.io, etc.), WebSearch permission, and Google Drive read access. Separated from `settings.json` so that domain preferences do not pollute the shared config.

### Phase 5: CLAUDE.md Slimming

CLAUDE.md was reduced from 236 to 161 lines by extracting detailed convention sections into rules files. Removed:

- SRAI-First section (now `srai-spatial.md` rule, auto-loaded when touching stage code)
- Multi-Agent Workflow details (now `multi-agent-protocol.md` rule, auto-loaded when touching `.claude/`)
- Study Area Organization details (now `data-code-separation.md` rule, auto-loaded when touching stage/script code)
- SRAI region_id Index Standard and Stage Boundary Convention (now `index-contracts.md` rule)
- TIFF Processing Architecture (now in `data-code-separation.md` rule)

Kept in CLAUDE.md: core principles (with cross-references to the relevant rule files), three-stage architecture overview, setup instructions, key commands, cone-based training details, archived techniques reference, common pitfalls, and essential resources. The Multi-Agent Workflow section was replaced with a 5-line pointer to the rules, hooks, skills, and this spec.

## Hook Lifecycle

The four hooks form a lifecycle that brackets every session and every agent invocation:

```
Session Start
    |
    v
[SessionStart hook fires]
    - Reads coordinator's latest scratchpad (25 lines max)
    - Reads ego's latest assessment (20 lines max)
    - Reads git log --oneline -5
    - Injects all as additionalContext
    |
    v
Coordinator works, invokes /coordinate, delegates to specialists
    |
    +---> For each specialist agent:
    |        |
    |        v
    |    [SubagentStart hook fires]
    |        - Reads stdin JSON for agent_type
    |        - Injects: date, scratchpad path, protocol reminder
    |        - Injects: ego tail (3 lines), coordinator tail (3 lines)
    |        |
    |        v
    |    Agent does work
    |        |
    |        v
    |    [SubagentStop hook fires]
    |        - Prompt checks: did agent write scratchpad today?
    |        - Checks for 3 required sections
    |        - BLOCKS if missing -> agent must write scratchpad
    |        - ALLOWS if present -> agent completes
    |        |
    |        v
    |    Agent result returned to coordinator
    |
    v
Session ending
    |
    v
[Stop hook fires]
    - Prompt checks: was /coordinate used this session?
    - If yes: was coordinator scratchpad written?
    - BLOCKS if coordinator mode active but no scratchpad
    - ALLOWS otherwise
    |
    v
Session ends
```

Key property: the SubagentStop hook creates a hard gate. An agent that forgets to write its scratchpad will be blocked and told to write it before completing. This directly addresses the ego's Day 1 observation that agents existed in name but did not communicate through the scratchpad medium.

## Rules Auto-Loading

Claude Code's rules system uses path-scoped YAML frontmatter to determine which rules are loaded into context. A rule is loaded when the agent is working on files matching any of its declared paths.

| Rule | Path Triggers | Loaded When... |
|------|--------------|----------------|
| `srai-spatial.md` | `stage1_modalities/**`, `stage2_fusion/**`, `stage3_analysis/**`, `scripts/processing_modalities/**`, `scripts/accessibility/**` | Editing any stage code or spatial scripts |
| `multi-agent-protocol.md` | `.claude/**` | Editing agent definitions, hooks, skills, or scratchpads |
| `data-code-separation.md` | `stage1_modalities/**`, `stage2_fusion/**`, `stage3_analysis/**`, `scripts/**` | Editing any code directory |
| `index-contracts.md` | `stage1_modalities/**`, `stage2_fusion/**`, `stage3_analysis/**`, `scripts/**` | Editing any code directory |

This replaces the CLAUDE.md approach (everything in one file, always loaded, mostly ignored) with targeted injection: the SRAI rules appear only when touching spatial code, the scratchpad protocol appears only when touching agent infrastructure. Rules are surfaced when they are relevant, not when they are not.

## Skill Invocation Patterns

| Skill | When to Use | Who Typically Invokes |
|-------|------------|----------------------|
| `/coordinate [task]` | Starting a multi-agent work session. The main agent enters OODA mode, reads scratchpads, delegates to specialists. | User or main agent at session start |
| `/summarize-scratchpads` | Quick orientation on cross-agent state. Produces a 20-30 line compressed summary. Useful when the SessionStart hook context is not enough, or mid-session to check progress. | Coordinator, or user directly |
| `/ego-check` | End of a productive session. Produces a Working/Strained/Forward-Look assessment and seeds tomorrow's coordinator scratchpad. | Coordinator at session end, or user |
| `/librarian-update` | After significant code changes (model renames, new modules, deleted files, changed interfaces). Keeps the codebase graph current. | Coordinator after code changes |

The `/ego-check` and `/librarian-update` skills use `disable-model-invocation: true`, meaning they run the full agent protocol (ego or librarian) with tool access but without additional model context injection. The `/summarize-scratchpads` skill uses the `Explore` agent type with limited tools (Read, Glob, Grep only) to prevent it from modifying anything.

## Agent Communication Flow

Before the restructure, cross-agent communication depended entirely on voluntary compliance with CLAUDE.md instructions. The restructure makes it structural:

```
                    SCRATCHPAD PROTOCOL (enforced)
                    ==============================

    SubagentStart hook               SubagentStop hook
    injects protocol  ------>  Agent works  ------> checks scratchpad
    + ego/coord tail                                 blocks if missing
         |                                                |
         v                                                v
    Agent reads:                                  Agent writes:
    - ego tail (3 lines)                          - What I did
    - coord tail (3 lines)                        - Cross-agent observations
    - its own prior scratchpad                    - Unresolved
         |                                                |
         v                                                v
    Minimal orientation                           Entry in scratchpad/
    without reading                               {agent}/YYYY-MM-DD.md
    entire scratchpads
```

The SubagentStart hook solves the "agents not reading each other" problem by injecting the ego's and coordinator's latest conclusions directly into every agent's context. Agents do not need to proactively seek this information -- it is delivered to them.

The SubagentStop hook solves the "agents not writing scratchpads" problem by making completion conditional on the scratchpad existing with the correct sections.

The 80-line scratchpad limit (in `multi-agent-protocol.md`) and the "consolidate, don't append" instruction (injected by the SubagentStart hook) address bloat.

## File Manifest

| # | File | Type | Purpose |
|---|------|------|---------|
| 1 | `.claude/hooks/session-start.py` | Hook (command) | SessionStart: inject coordinator + ego context + git log |
| 2 | `.claude/hooks/subagent-context.py` | Hook (command) | SubagentStart: inject scratchpad protocol + ego/coord tail |
| 3 | `.claude/settings.json` | Settings (committed) | Hook configs, shared permissions, defaultMode |
| 4 | `.claude/settings.local.json` | Settings (gitignored) | Personal WebFetch domains, WebSearch, Drive access |
| 5 | `.claude/rules/srai-spatial.md` | Rule (path-scoped) | SRAI-first enforcement, h3 audit checklist |
| 6 | `.claude/rules/multi-agent-protocol.md` | Rule (path-scoped) | Scratchpad format, discipline, coordination architecture |
| 7 | `.claude/rules/data-code-separation.md` | Rule (path-scoped) | Data-code boundary, study-area organization |
| 8 | `.claude/rules/index-contracts.md` | Rule (path-scoped) | region_id standard, stage boundary convention |
| 9 | `.claude/skills/summarize-scratchpads/SKILL.md` | Skill | Compress scratchpads into 20-30 line summary |
| 10 | `.claude/skills/ego-check/SKILL.md` | Skill | Ego process health assessment + forward-look |
| 11 | `.claude/skills/librarian-update/SKILL.md` | Skill | Update codebase graph from recent git diffs |
| 12 | `CLAUDE.md` | Modified | Slimmed from 236 to 161 lines, added rule cross-references |

Note: `.claude/skills/coordinate/skill.md` already existed and was updated to reference the new infrastructure, but is not counted as a "new" file.

## Before/After: Coordinator Workflow

### Before (Feb 7-8)

```
1. Session starts cold. No automated context.
2. Coordinator manually reads:
   - .claude/scratchpad/coordinator/latest
   - .claude/scratchpad/ego/latest
   - .claude/scratchpad/librarian/codebase_graph.md
   - git log --oneline -10
   (This takes 4-6 tool calls and 10+ minutes)
3. Coordinator delegates to specialists via Task tool.
4. Specialists may or may not write scratchpads.
   No enforcement -- compliance is voluntary.
5. If specialists skip scratchpads, the ego detects
   the gap... in the NEXT session. No real-time feedback.
6. CLAUDE.md is 236 lines. Agents load the full file
   but frequently miss specific conventions buried in it.
7. Session ends. Coordinator may or may not write scratchpad.
   If not, next session starts even colder.
```

### After (Feb 13+)

```
1. Session starts. SessionStart hook auto-injects:
   - Coordinator's latest scratchpad (25 lines)
   - Ego's latest assessment (20 lines)
   - git log --oneline -5
   Coordinator is oriented in <1 second, zero tool calls.
2. Coordinator invokes /coordinate [task].
3. Coordinator delegates to specialists via Task tool.
4. SubagentStart hook injects into each specialist:
   - Today's date, scratchpad path
   - Mandatory protocol reminder
   - Ego + coordinator tail context
5. Specialist does work. On completion:
   SubagentStop hook checks for scratchpad.
   BLOCKS if missing. Agent must write before completing.
6. Path-scoped rules auto-load relevant conventions:
   - Touching stage code? srai-spatial.md + index-contracts.md
   - Touching .claude/? multi-agent-protocol.md
   - CLAUDE.md is 161 lines of architecture, not convention details.
7. Session ending. Stop hook checks:
   - Was /coordinate used? If yes, coordinator scratchpad required.
   BLOCKS if coordinator scratchpad missing.
8. Optional: /ego-check produces health assessment and seeds
   tomorrow's coordinator scratchpad automatically.
```

## Consequences

### Positive
- Scratchpad compliance is enforced at the hook level, not by voluntary memory. The SubagentStop gate directly addresses the Day 1 failure where agents completed without writing scratchpads.
- Session cold-start is eliminated. The SessionStart hook provides full orientation in under 1 second with zero manual tool calls.
- Convention rules are surfaced contextually via path-scoped loading, not dumped into a 236-line monolith where agents miss them.
- CLAUDE.md is 31% shorter (236 to 161 lines), reducing context window consumption for every agent invocation.
- Skills provide one-command access to recurring coordination tasks (scratchpad summary, ego check, librarian update) that previously required multi-step manual orchestration.

### Negative
- The SubagentStop prompt hook relies on the model correctly evaluating whether the scratchpad file exists and contains the right sections. This is a soft check (the model interprets the prompt), not a hard filesystem check. False positives (allowing without scratchpad) or false negatives (blocking despite scratchpad) are possible.
- The SessionStart hook reads at most 25+20 lines of coordinator and ego context. If the most recent entries are not the most relevant, the hook injects stale context. No mechanism exists to override which scratchpad the hook reads.
- Settings split between `settings.json` (committed) and `settings.local.json` (gitignored) means a developer must know to check both files. The `.local` file can silently override or extend the shared config.
- Rule path matching is glob-based. If a file is outside all declared paths, no rules load for it. Novel directories or one-off scripts in the project root receive no rule injection.

### Neutral
- The hooks are Python scripts that read filesystem state. They add ~100ms to session start and ~50ms to each subagent spawn. This is imperceptible relative to model inference time.
- The four rule files total approximately 120 lines of markdown. The CLAUDE.md reduction was 75 lines. Net context savings depend on how many rules are loaded simultaneously for a given task. When touching stage code, two rules load (srai-spatial + index-contracts/data-code-separation), which is roughly equivalent to the removed CLAUDE.md sections. The benefit is relevance, not raw size.
- The `/ego-check` and `/librarian-update` skills use `disable-model-invocation: true`, which means the agent runs its full protocol using its agent definition file rather than receiving additional instructions from the skill's markdown. This gives the ego and librarian agents maximum autonomy but means the skill file serves primarily as documentation and launch configuration, not as detailed instructions.

## Implementation Notes

### Ordering

The implementation followed a dependency order:

1. **Hooks first** -- these are the enforcement backbone. Without SubagentStop blocking, everything downstream is advisory.
2. **Settings.json** -- registers the hooks. Hooks do not fire without configuration in settings.json.
3. **Rules** -- extract convention content from CLAUDE.md. Must happen after hooks are in place so that the slimmed CLAUDE.md can reference the rule files.
4. **Skills** -- convenience wrappers. These work with or without hooks/rules, but their value is maximized when hooks ensure the scratchpads they summarize actually exist.
5. **CLAUDE.md slimming** -- must happen last, after all extracted content is in its new home.

### Dependencies

- Hooks depend on the scratchpad directory structure (`.claude/scratchpad/{agent}/`) existing. This was created during the multi-agent bootstrap on Feb 7.
- The SessionStart hook depends on Python 3.10+ (uses `Path | None` type hint syntax).
- The SubagentStart hook depends on Claude Code passing `agent_type` in the stdin JSON payload for SubagentStart events.
- Rules depend on Claude Code's path-scoped rule loading feature being available in the installed version.

### Testing

The hook scripts can be tested standalone:

```bash
# Test session-start hook
echo '{"source": "startup"}' | python .claude/hooks/session-start.py

# Test subagent-context hook
echo '{"agent_type": "librarian"}' | python .claude/hooks/subagent-context.py
```

The SubagentStop and Stop hooks are prompt-type (not command-type), so they cannot be tested outside a Claude Code session. Their effectiveness will be validated by checking whether future scratchpad entries exist after agent invocations.
