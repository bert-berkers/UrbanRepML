# Hooks Architecture

## Status: Draft (SubagentStop and Stop hooks pending migration to command type)

## Context

The multi-agent scratchpad protocol requires agents to read context on spawn and write scratchpad entries before completing. Without enforcement, agents skip writing (observed Feb 7-8). Claude Code lifecycle hooks make the protocol structural rather than advisory.

## Four Lifecycle Hooks

```
SessionStart --> SubagentStart --> SubagentStop --> Stop
(session open)   (agent spawns)   (agent exits)   (session close)
```

All four hooks are command-type in the target state. They receive JSON on stdin and emit JSON on stdout. Prompt-type hooks (which ask the LLM to assess compliance) are being replaced because they are non-deterministic -- the model may misjudge whether a file exists or contains required sections.

### 1. SessionStart

- **When**: Fresh session start (source = "startup"). Skips on resume/compact.
- **Script**: `.claude/hooks/session-start.py`
- **Input**: `{"source": "startup"}` on stdin
- **Output**: `{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "..."}}`
- **What it does**: Reads the most recent coordinator scratchpad (25 lines max), ego assessment (20 lines max), and `git log --oneline -5`. Injects all as `additionalContext` so the coordinator opens every session already oriented.
- **Failure mode**: Fail open. If the script errors or times out (10s), the session starts without injected context. No blocking.

### 2. SubagentStart

- **When**: Any subagent spawns. No matcher filter (fires for all agent types).
- **Script**: `.claude/hooks/subagent-context.py`
- **Input**: `{"agent_type": "spec-writer"}` on stdin (agent_type varies)
- **Output**: `{"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext": "..."}}`
- **What it does**: Extracts `agent_type` from stdin. Injects: today's date, the agent's scratchpad path (`.claude/scratchpad/{agent_type}/YYYY-MM-DD.md`), the mandatory three-section protocol reminder, and the last 3 lines each of the ego's and coordinator's latest scratchpad entries.
- **Failure mode**: Fail open. Agent spawns without protocol reminder. This degrades compliance but does not block work.

### 3. SubagentStop (TARGET: command, currently prompt)

- **When**: Any subagent completes.
- **Script (target)**: `.claude/hooks/subagent-stop.py`
- **Input**: `{"agent_type": "spec-writer"}` on stdin
- **What it must verify**: The file `.claude/scratchpad/{agent_type}/YYYY-MM-DD.md` (a) exists and (b) contains all three required headings: "What I did", "Cross-agent observations", "Unresolved".
- **Output on pass**: `{}` (empty JSON, allow completion)
- **Output on fail**: `{"decision": "block", "reason": "Scratchpad missing or incomplete. Write to .claude/scratchpad/{agent_type}/YYYY-MM-DD.md with all three required sections before completing."}`
- **Failure mode**: Fail closed. If the scratchpad is missing or incomplete, the agent is blocked and must write it. Script errors should fail open (allow completion) to avoid permanently stuck agents.
- **Why command > prompt**: The current prompt hook asks the LLM "did this agent write its scratchpad?" The LLM must infer filesystem state from conversation context -- it cannot actually check `os.path.exists()`. A command hook runs `Path.exists()` and parses the file for headings. Deterministic, no false positives.

### 4. Stop (TARGET: command, currently prompt)

- **When**: Main session ends.
- **Script (target)**: `.claude/hooks/stop.py`
- **Input**: `{}` on stdin (no agent_type -- this is the main session)
- **What it must verify**: If multi-agent coordination happened this session, a coordinator scratchpad must exist at `.claude/scratchpad/coordinator/YYYY-MM-DD.md`.
- **Sentinel detection**: The script checks for any subagent scratchpad files dated today across all `.claude/scratchpad/*/YYYY-MM-DD.md` directories (excluding `coordinator/` and `ego/`). If any exist, multi-agent delegation occurred and the coordinator scratchpad is required.
- **Output on pass**: `{}` (allow session end)
- **Output on fail**: `{"decision": "block", "reason": "Multi-agent delegation detected but no coordinator scratchpad for today. Write to .claude/scratchpad/coordinator/YYYY-MM-DD.md before ending session."}`
- **Failure mode**: Fail closed for missing coordinator scratchpad when delegation detected. Fail open for non-coordinator sessions (no subagent scratchpads found) and for script errors.

## Scratchpad Enforcement Chain

```
SubagentStart            SubagentStop              Stop
  tells agent              verifies agent          verifies coordinator
  WHERE to write           DID write               wrote (if delegation
  and WHAT format                                  happened)

  Inject:                  Check:                  Detect:
  - scratchpad path        - file exists?          - any subagent scratchpads
  - 3 required sections    - has 3 headings?         for today? (sentinel)
  - ego/coord tail         Block if no.            - coordinator scratchpad
                                                     exists? Block if no.
```

The chain is complete: agents cannot claim they did not know the protocol (SubagentStart injects it), cannot skip writing (SubagentStop blocks them), and the coordinator cannot delegate without logging (Stop blocks if subagent scratchpads exist but coordinator's does not).

## Why Command Over Prompt

| Aspect | Prompt hook | Command hook |
|--------|-------------|--------------|
| Filesystem check | LLM infers from context | `Path.exists()` -- deterministic |
| Content validation | LLM guesses at headings | Regex/string search on file contents |
| Failure consistency | Non-deterministic | Deterministic |
| Timeout | 30s (model inference) | 10s (filesystem I/O only) |
| False negatives | LLM may block despite valid scratchpad | Only if script has a bug |
| False positives | LLM may allow without scratchpad | Only if script has a bug |

The SubagentStop prompt hook was observed to be unreliable because the LLM sometimes concludes "the agent mentioned writing a scratchpad" without verifying the file actually exists. Command hooks eliminate this class of error.

## Configuration

All hooks are registered in `.claude/settings.json` under the `hooks` key. Each hook entry specifies `type: "command"`, the `command` string (Python script path), and a `timeout` in seconds. The SubagentStart and SubagentStop hooks use an empty `matcher` string to fire for all agent types.
