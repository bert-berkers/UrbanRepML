---
name: ego-check
description: "Run the ego agent to produce a process health assessment (coherent/strained/drifting) and forward-look for the next session."
user-invocable: true
disable-model-invocation: true
context: fork
agent: ego
argument-hint: "[optional: specific concern to investigate]"
---

## Task

Perform a full ego process health assessment for today's development session.

$ARGUMENTS

## Protocol

Follow the ego agent's standard protocol (defined in `.claude/agents/ego.md`):

1. **Read ALL agent scratchpads** for today — you have full visibility across cognitive light cones
2. **Read recent git diffs** — `git diff HEAD~5..HEAD --stat` and targeted file diffs
3. **Check specs vs implementation** — compare `specs/` goals with actual changes
4. **Look for agent output patterns** — confusion, repeated failures, conflicting edits

## Required Output

Write to `.claude/scratchpad/ego/YYYY-MM-DD.md` (today's date) with:
- **Working**: what's going well
- **Strained**: friction, confusion, conflict
- **Health Metrics** (lightweight ASI-inspired):
  - *Scratchpad freshness*: which agents wrote today? Which are >3 days stale?
  - *Cross-referencing*: are agents reading each other's scratchpads? (look for mentions)
  - *Recommendation closure*: are previously flagged issues being addressed?
- **Forward-Look**: recommendations for next session

Then write the forward-look to `.claude/scratchpad/coordinator/YYYY-MM-DD+1.md` (tomorrow's date) to seed the next OODA cycle.

## Lateral Coordination Health

Check the coordinator-to-coordinator system for signs of dysfunction:

1. **Stale claims**: Are there `.claude/coordinators/session-*.yaml` files with heartbeats older than 30 minutes? These indicate crashed sessions that weren't cleaned up.
2. **Claim narrowing**: Did coordinators narrow their `claimed_paths` from the initial `["*"]` within their first OODA cycle? Claim squatting degrades the protocol's value.
3. **Message responsiveness**: Are messages in `.claude/coordinators/messages/` being read and acted on? Check if `request`-level messages got a corresponding `done` response.
4. **Signal propagation quality**: Are pipeline signals (BLOCKED, SHAPE_CHANGED, etc.) actually reaching adjacent agents? Check SubagentStart hook output for false positives or missed signals.
5. **Heartbeat regularity**: Are heartbeats updating at each agent completion (SubagentStop hook), or are there gaps suggesting the hook is failing silently?

Report findings under a **Lateral Health** subsection in the scratchpad.

## Supra-Coordinator Communication Quality

Check whether the coordinator is properly serving the human (supra-coordinator):

1. **Escalation calibration**: Did the coordinator escalate the right things? Look for:
   - Under-escalation: making irreversible decisions, changing scope, or overriding priorities without asking the human
   - Over-escalation: asking the human trivial questions that fall within coordinator autonomy (agent selection, prompt wording, scratchpad format)
2. **OODA report quality**: Were reports concise and decision-oriented? Did they include "Needs your call" items when appropriate? Or were they walls of text the human had to parse?
3. **Compression quality**: Did the coordinator synthesize agent results before presenting to the human, or did it dump raw agent output?
4. **Intent fidelity**: Did the coordinator's delegations faithfully reflect the human's stated intent, or did scope drift occur without acknowledgment?

Report findings under a **Supra Communication** subsection in the scratchpad.

## Agent Gap Detection (Self-Assemblage)

In addition to the standard health assessment, check for signs that the agent ecosystem needs to grow:

1. **Unmatched task patterns**: Were there tasks this session that no existing agent was well-suited for? Did the coordinator handle a recurring category of work itself?
2. **Scratchpad pattern mining**: Are there patterns in specialist scratchpads suggesting a missing capability? (e.g., multiple agents doing ad-hoc data validation → suggests a dedicated data-contract agent)
3. **Repeated escalation**: Are agents consistently escalating the same type of decision? This may indicate the autonomy contract is too narrow, or a new specialist is needed.

If a gap appears in 2+ sessions, recommend a new agent type in the scratchpad with:
- **Name**: proposed agent type identifier
- **Domain**: what expertise it would hold
- **Triggers**: when the coordinator should dispatch it
- **Autonomy scope**: what it can decide autonomously vs must escalate

This is how the system grows new organs — not by top-down design, but by detecting functional gaps in lived operation.
