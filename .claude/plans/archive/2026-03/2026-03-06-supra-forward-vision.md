# Supra Precision Weighting: Forward Vision

## Status: Vision (not an implementation plan)

## What exists today

The supra layer is a three-file system. `schema.yaml` defines 6 dimensions across 3 groups (execution, quality, domain) with per-agent relevance scores and 4 named modes. `characteristic_states.yaml` stores the human's current settings: raw dimension values, mode, focus/suppress directives, and a timestamp. `supra_reader.py` applies mode biases at read-time, formats full landscapes for coordinators, and filters dimensions by agent relevance for specialists. The `/attune` skill lets the human adjust weights via shorthand (`/attune speed 5, tests 1`) or a 4-question dialogue. Two hooks inject the result: `session-start.py` gives coordinators the full landscape table; `subagent-context.py` gives each specialist only dimensions where their agent_relevance >= 0.5.

This works. The question is: what could it become?

## Tier 1: Near-horizon (prototypable within current Claude Code affordances)

### 1.1 Temporal weight dynamics (NOT naive decay)

**The wrong version**: Weights drift toward a fixed default (3) when stale. This is tempting to implement but fundamentally misguided. Precision weights do not relax toward a fixed point — they relax toward a *moving attractor trajectory* determined by the system's generative model and history.

**The right version (requires Bayesian mechanics)**: Under the free energy principle, precision π evolves as `dπ/dt = -∂F/∂π` where F is the variational free energy. The equilibrium π*(t) is not constant — it depends on current observations, the generative model, and the history of prediction errors. When the generative model shifts (new task context), the entire attractor shifts.

**What this means concretely**: Instead of `speed=5` decaying to `speed=3` after 3 days, the system should infer what `speed` *should* be based on: (a) the task type the human is about to do, (b) the outcome history for that task type, (c) the time-of-day/week pattern. The "default" is not 3 — it's the predicted optimal precision for the current context.

**What blocks it**: This requires the outcome-linked learning store (1.2) as prerequisite data, plus a generative model that predicts optimal weights from context. The mathematical framework exists (Bayesian mechanics — see references below), but the data collection and inference machinery does not. Naive decay (toward a fixed mean) would be worse than no decay because it would actively fight the human's last attunement with no information about what would be better.

**Key references** (Bayesian mechanics formalism):
- Ramstead et al., "On Bayesian Mechanics: A Physics of and by Beliefs" ([arXiv:2205.11543](https://arxiv.org/abs/2205.11543)) — foundational review, defines precision on statistical manifolds
- Da Costa et al., "Bayesian Mechanics for Stationary Processes" ([arXiv:2106.13830](https://arxiv.org/abs/2106.13830)) — proves systems can track *moving* external states through internal inference trajectories
- Friston et al., "Path integrals, particular kinds, and strange things" ([arXiv:2210.12761](https://arxiv.org/abs/2210.12761)) — path integral formulation; precision weights emerge at particle interfaces
- "Bayesian Mechanics of Synaptic Learning" ([arXiv:2410.02972](https://arxiv.org/abs/2410.02972)) — precision as dynamically evolving synaptic gain, not fixed parameter

**The key mathematical insight**: In generalized coordinates of motion, belief trajectories include position, velocity, acceleration. Precision weights apply separately to each order of derivative. The "moving mean" for level i is the output of level (i+1)'s precision-weighted inference. This is recursive — there is no fixed bottom.

### 1.2 Outcome-linked weight profiles

**Capability**: After each session, the ego correlates its quality assessment (Working/Strained/Forward-Look) with the supra weights that were active. Over 20+ sessions, patterns emerge: "sessions with explore>=4 and quality>=4 produce the best Forward-Look scores" or "sprint mode correlates with 2x more items in Strained." This becomes a recommendation engine for `/attune`.

**Why it matters**: The human currently tunes by intuition. Outcome data would let the system say: "For tasks like hex2vec training, your best sessions used focused mode with model_architecture=5."

**What blocks it**: No persistent session-outcome store exists. Ego writes scratchpads but does not score sessions numerically. The ego-check skill would need to emit a structured outcome record (JSON sidecar next to the scratchpad) with fields like `{session_quality: 0-5, task_type: "training|analysis|infra", supra_snapshot: {...}}`. The `recommend_dimensions()` function could then query this store.

### 1.3 Compound states (named weight vectors) — IMPLEMENTED

**Status**: Implemented 2026-03-06 (commit pending). Six compound states defined in `schema.yaml` under `compound_states:`, detected automatically by `supra_reader.detect_compound_state()` with ±1 tolerance, shown in coordinator landscape. Applied via `/attune creative-prototyping` etc.

**What remains**: Letting the human define new compound states during attunement (meta-configuration). Currently the 6 states are hardcoded in schema. A `/attune define:name` syntax could capture the current state as a new compound state — similar to profiles but with semantic description and tension warnings.

### 1.4 Dimensional coupling awareness — PARTIALLY IMPLEMENTED

**Status**: Tension warnings are implemented as part of compound states. `careful-exploration` warns that explore and quality fight each other. But this is per-compound-state, not a general coupling matrix.

**What remains**: A standalone coupling detection that works regardless of compound state matching. A `couplings:` key in schema.yaml mapping dimension pairs to interaction descriptions. `detect_couplings(effective_dims, schema)` would check all pairs and return active interactions. This would catch novel combinations the human creates that don't match any predefined compound state.

## Tier 2: Mid-horizon (needs Claude Code platform evolution)

### 2.1 Mid-session weight adjustment

**Capability**: A background process monitors agent behavior during a session. If QAQC finds 5+ failures, it proposes raising `code_quality`. If the coordinator skips 3 planned waves, it proposes raising `execution_speed`. The proposal appears as a coordinator message: "Supra suggests: raise code_quality to 5 based on QAQC findings. `/attune quality 5` to accept."

**Why it matters**: Current weights are session-static. But sessions evolve -- a session that starts exploratory may discover a critical bug and need to shift to quality-focused. The human should not have to anticipate this in advance.

**What blocks it**: Claude Code has no background process capability. Hooks fire at lifecycle boundaries (session start, agent spawn/stop), not continuously. A workaround: the SubagentStop hook could check agent output patterns and append supra adjustment suggestions to the coordinator's next context injection. This would create a quasi-continuous signal -- one update per agent completion rather than real-time, but possibly sufficient.

### 2.2 Persistent named profiles

**Capability**: The human defines and switches between saved weight profiles. `/attune save:training` captures current state as "training". `/attune load:training` restores it. Profiles persist across sessions in a `profiles/` subdirectory of `supra/`.

**Why it matters**: The human repeats similar task types (training runs, EDA sessions, infrastructure sprints). Re-attuning each time is friction. Profiles eliminate this -- the human says `/attune training` and the entire cognitive hierarchy reconfigures for that workload.

**What blocks it**: Nothing fundamental. Could be a YAML file per profile in `.claude/supra/profiles/`. The `/attune` skill would need `save:` and `load:` syntax. This is probably the lowest-effort Tier 2 item -- it could arguably be Tier 1.

### 2.3 Cross-session supra coordination

**Capability**: When multiple coordinator sessions run concurrently, the supra layer coordinates resource allocation across them. Terminal A doing GPU-heavy training gets `speed=5, quality=2` (let it rip). Terminal B doing code review gets `quality=5, speed=2` (be thorough). The supra layer knows both sessions exist and prevents both from claiming GPU resources simultaneously.

**Why it matters**: The lateral coordinator protocol already handles file-level claim conflicts. But attention and compute are also contested resources. The human is the current bottleneck for cross-session attention allocation -- supra coordination could partially automate this.

**What blocks it**: `characteristic_states.yaml` is a single global file. Multiple sessions reading it see the same weights. Per-session supra state would require either per-session state files (keyed by session_id) or a mechanism for the human to scope `/attune` to a specific session. The coordinator registry already has session IDs; linking supra state to sessions is architecturally straightforward but adds significant complexity.

## Tier 3: Far-horizon (requires new AI capabilities)

### 3.1 Predictive attention

**Capability**: The system predicts what the human will care about next. Monday mornings tend to start with `/attune exploratory` -- the system pre-loads that state. After a commit streak, the human usually runs ego-check -- the system queues it. After 3 sessions on model architecture, the human typically pivots to EDA -- the system suggests the transition.

**Why it matters**: The best interface is one you do not need to use. If the system can predict attunement with 80% accuracy, the human only needs to correct the 20% -- a 5x reduction in cognitive overhead.

**What blocks it**: No temporal pattern model exists. This requires: (a) a session history log with timestamps, task types, and supra snapshots; (b) a pattern matcher (even simple: "time-of-day + day-of-week + recent-task-type -> likely-next-state"); (c) a suggestion mechanism that does not overfit (the system should suggest, never auto-apply). The data collection (a) could start now; the model (b) needs 50+ sessions of history; the UX (c) needs careful design to avoid the "Clippy problem."

### 3.2 Implicit precision weighting

**Capability**: Instead of explicit `/attune`, the system infers the human's attention from behavior. Fast typing + many file opens = exploring. Slow, careful edits to one file = production hardening. Long pauses between commands = deep thinking (raise model_architecture). The supra layer reads the human's work rhythm and adjusts weights accordingly.

**Why it matters**: This is the endgame of the supra concept. The human's attention IS the weight vector -- the system just needs to read it rather than asking.

**What blocks it**: Claude Code has no access to IDE telemetry (keystrokes, file open patterns, pause durations). Even with access, inferring intent from behavior is an unsolved problem -- typing fast could mean confidence or could mean frustration. This requires both platform capabilities (telemetry API) and modeling capabilities (intent inference from behavioral signals) that do not exist. It also raises privacy questions: should the system watch how you type?

### 3.3 Multi-human supra

**Capability**: Multiple humans on a project, each with their own precision profile. Alice cares about spatial correctness; Bob cares about model architecture; Carol cares about test coverage. The system reconciles their profiles when agents work on shared code: a file Alice and Bob both touch gets `spatial_correctness=max(Alice, Bob)` and `model_architecture=max(Alice, Bob)`.

**Why it matters**: Teams have heterogeneous expertise and attention. The supra layer could route work to the human who cares most about it, or raise quality gates when code crosses expertise boundaries.

**What blocks it**: UrbanRepML is a single-human project. Multi-human supra requires: per-user state files, a reconciliation function (max? weighted average? domain-based delegation?), and an ownership model (who is supra for which subsystem?). This is interesting but not relevant until the project has multiple contributors.

## Anti-objectives

The supra system should explicitly NOT become any of the following.

**Not a bureaucracy.** `/attune` should never take more than 30 seconds. If the questionnaire grows beyond 4 questions, something is wrong. The system compresses toward the human -- it should not demand the human's attention to configure it.

**Not autonomous.** The supra layer should never auto-apply weights without human confirmation. Suggestions are fine; auto-application is not. The human's explicit intent is the only valid source of truth for how they want to allocate attention. Even predictive attention (3.1) must propose, not impose.

**Not a performance metric.** Outcome-linked learning (1.2) correlates weights with outcomes to improve recommendations. It must never become a scorecard that judges the human. "Your sprint sessions produce lower quality" is a system observation, not a performance review.

**Not a substitute for communication.** The supra layer enhances human-to-system communication but should not replace direct conversation. If the human says "focus on spatial correctness" in chat, that overrides any supra weight. Natural language is always higher-fidelity than a 1-5 scale.

**Not infinitely dimensional.** The ego's `recommend_dimensions()` can suggest new dimensions. But unbounded growth makes the system unusable. A hard cap of 10 dimensions is wise. If a proposed dimension is better modeled as a focus directive ("focus on hex2vec"), it should stay a focus directive, not become a dimension.

## North Star

The fully-realized supra system is invisible. The human works; the system reads their intent from the texture of their work -- what they linger on, what they skip, what they return to. Agents receive not just instructions but emphasis: "the human cares deeply about this part." The coordinator allocates attention the way a conductor allocates orchestral focus -- not by micromanaging each instrument, but by shaping the ensemble's collective attention toward what matters right now.

The path from here to there is: explicit weights (today) -> compound states + outcome data (Tier 1) -> Bayesian weight dynamics with moving attractors (requires 1.1 + 1.2 as prerequisites) -> predictive suggestion (Tier 3.1) -> implicit reading (Tier 3.2). Each step reduces the human's cognitive overhead while preserving their authority.

The mathematical foundation is Bayesian mechanics (Ramstead et al., 2022; Da Costa et al., 2021). Precision weights are not dials the human turns and the system holds — they are the system's encoding of the human's attentional posture, evolving on a statistical manifold toward trajectories that minimize variational free energy. The "default" is not neutral (3). The default is whatever the system's generative model predicts is optimal for this human, this task, this moment. We cannot build this yet. But we can build toward it by collecting the data (outcome-linked profiles) and formalizing the compound states that represent the manifold's natural coordinates.

The supra layer is not a control system. It is an attention system. The difference matters.
