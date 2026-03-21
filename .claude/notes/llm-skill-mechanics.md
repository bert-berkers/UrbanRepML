# Skills Are Context, Not Commands

Skills (e.g. `/valuate`, `/niche`) are not CLI commands with execution boundaries. They are **context injections** — text loaded into the same token stream. There is no call stack, no return address, no dispatch mechanism.

## Implications

- **Chaining works by accident**: `/valuate` can invoke `/niche` not because there's a function call mechanism, but because the valuate skill text biases the model to produce a `/niche` invocation as its next action. The "boundary" between skills is soft — it's just more text.

- **Context engineering > explicit commands**: If the valuate scratchpad is well-written, a terminal can `/valuate` with shorthand and the model will figure out the right `/niche` task from context alone. No explicit task argument needed.

- **The model doesn't know this**: Claude defaults to reasoning about skills as discrete commands (function calls with clear boundaries). It will give wrong advice like "you need to invoke them separately" because it applies software engineering concepts to something that isn't software. The human has to correct this.

- **One prompt can do multiple skills**: Because skills are just context, a single user prompt can trigger valuation AND task execution if the context is shaped right. This is fundamentally different from any shell or CLI.

## Common failure: discussing = doing

The model will discuss setting weights, talk through the values, even write them in a scratchpad — and then never actually write them to the target file (e.g. supra session YAML). It confuses *reasoning about an action* with *performing the action*. This happened across three parallel terminals simultaneously — all discussed weights, none persisted them.

## For skill authors

When writing SKILL.md files, remember the "execution" is just the model reading your text and deciding what to do. Write for context-shaping, not for instruction-following. The skill text is a bias, not a program.
