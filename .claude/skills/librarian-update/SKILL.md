---
name: librarian-update
description: "Run the librarian agent to update the codebase graph after recent changes. Reads git diff and updates .claude/scratchpad/librarian/codebase_graph.md."
user-invocable: true
disable-model-invocation: true
context: fork
agent: librarian
argument-hint: "[optional: specific area to focus on]"
---

## Task

Update the codebase graph to reflect recent changes.

$ARGUMENTS

## Protocol

1. Run `git log --oneline -10` to gauge how many commits since last update, then
   `git diff HEAD~N..HEAD --stat` where N covers all unreviewed commits (default to 5 if unsure)
2. For significantly changed files, read them to understand the new structure
3. Read the current `.claude/scratchpad/librarian/codebase_graph.md`
4. Update the codebase graph with:
   - New modules, classes, or functions
   - Changed interfaces or data shapes
   - Removed or renamed components
   - Updated import chains or dependencies

## Required Output

1. Update `.claude/scratchpad/librarian/codebase_graph.md` in place
2. Write a scratchpad entry to `.claude/scratchpad/librarian/YYYY-MM-DD.md` documenting what changed
