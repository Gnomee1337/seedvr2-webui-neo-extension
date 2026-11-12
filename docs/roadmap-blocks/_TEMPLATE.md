# BXX — <block name>

**Status:** ⬜ not started | 🔄 in progress | ✅ closed
**Weight:** N (from `docs/03-ROADMAP.md`)
**Decisions this block rests on:** <headings from `05-DECISIONS.md`>

## Why this block
One or two sentences: what becomes possible after it and what breaks if it is skipped.

## What we do (steps)
1. …

## Acceptance criteria
Only verifiable items — ones where you can show a command/screen and see the result.
The wording "works correctly" is forbidden.
- [ ] …

## Which checks are needed on `/block-done`
Always: `code-auditor`, `fresh-eyes-reviewer`, `plan-drift-checker`. Additionally (tick what applies):
- [ ] `integration-checker` — new module, data format changed, the block must be runnable
- [ ] `security-auditor` — user data, authorization, external requests, files, money
- [ ] `stress-tester` — user input, money, external services

## Dependencies
What must be ready BEFORE starting (other blocks, access, manual actions by the user).

## Out of scope for this block
An explicit list of what was moved to other blocks — so the scope does not creep.

## Hidden assumptions
What we silently assume (about the external API, data, user behavior). Filled in on `/go`
from the pre-work review and the `fact-checker` report.

## Open questions
