# Protocol commands and agents

Overview: which commands drive the work, and which reviewer agents look at changes at which stage.
The instructions themselves are in `.claude/commands/<command>.md` and `.claude/agents/<agent>.md`; the rules
for processing findings are in `docs/process/protocol.md` §4.

## Project lifecycle

```
/start-project ─→ feasibility-scout (is it feasible at all) ─→ VISION, ROADMAP, STATE
      │
      ▼
┌──────────────── block cycle (one chat session) ─────────────────┐
│ /go ─→ fact-checker (if external API) ─→ work                   │
│                                             │                   │
│      significant edit ─→ fresh-eyes-reviewer ─→ fixes ─→ commit │
│      /check ─→ code-auditor + fresh-eyes-reviewer (on diff)     │
│      /idea  ─→ idea-integrator                                  │
│                                             ▼                   │
│ /block-done ─→ in parallel: code-auditor + fresh-eyes-reviewer  │
│                + plan-drift-checker                             │
│                + conditionally: integration-checker,            │
│                  security-auditor, stress-tester                │
│             ─→ fixes ─→ rerun the finding agent ─→ commit       │
└──────────── /save (if context 80%+) → /clear → /go ─────────────┘
      │
      ▼
/finish ─→ all reviewer agents on the strongest model (pinned ones stay on sonnet) ─→ acceptance against VISION
```

An "inherited" project (code exists, `docs/04-STATE.md` doesn't): `/migrate` instead of `/start-project`.

Outside the block cycle (Git Flow, `docs/process/git-rules.md`):
```
/hotfix  ─→ hotfix/* from main ─→ test for the bug ─→ fix ─→ code-auditor + fresh-eyes-reviewer ─→ main + develop, tag vX.Y.(Z+1)
/release ─→ release/* from develop ─→ tests ─→ integration-checker + plan-drift-checker ─→ main + develop, tag vX.(Y+1).0
```

## Commands

| Command | When | What it does (briefly) |
|---|---|---|
| `/start-project` | new project from scratch | owner interview → `feasibility-scout` → VISION, ARCHITECTURE (stack), ROADMAP with weights, STATE, first decisions |
| `/migrate` | code exists, our documents don't | code analysis without edits → documents from the template → roadmap from what's done and not done → owner confirmation |
| `/go` | start of each session/block | STATE → block card → report of up to 8 lines → `fact-checker` if there's an external API → plan → wait for "yes" |
| `/check` | mid-block, optional | area tests + `code-auditor` and `fresh-eyes-reviewer` on changes since the block started → summary |
| `/idea` | a new idea came up | `idea-integrator` → weight, place, effect on percentage → owner's decision: block / to ideas / rejected |
| `/progress` | asking "how much is left" | `plan-drift-checker` → honest percentage and plan-vs-code discrepancies |
| `/save` | context 80%+ or end of day | STATE updated → `wip:` commit → "type /clear, then /go" |
| `/block-done` | all block steps done | acceptance criteria → all tests → agents in parallel → fixes → documents → percentage → commit → merge into `develop` |
| `/finish` | all blocks ✅ | only on the strongest model: all agents, all VISION requirements, owner checklist → acceptance report |
| `/hotfix` | urgent production breakage | branch from `main` → test that catches the bug → fix → `code-auditor` + `fresh-eyes-reviewer` (+ `security-auditor` if security is affected) → rollout with the owner's consent → merge into `main` and `develop`, PATCH tag |
| `/release` | time to ship what's accumulated in `develop` | `release/*` branch → all tests → `integration-checker` + `plan-drift-checker` → owner checklist → merge into `main` and `develop`, MINOR/MAJOR tag |

## Agents

| Agent | Stage | What it checks | Writes files? | Model |
|---|---|---|---|---|
| `feasibility-scout` | `/start-project` | whether it's realistic: external APIs, access, limitations, pitfalls | no (report) | sonnet |
| `fact-checker` | `/go`, before code with an external API or a new library | facts from official documentation with links | `docs/external-apis/` | sonnet |
| `code-auditor` (code reviewer) | `/check`, `/block-done`, `/hotfix`, `/finish` | the block's changes: logic errors, `docs/process/code-style.md` rules, tests next to code, dead code, duplicates | no (report) | same as chat |
| `fresh-eyes-reviewer` (fresh eyes) | after each significant edit before commit; `/check`, `/block-done`, `/hotfix`, `/finish` | without the author's context: is it clear, hidden assumptions, silently changed behavior (in refactoring: comparing old and new code), what was forgotten to change alongside | no (report) | same as chat |
| `plan-drift-checker` | `/block-done`, `/progress`, `/release`, `/finish` | percentage by formula, roadmap/STATE/card drift from code, acceptance criteria | no (report) | sonnet |
| `integration-checker` | `/block-done` if there's a new module, a changed data format, or the block must run; always on `/release` and `/finish` | modules fit together: calls, formats, migrations, config, startup | no (report) | same as chat |
| `security-auditor` | `/block-done` for user data, authorization, external requests, files, money; `/hotfix` if security is affected; always on `/finish` | secrets, injections, SSRF, permissions, personal data leaks into logs | no (report) | same as chat |
| `stress-tester` | `/block-done` for input, money, external services; always on `/finish` | hostile scenarios: empty, huge, wrong type, someone else's id, repeat, race, external service failure | tests | same as chat |
| `idea-integrator` | `/idea` | where the idea fits, overlaps with the plan, weight, dependencies, risks | no (report) | same as chat |

**General agent rules:**
- Only the main chat edits code; agents review and report. Exceptions are in the "Writes files?" column.
- An agent puts its full report in `docs/agent-reports/<agent>-<subject>.md` (subject: the block, idea, or whatever was reviewed) and returns a summary
  of up to 10 lines. On `/block-done` the conclusions are moved into permanent documents and the reports are deleted.
- Each finding comes with an `R = S × L` rating (`07-RISKS.md`), a location in the code, and a way to reproduce it.
- Several agents go in one message, in parallel.
- After a finding is fixed, rerun the same agent; two rounds without result go to the owner.
- Agents don't commit.
- **Fresh eyes means without the author's context.** `fresh-eyes-reviewer` gets only the task and
  the diff, without a retelling of the work or the main chat's conclusions: otherwise it will see the same things the author did.
  A significant edit (refactoring, a dangerous module, a bug fix, a rewritten document) isn't
  committed until fresh eyes have reviewed it.
