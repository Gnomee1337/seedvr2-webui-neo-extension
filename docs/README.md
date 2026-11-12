# Project documentation — map

All documents live in `docs/`. The repository root holds only `README.md` (entry point for a human) and
`CLAUDE.md` (entry point for Claude Code). Documents describe the **current state**; history lives in git.

## Working rules — `process/` (identical across projects, not edited per project)

| File | Contents |
|---|---|
| `process/protocol.md` | Protocol: blocks, context, agents, percentage, decisions, tests, models, how to write documents |
| `process/agents-and-commands.md` | Commands and checker agents: who checks what at which stage |
| `process/code-style.md` | Code rules: what stays unchanged during refactoring, clean code, modules, classes, names, exceptions, security, tests |
| `process/git-rules.md` | Git Flow, what goes into one commit, message format, who commits, rollback |

## Project knowledge — numbered documents (filled in per project)

| File | Contents |
|---|---|
| `01-VISION.md` | What we build, for whom, what is out of scope, signs of completion |
| `02-ARCHITECTURE.md` | WHAT exists: stack, modules, entry points, background jobs, data, flags, dead code |
| `03-ROADMAP.md` | Blocks, weights, statuses, completion percentage |
| `04-STATE.md` | Where we stopped — the single source of truth |
| `05-DECISIONS.md` | WHY we decided this way — only decisions in force, grouped by topic |
| `06-IDEAS.md` | Open ideas; at the bottom — those rejected by the owner |
| `07-RISKS.md` | What can break and bugs found, rated severity × likelihood |
| `08-GLOSSARY.md` | Terms in plain words |
| `09-PROCESSES.md` | HOW each process works: trigger → steps → result → flags |

## Folders

| Folder | Contents | Required? |
|---|---|---|
| `roadmap-blocks/` | Block cards of **open** blocks (`_TEMPLATE.md` is the template). When a block is closed, its card is deleted | yes |
| `external-apis/` | Facts about external code we depend on: `forge-neo.md` (host WebUI API), `seedvr2-upstream.md` (vendored pipeline and local patches); `_TEMPLATE.md` is the template | yes |
| `agent-reports/` | Temporary agent reports; on `/block-done` the conclusions go into documents and the reports are deleted | appears on its own |

⚠️ Do not put materials with personal data into the repository: they cannot be removed from git
history without rewriting history.
