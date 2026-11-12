# Git rules: branches and commits

Git is insurance: the history should show what changed and why, and any behavior change
should be revertible precisely, without touching its neighbors.

## Branches: Git Flow

The branch model is Git Flow. Which model the project uses (full Git Flow or simplified,
see below) is recorded in `05-DECISIONS.md`.

```
main      ──●────────────────●──────────●───────   released only, every commit is a vX.Y.Z tag
             \              / \        /
hotfix/*      \            /   ●──●───●            urgent production fix: from main → into main and develop
               \          /
release/*       \    ●──●                          release preparation: from develop → into main and develop
                 \  /
develop   ──●─────●──────●──────────●──────────    integration: everything ready for the next release
             \          /
feature/*     ●──●──●──●                           one block's work: from develop → into develop
```

| Branch | From | Merges into | What's in it |
|---|---|---|---|
| `main` | — | — | What runs in production. No direct commits. Every merge gets a version tag |
| `develop` | `main` (once) | — | What's ready for the next release. Tests always green |
| `feature/BXX-<what>` | `develop` | `develop` | The work of one roadmap block. One branch, one block |
| `release/X.Y.0` | `develop` | `main` + `develop` | Release preparation: only bug fixes, version, documentation. No new features |
| `hotfix/X.Y.Z-<what>` | `main` | `main` + `develop` | Urgent production fix, without waiting for a release |

**Rules:**
- Merges use `git merge --no-ff`: the history keeps showing which commits were one branch.
- Into `main` and `develop` only by merging a branch with green CI. The owner merges into `main`
  (or it's done with their explicit consent).
- Versions are semantic `vMAJOR.MINOR.PATCH`: MAJOR for incompatible changes (formats,
  API, database without rollback), MINOR for new features (`release/*`), PATCH for fixes (`hotfix/*`).
  The tag goes on the merge commit in `main`: `git tag -a v1.4.0 -m "<what's in the release>"`.
- `hotfix/*` must also be merged into `develop`, otherwise the fix gets lost in the next release.
- A branch handed over for production verification (`release/*`) is frozen: nothing new goes in, so
  that if something breaks it's clear what's to blame.
- A merged branch is deleted (`git branch -d`); history remains in the merge commits.
- One block's work only in its own `feature/*`; don't spread it across other branches.
- A long-lived `feature/*` regularly pulls in `develop` (merge, or rebase before the branch is published),
  so conflicts don't pile up for weeks.
- Current branches and what's in each go in `docs/04-STATE.md`, section "Where we stopped".
- From the very start, enable protection for `main` and `develop` (GitHub → Settings → Branches →
  "Require status checks to pass"): merging only with green CI.

**Simplified variant for small projects** (one developer, no versioned releases):
`main` + short `feature/*` / `fix/*` branches from `main`, merged back through green CI
(GitHub Flow). Switch to full Git Flow when production users appear and you need to
ship in verified batches. In the simplified variant, wherever commands and agents say
`develop`, read `main`; `/release` isn't needed; `/hotfix` is a `fix/*` branch from `main`, merged once into `main`, no tag. The chosen model is recorded in `05-DECISIONS.md`.

## When to commit

- On every `/save`: `wip: <what was done>`.
- On every `/block-done`: `feat(BXX): <what the block delivers>`, then `feature/BXX-*` is merged
  into `develop` (`--no-ff`) with green CI.
- **Before a risky rework**: a separate commit, so there's something to roll back to.
- A finished logical piece is a commit; don't pile a day's work into one.

## What goes into one commit

- **One reason for change, one commit.** Refactoring, behavior fixes, documentation,
  tests for old code go in separate commits.
- **A behavior (bug) fix is always a separate commit** with a clear title: it must
  be possible to find and revert it without touching anything else. A bug found during refactoring
  is not mixed into the refactoring commit.
- **Code and its tests go in one commit.** A commit that changes behavior without tests is not done.
- Renaming a public name happens together with all callers and tests.
- Documentation describing changed behavior is updated in the same commit or the very next one.

## Commit message

Format: `type(area): what changed — why/how`, in the project language (`CLAUDE.md`), in imperative or
descriptive form, no trailing period. A body if the "why" didn't fit in the title.

| Type | When |
|---|---|
| `feat` | new feature, closed block (`feat(B07): …`) |
| `fix` | behavior fix |
| `refactor` | code structure changes, behavior doesn't |
| `test` | tests only |
| `docs` | documentation and comments only |
| `chore` | housekeeping: dependencies, `.gitignore`, CI |
| `wip` | intermediate save (`/save`) |

Examples:
```
fix(orders): don't charge payment twice on a repeated webhook
refactor(config): settings split by topic
docs(risks): audit findings in the risk log, S×L matrix rating
test(reports): day boundary by the client's time zone, not the server's
```

## Who commits

- **Only the main chat commits.** Agents may write files (tests, references, reports), but
  the main chat reviews their work and commits itself.
- Commit and push only when the owner asked or it's a protocol step (`/save`, `/block-done`).

## What's not allowed

- Committing `.env`, keys, passwords, tokens, secret webhook URLs. Before committing: `git status`
  and a look at the file list.
- Committing materials with personal data (conversation screenshots, exports with phone numbers):
  they can't be removed from history without rewriting it.
- Skipping checks (`--no-verify`) and rewriting published history (`push --force`)
  without a direct request from the owner.
- Merging into the production branch without green CI and the owner's consent.

## "It broke, put it back"

This is a rollback to the last commit of a closed block. Give the owner the exact command, for example:
`git log --oneline` (find `feat(BXX)`), then `git revert <commit>` for a precise rollback
of something published, or `git switch -c rollback <commit>` to look at the state without losing work.
