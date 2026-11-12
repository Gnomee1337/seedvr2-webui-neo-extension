---
description: Release what has accumulated in develop — release branch, checks, version tag, merge into main
argument-hint: "[version, e.g. 1.4.0]"
---

Preparing a release per Git Flow (`docs/process/git-rules.md`).

1. **What we release:** closed blocks in `develop` since the last tag (`git log <tag>..develop --oneline`).
   Version — `$ARGUMENTS`, or propose one yourself: MINOR for new features, MAJOR for incompatible
   changes (formats, API, database without rollback).
2. **Branch** `release/X.Y.0` off `develop`. From now on it gets only bug fixes.
3. **Checks:** all tests; `integration-checker` (startup, migrations, config, new flags
   off by default) and `plan-drift-checker` (docs match the code) — in parallel.
4. **Checklist** `docs/owner-checklists/processes.md` (if present) — update the statuses and what the owner
   should check after deployment. New variables go into `.env.example`.
5. **What changes for people** — a plain-language list for the owner (from `feat`/`fix` commit
   messages), what to turn on by hand, which flags.
6. **The owner's consent** to deploy. Deploy — per `docs/server-and-deploy/deploy.md` (if present).
7. **Merge** `release/*` into `main` (`--no-ff`, tag `vX.Y.0`) and back into `develop`; delete the branch.
8. `04-STATE.md`: which version is in production.
