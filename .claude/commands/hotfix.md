---
description: Urgent fix of a production breakage — outside the block cycle, per Git Flow
argument-hint: "<what broke>"
---

Urgent production breakage: $ARGUMENTS

The current block is put aside: first save its state `/save`-style (a `wip:` commit in its branch).

1. **Understand the scope** — what people see, since when, how many are affected. If the harm
   can be stopped with a kill-switch flag — offer that to the owner **first**, before changing code
   (flags are changed only on the owner's word).
2. **Branch** `hotfix/X.Y.Z-<what>` off `main` (`docs/process/git-rules.md`).
3. **A test that catches the bug**: fails before the fix. Do not start the fix without it.
4. **Minimal fix** — only the cause of the breakage, no refactoring along the way.
   The test passes; the area's tests are green.
5. **Check:** `code-auditor` and `fresh-eyes-reviewer` (only the gist of the breakage and the diff) on the diff; if user data, authorization or
   outbound requests are touched — also `security-auditor`. BLOCKER — fix and re-run.
6. **Docs:** remove the entry from `07-RISKS.md` (if the bug was there) or don't add one; if
   the cause is an unknown fact about an external service, add it to `docs/external-apis/`.
7. **Commit** `fix(<area>): <what was fixed>`. Deploy — **only with the owner's consent**,
   per `docs/server-and-deploy/deploy.md` (if present), with a check afterwards.
8. **Merge** into `main` (tag `vX.Y.Z+1`) **and into `develop`** — otherwise the fix will be lost.
9. Summary for the owner: what happened, the cause, what was done, how we verified it. Return to the block.
