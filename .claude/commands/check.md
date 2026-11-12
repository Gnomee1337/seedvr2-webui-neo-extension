---
description: Mid-block check — tests of the affected area and the code reviewer on the changes
---

Interim check of the current block. Nothing is closed or committed unless needed.

1. Identify the block's changes: `git diff develop...HEAD --stat` and uncommitted work (`git status`).
2. Run the tests of the affected areas (mirrored test folders). If only docs
   and comments changed — no tests needed.
3. Run in parallel on these changes: `code-auditor` (give it `docs/process/code-style.md` and the block card)
   and `fresh-eyes-reviewer` (give it **only** the gist of the task and the diff, without your own explanations).
4. Summary for the owner, 5–10 lines: what is green, what was found (with S×L rating), what we fix now,
   what we postpone. The full report goes into `docs/agent-reports/`.
5. Anything at BLOCKER level — fix right away, then re-run the agent that found it.
