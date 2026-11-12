---
description: Move a project started under another system onto our protocol without losing anything
---

The project has already started, but without our docs. The goal is to describe what exists **without changing the code**.

1. **Survey without edits.** Hand it to agents (Explore, in parallel): structure and stack; entry points and
   external services; database and migrations; background jobs and flags; tests and how to run them; what
   looks unfinished or dead. Existing docs (README, notes, TODO) — read them
   and keep what is valuable.
2. **Questions for the owner** — only what cannot be learned from the code: the goal, who it is for, signs of
   being done, what already works for people, what is dangerous to touch.
3. **Docs from the `docs/` templates:** VISION, ARCHITECTURE (from the code), 09-PROCESSES (the processes
   found), RISKS (findings with S×L rating), GLOSSARY, DECISIONS (choices visible
   in the code — marked "reconstructed from code" if the reason is unknown).
4. **Roadmap:** what is done — as ✅ blocks (weight by size), what is not done — as ⬜ blocks; the percentage — via
   `plan-drift-checker`. Old docs replaced by ours — delete them (they remain in git).
5. **Branches:** agree with the owner on the branching model (`docs/process/git-rules.md`) and record it in
   `05-DECISIONS.md`. With full Git Flow — create `develop` from the production branch (in older repositories
   it is often called `master` — record it as such).
6. `04-STATE.md`, `CLAUDE.md`. Show the owner the roadmap and the percentage, wait for agreement.
7. Commit `docs: move project onto the protocol — vision, architecture, roadmap, state`.
   Say: "Type /clear, then /go".
