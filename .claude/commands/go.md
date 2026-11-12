---
description: Start a session — read the state, report, plan the current block
argument-hint: "[block number, if not the one in STATE]"
---

Session start per the protocol `docs/process/protocol.md`. Read only levels L0–L1.

1. `docs/04-STATE.md` is already loaded. Determine the current block (or take `$ARGUMENTS`).
2. Read the block's row in `docs/03-ROADMAP.md` and its card `docs/roadmap-blocks/BXX-*.md`.
   No card — create one from `docs/roadmap-blocks/_TEMPLATE.md` and agree with the owner on the
   acceptance criteria and "Out of scope for this block".
3. Skim the **headings** of `docs/05-DECISIONS.md` and the "Modules" section of `docs/02-ARCHITECTURE.md` —
   read the bodies only if they concern the block.
4. **Report to the owner, at most 8 lines:** where we stopped, which block, what we will do in it,
   what is needed from the owner (access, answers), questions.
5. If the block touches an external API or an unfamiliar library — run `fact-checker` **before the code**
   on the needed facts (first check they are not already in `docs/external-apis/`). Its conclusions go into
   "Hidden assumptions" of the card.
6. Put the step plan into the card. Note which checks will be needed at `/block-done`.
7. Git: make sure work happens in the block branch `feature/BXX-*` off `develop` (`docs/process/git-rules.md`).
8. Suggest a model if needed (protocol §10) and **wait for the command** — do not start coding without a "yes".
