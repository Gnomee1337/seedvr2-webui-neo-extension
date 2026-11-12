---
description: Save the state before clearing the context — STATE and a wip commit
---

We save everything important to files so that after `/clear` we can continue without loss.

1. Update `docs/04-STATE.md` (up to 60 lines): where we stopped — down to the step of the card;
   what was done in this session and is not yet reflected; what is next (specifically); open questions.
2. Into the block card — marks for completed steps, pitfalls and assumptions found.
3. Decisions of this session — into `docs/05-DECISIONS.md`; new terms — into `08-GLOSSARY.md`;
   new risks — into `07-RISKS.md`.
4. Commit `wip: <what was done>` (without `.env` and secrets — check `git status`).
5. Tell the owner: "Saved. Type /clear, then /go".
