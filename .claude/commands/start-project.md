---
description: Start a new project from scratch — interview, scouting, docs, roadmap with weights
---

Starting a new project per the protocol `docs/process/protocol.md`.

If the folder already has code but no `docs/04-STATE.md` — stop and suggest `/migrate`.
If the chat is not on a strong model — suggest in one line: "The decisions here are costly, better use `/model opus`".

1. **Interview the owner** — one or two questions at a time, in plain words:
   - what we are building, in one sentence, and what problem it removes;
   - for whom (roles) and what each of them gets;
   - what is definitely NOT included;
   - by which signs the owner will say "done, we deliver" (verifiable);
   - external services, access, constraints on the stack, server, deadlines;
   - what is dangerous: live users, money, personal data.
2. **Scouting:** run the `feasibility-scout` agent with the owner's answers. Its output — is it feasible,
   where the risks are, what to verify before coding. Bad news — tell the owner plainly.
3. **Docs** (from the templates in `docs/`):
   - `01-VISION.md` — in full, delivery requirements REQ-1…REQ-n;
   - `02-ARCHITECTURE.md` — stack and folders (the rest will be filled in along the way);
   - `03-ROADMAP.md` — blocks with weights 1–5, status ⬜, the order and why;
   - `05-DECISIONS.md` — the stack choice and everything rejected; the branching model (`docs/process/git-rules.md`);
   - `07-RISKS.md` — risks from scouting with S×L rating;
   - `08-GLOSSARY.md` — terms that came up in the interview;
   - `04-STATE.md` — "project started, next block — B01";
   - `CLAUDE.md` — the "Project specifics" section, including the language and the branching model.
4. Show the owner the roadmap (blocks, weights, order) and wait for agreement. Edits — before the commit.
5. Git: `git init` (if there is no repository yet), branches per `docs/process/git-rules.md`, first commit
   `docs: project start — vision, roadmap, state`. Remind the owner to enable protection for `main`
   (and `develop`) — on GitHub: Settings → Branches → "Require status checks to pass".
6. Say: "Next block — B01. Type /clear, then /go". If the next block is an ordinary one —
   suggest `/model sonnet`.
