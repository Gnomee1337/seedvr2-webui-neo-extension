---
description: Final project acceptance — all checks on the strongest model
---

Final acceptance. **Precondition:** the chat is on the strongest model. If not — ask to
switch (`/model …`) and wait for confirmation. Do not start without it.
Raise the effort: `/effort high`.

1. Are all blocks in `03-ROADMAP.md` ✅? If not — list the open ones and stop.
2. Full run of all tests.
3. **All reviewer agents in a single message, in parallel**, on the whole project (not on a diff):
   `code-auditor`, `fresh-eyes-reviewer` (give it only `01-VISION.md` and the code, no explanations), `security-auditor`,
   `stress-tester`, `integration-checker`, `plan-drift-checker`.
4. Triage findings as in `/block-done` (S×L rating, fixes, re-run the agent that found it).
5. **Delivery requirements** from `01-VISION.md` — for each REQ: is it met, and the proof.
   Checklist `docs/owner-checklists/processes.md` (if present) — is it up to date, each process with its status.
6. Docs reconciled with the code: ARCHITECTURE, 09-PROCESSES, RISKS (only accepted risks remain),
   instructions for the client.
7. **Acceptance report** for the owner: what is delivered, which requirements are met, known limitations and
   risks (with ratings), what must be done by hand at launch. The percentage is 100% only per `plan-drift-checker`.
8. Release — via `/release`, only with the owner's consent.
