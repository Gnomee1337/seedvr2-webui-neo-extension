---
description: Close a block — criteria, tests, reviewer agents, fixes, docs, percentage, commit
---

Closing the current block per `docs/process/protocol.md` §3–§5.

1. **Acceptance criteria** from the card — go through each one and show proof (command,
   output, screen). A criterion not met — the block is not closed; say what is missing.
2. **All tests** of the project (not only the new ones) are green. Red ones — fix before the agents.
3. **Agents — in a single message, in parallel**, each gets: the block's changes (`git diff develop...HEAD`),
   the card, `docs/process/code-style.md`:
   - always: `code-auditor`, `fresh-eyes-reviewer` (give it only the gist of the block and the diff, no explanations), `plan-drift-checker`;
   - new module, changed data format, the block must run: `integration-checker`;
   - user data, authorization, outbound requests, files, money: `security-auditor`;
   - user input, money, external services: `stress-tester`.
4. **Triage findings:** remove duplicates; S×L rating; BLOCKER → IMPORTANT → MINOR. BLOCKER — fix;
   IMPORTANT — fix if it fits, otherwise a new block in the roadmap; MINOR — into `07-RISKS.md`.
   After a fix — **re-run the agent that found it**. Two rounds with no result — go to the owner.
5. **Docs:** conclusions from `docs/agent-reports/` and the card (pitfalls, assumptions) go into the permanent
   docs (`02-ARCHITECTURE`, `09-PROCESSES`, `05-DECISIONS`, `07-RISKS`, `external-apis/`,
   `08-GLOSSARY`). Commit the reports first (`docs: agent reports BXX`), then delete them together with the block card.
6. **Percentage:** the block gets ✅ in `03-ROADMAP.md` only if everything above is done; the percentage comes from the
   `plan-drift-checker` report. Update `04-STATE.md`: what is closed, the next block, questions.
7. **Git** (`docs/process/git-rules.md`): commit `feat(BXX): <what the block delivers>`; push the branch and wait for CI (no CI — run the full test suite locally); when green — merge
   `feature/BXX-*` into `develop` (`--no-ff`).
8. Summary for the owner, 5–10 lines. Model hint for the next block (protocol §10).
   "Type /clear, then /go".
