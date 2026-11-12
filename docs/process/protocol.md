# Project work protocol

These are instructions for Claude Code. Follow them always, without reminders.

The `docs/process/` folder is identical in every project: it is not edited per project; rule
improvements go into the template and the whole folder is copied over. Everything specific to a
given project lives in `CLAUDE.md` ("Project specifics") and in the numbered documents in `docs/`.
**Rules version: 2.0**

---

## 0. Five core principles

1. **One roadmap block = one chat session.** Context is cleared between blocks.
2. **`docs/04-STATE.md` is the single source of truth** about where we stand. It must be short (up to 60 lines) and always up to date.
3. **Make nothing up.** Facts about external services and libraries come only through the `fact-checker` agent, with a link to the official documentation.
4. **Only the main chat edits code.** Agents only review and report. Exception: `fact-checker` writes to `docs/external-apis/`, `stress-tester` writes tests.
5. **Every decision that affects the future goes briefly into `docs/05-DECISIONS.md`.** Three context clears later you won't remember why you chose this path, and you'll redo it.

---

## 1. Saving context: three reading levels

Don't go up a level higher than the current task needs.

| Level | What | When |
|---|---|---|
| **L0** | `CLAUDE.md` and everything it includes: the protocol, `docs/04-STATE.md`, the `docs/README.md` map | always, loaded automatically |
| **L1** | current block card + the "Modules" section of `02-ARCHITECTURE.md` + decision **headings** from `05-DECISIONS.md` | at the start of a block |
| **L2** | specific code files, `docs/external-apis/*`, agent reports, the body of a specific decision | only when directly needed |

**Forbidden:**
- Reading the whole project "to remember". Instead: the module map plus targeted search by name.
- Retelling whole file contents in the chat. Only the conclusion.
- Re-reading a file you already read in this session.

**Hand off to agents** everything heavy: code search, reading documentation online, audits. An agent has its own separate context, so the main chat doesn't fill up.

---

## 2. Watching context usage

The bar at the bottom of the screen shows how full the context is.

| Usage | What to do |
|---|---|
| up to 60% | work calmly |
| 60–80% | finish the current piece, don't start a new one |
| 80%+ | `/save` immediately, then tell the user: "type /clear, then /go" |

Exact numbers are shown by the `/context` command.

**Don't use `/compact` instead of `/save` + `/clear`.** Compaction drops details unpredictably: you don't know what you lost. Our way drops exactly what we ourselves recorded as "no longer important", and what matters stays in files.

## 2a. If the project was "inherited"
If the folder has code but no `docs/04-STATE.md`, **don't start planning from scratch and don't write code**. Tell the user: "The project is already started, but under a different system. Type `/migrate`: I'll sort out what's there and move it to our process without losing anything." And stop.
Same if `docs/04-STATE.md` exists but clearly doesn't match the code (describes something else entirely).

## 3. One block cycle

```
/go                          ← read STATE, reported, planned
  ↓
fact-checker (if any API)    ← checked facts BEFORE code, not after
  ↓
work + explanations in plain words
  ↓
/check (optional, mid-block)
  ↓
/block-done                  ← review agents → fixes → updated plan and % → commit
  ↓
/clear → /go                 ← next block
```

If a new idea comes up mid-block, use the `/idea` command. Don't drop the current block for it.

**Outside the block cycle:** an urgent production breakage is `/hotfix` (short cycle: test → fix →
reviews → rollout with the owner's consent); releasing what has accumulated in `develop` is `/release`.

---

## 4. Agent rules

**Who runs when**: the table and stage diagram are in `docs/process/agents-and-commands.md`.

**Fresh eyes:** a significant change (refactoring, a dangerous module, a bug fix, a rewritten document) is reviewed by `fresh-eyes-reviewer` before the commit; it gets only the task and the diff, without your explanations.

**How to launch:** several agents at once go **in one message, in parallel**. Faster and cheaper.

**How to process findings:**
- Remove duplicates across reports.
- Give each finding a risk rating using the "severity × likelihood" matrix (`docs/07-RISKS.md`) and an icon.
- Sort BLOCKER → IMPORTANT → MINOR.
- BLOCKER: fix immediately. IMPORTANT: fix immediately if it fits in the block, otherwise a new block in the roadmap. MINOR: into `docs/07-RISKS.md`.
- After a fix, **rerun the agent that found the problem** and make sure the finding is gone. Not "I fixed it, probably fine now".
- If the problem doesn't go away after two rounds, stop, explain to the user, offer options.

**Don't dump full agent reports into the chat.** A 5-10 line summary in plain words; full reports go into the temporary folder `docs/agent-reports/`. On `/block-done` the conclusions from reports are moved into permanent documents (`07-RISKS`, `05-DECISIONS`, `external-apis/`, `docs/09-PROCESSES.md`), and the reports themselves are deleted. Before that they are committed together with the work (`/save`), so they remain in git history.

---

## 5. How the completion percentage is calculated

In `docs/03-ROADMAP.md` every block has a **weight from 1 to 5**: how much work it is.

```
PROGRESS = (sum of weights of ✅ blocks) / (sum of weights of ALL blocks) × 100, rounded down
```

- A block counts **only as a whole**. A half-done block = 0. This is deliberate, so the percentage doesn't lie.
- A block can be marked ✅ only when **all** acceptance criteria from its card are met and agent reviews have passed.
- When a new idea is added, a new block with a weight appears, and the percentage **honestly drops**. Tell the user exactly that: "it was 60%, now 52%: the project grew, it didn't regress".
- The percentage is calculated by the `plan-drift-checker` agent, not the main chat. The main chat tends to flatter itself.

---

## 6. What to write in DECISIONS

Record a decision if at least one applies:
- chose one technology out of several;
- rejected something that seemed obvious;
- agreed on a data format between modules;
- the user said "no" to some idea;
- worked around an external service limitation in a non-standard way.

Format: a heading and 1–5 lines readable six months later, in a suitable thematic section:
```
### Store settings in a JSON file, not in the database
Why: there are few settings; a database for them is needless complexity.
```

**The log keeps only decisions in force.** No numbers or dates. If a new decision cancels an
old one, the old one is deleted or rewritten; history stays in git. If a decision refines an
existing one, the existing entry is edited rather than a new one added.

---

## 7. Tests and stress tests, not only at the end

Don't put everything off to the finale: by the end a snowball builds up that can't be untangled.

- In every block with user input: at least the basic hostile scenarios (empty, very long, wrong type, someone else's identifier).
- Save scenarios into the project's tests so they can be rerun.
- At the end of each block run **all** accumulated tests, not just the new ones. That catches "fixed it here, broke it there".
- Tests where relevant: a change to documentation or comments only doesn't require a full run
  (it's enough to confirm the code didn't change); a code change needs the tests of the affected area,
  with a full run at the end of the block or in CI.
- Keep a living list of "what else can break" in `docs/07-RISKS.md` and add to it as you go.

---

## 8. Git as insurance

Full branch and commit rules: `docs/process/git-rules.md`. The essentials:
- Commit on every `/save` (`wip: ...`) and on every `/block-done` (`feat(BXX): ...`).
- Before a risky rework, a separate commit so there's something to roll back to.
- A behavior (bug) fix goes in a separate commit, not mixed with refactoring.
- Code and its tests go in one commit. Only the main chat commits, not agents.
- Never commit `.env`, keys, passwords.
- If the user says "it broke, put it back", that's a rollback to the last commit of a closed block; give them the exact command.

## 8a. Code rules

Full rules: `docs/process/code-style.md`: what must not change during refactoring, clean code (SOLID,
DRY, KISS/YAGNI), modules, classes, meaningful names, exceptions, security, tests in the mirrored
test folder. This is a block acceptance condition: code that works but is hard to read is not done.
Give this file, along with the task, to agents that write or review code.

---

## 9. How to talk to the user

- In the communication language from `CLAUDE.md` ("Project specifics"), in plain words, briefly.
- A term gets an explanation in parentheses right away and an entry in `docs/08-GLOSSARY.md`.
- Manual actions step by step: where to go, what to click, what to enter, how to tell it worked.
- Call a bad idea bad and explain why. Don't just agree.
- "Done" only after verification.

---

## 10. Which model at which stage

The user picks the main chat model with the `/model` command. You don't switch it for them, but you **must suggest** when it's worth doing.

| Stage | Model | Why |
|---|---|---|
| `/finish`: final acceptance | **the strongest** | digs deeper and double-checks itself more often. No saving here |
| `/start-project`, `/idea`, major architecture rework | **strong (opus)** | decisions are made where a mistake costs weeks of rework |
| regular blocks from a ready card | **sonnet** | decisions are already made, careful implementation is needed: faster and cheaper |
| fact-checking, percentage calculation, reconnaissance | sonnet | set in the agents themselves, inheritance doesn't apply |

Separately: the **`opusplan`** alias keeps opus for planning and switches to sonnet for implementation on its own. If the user doesn't want to think about switching, suggest it once at the start of the project.

**When to prompt the user:**
- At the end of `/start-project` and `/block-done`: if the next block is a regular one and the chat is on a strong model, say in one line: "The next block is simple, you can switch: `/model sonnet` will be faster and cheaper".
- If the user is on sonnet and enters `/idea` or a weight-5 block, say: "Decisions here are expensive, better use `/model opus`".
- Before `/finish`, **always** ask for the strongest model and wait for confirmation. This isn't a suggestion but a precondition for running acceptance.
- Once per block, no more. Don't nag.

**Agents** inherit the main chat model unless their file says otherwise. This means: on sonnet reviews are cheaper, but `code-auditor`, `security-auditor` and `stress-tester` find less. At the finale, reviewer agents (except those pinned to sonnet in their own file) run on the strongest model; that's exactly why acceptance is done on it.

There's also an effort level: the `/effort` command (`low`/`medium`/`high`). Raise it to `high` for architecture and the finale; if the user asks how to speed things up or make them cheaper, tell them about it.

---

## 11. How to write documentation and code comments

- **Documents describe the current state, not history.** No "first we did this, then
  found out, then redid it", and no change dates. Behavior changed: rewrite the paragraph.
  History lives in git (`git log -p -- <file>`).
- **No repetition.** Each kind of knowledge has one place:
  `02-ARCHITECTURE`: what exists in the code; `docs/09-PROCESSES.md`: how a process works step by step;
  `05-DECISIONS`: why it was decided so; `external-apis/`: facts about external APIs; `07-RISKS`: what can
  break; `06-IDEAS`: open ideas; `03-ROADMAP`: blocks and percentage. From other places, a link.
- **A block card lives while the block is open.** On `/block-done` the important parts (pitfalls, assumptions)
  are moved into permanent documents and the card is deleted.
- **Code comments explain "why" directly**, in your own words. No decision or block numbers
  ("see D12", "B07"), no dates, and no "which decision made it so". A link to the reference
  `docs/external-apis/...` is fine when a fact about someone else's API is long.
- **The language comes from `CLAUDE.md`** ("Project specifics") everywhere a developer reads: documentation, comments, docstrings, commit messages.
  What the client and the product's users see (UI texts, messages, labels on
  pages, `docs/for-client/`) stays in its own language; don't translate it.
