---
description: Honest completion percentage and gaps between the plan and the code
---

Run the `plan-drift-checker` agent. It calculates the percentage, not the main chat.

Summary for the owner:
- the percentage and how it was calculated (sum of ✅ weights / sum of all weights);
- what is open and what is closest to being closed;
- gaps: blocks marked ✅ without met criteria; code that is not in the plan;
  outdated parts of `04-STATE.md` and `02-ARCHITECTURE.md`.

If the percentage in `03-ROADMAP.md` / `04-STATE.md` differs from the calculated one — fix the docs
(with the owner's consent if block statuses change).
