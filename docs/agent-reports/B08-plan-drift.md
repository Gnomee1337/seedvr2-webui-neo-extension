# B08 — plan-drift-checker report (relayed by the main chat; the agent could not write files)

Percentage with B08 closed: **84%** (28 of 33). Weights: B04 2 + B05 2 + B06 4 + B07 3 + B09 3 + B10 1 + B11 3 +
B08 3 + B12 5 + B01 3 + B02 3 + B03 1 = 33; closed: all but B12 = 28.

Gaps:
1. `03-ROADMAP.md`: B08 row, Summary and the sum check to update on close.
2. `04-STATE.md`: rewrite for B08 closed, 84%, B12 the only open block, test count 2583.
3. Card to delete on close; the `security-auditor` line is unchecked (no user data, files or requests touched — skip
   reason recorded in the close commit). Hidden assumptions already in `07-RISKS.md`.
4. `README.md:23-24` still said the original extension shows the same name — stale since the rename.
5. `07-RISKS.md` new entries in section 4 not sorted by R (cosmetic).
6. No stale "fixed lab/sdpa" / "interrupt only between phases" left; the remaining "(24G)" mentions are intentional.
7. Every card step has its code, tests and docs. No `seedvr2_vendor/` change, so no local-patch entry needed.
