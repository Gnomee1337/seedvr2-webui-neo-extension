# B13 — plan-drift-checker

Percentage once B13 is ✅: 100% (31 of 31). Today 28/31 = 90%. On `main` it was 28/33 = 84%; the jump comes from
deferring B12 (weight 5) to `06-IDEAS.md` as well as from closing B13.

Gaps:
1. `04-STATE.md` "Where we stopped" still describes B08 (190–220 s run, "top: R6 VAE decode spill", 3115 tests).
2. `03-ROADMAP.md` B13 ⬜ / "all"; STATE "1 block open"; card 🔄 — to ✅, 100%, sum check lines.
3. Card: CI criterion open until a push (owner's consent). Other criteria backed by the card's live tables, 3132 tests,
   new tests in `tests/seedvr2_neo/test_upscale_run.py`, code in `seedvr2_neo/upscale_run.py`.
4. Card criterion text still quotes the old baseline ("2.5–3 min vs ~15 s"); met at 9.0 s vs 8.0 s.
5. `07-RISKS.md` new entry is 🟢 R2 while the card says "close or lower R6"; the R6 entry was replaced — acceptable.
6. `02-ARCHITECTURE.md`, `09-PROCESSES.md`, `05-DECISIONS.md`, README match the code. No new module or argument.
