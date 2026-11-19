# B07 — plan-drift-checker report

(Recorded by the main chat from the agent's returned summary; the agent did not write files.)

## Percentage
Weights: B04 2 + B05 2 + B06 4 + B07 3 + B09 3 + B08 2 + B01 3 + B02 3 + B03 1 = 23.
- B07 not ✅: closed = 15 → 15/23 = 65.2% → **65%** (matches `03-ROADMAP.md`, `04-STATE.md`).
- B07 ✅: closed = 18 → 18/23 = 78.26% → **78%**.
- Test run: 1069 passed, 1 skipped, 2 xfailed (`04-STATE.md` still says 1004).

## Acceptance criteria (card)
- txt2img 2×2 with grid on — live via API (4 images, 4 seeds, 4 `-seedvr2`); UI gallery/info panel not checked.
- img2img with 2 inputs — Batch tab by the owner: 2 in → 2 out; the `save_to_dirs` fix was stub-only at review time
  (live re-run pending then).
- L/P/RGBA — met live.
- Save settings — met live ("Always save" off/on, no `-original` duplicate).
- Empty `process_images` result — stubs only.
- Interrupt — live for img2img only; txt2img interrupt and "gallery count matches" not shown live.
- Tests — met.

## Drift
1. Card criteria not ticked although evidence exists.
2. `03-ROADMAP.md` B07 row ⬜, summary, check-sum; `01-VISION.md` REQ-2 ⬜ — update on closing.
3. `04-STATE.md` stale (B06 as last closed, 1004 tests, 65% from B06, test branch `b06-test`).
4. `02-ARCHITECTURE.md` covers `run_inputs` / `run_results`; delete the card on `/block-done`.
5. `07-RISKS.md` §4 consistent; the old "only the first image" item is gone.
6. `01-VISION.md`: no uncovered REQs; `save_to_dirs` override limit documented (R2).
7. All changed files are in B07 scope; `seedvr2_vendor/` untouched.

Verdict: do not mark ✅ before the owner's live Batch-tab re-check; decide whether stub-only coverage of the empty
result and txt2img interrupt is acceptable.
