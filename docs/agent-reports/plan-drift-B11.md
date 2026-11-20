# plan-drift-checker — B11 (saved by the main chat: the agent's rules forbid it to write report files)

## Percentage (protocol §5)
- Now: 22 / 32 = 68%.
- B11 ✅ at weight 2: 24 / 32 = 75%. B11 ✅ at weight 3: 25 / 33 = 75%.
- With B08 trimmed to 2 as well: B11=3 → 25 / 32 = 78%; B11=2 → 24 / 31 = 77% — part of that rise is re-weighting.

## Weight and B08
- B11 should be 3: beyond the card it delivered the size preview (`seedvr2_neo/size_preview.py`, `javascript/seedvr2.js`,
  `style.css`), a regrouped UI and the UI default change (22 files, +1145/−139).
- B08's "a simple main UI (model and size on top, the rest in Advanced)" is done in B11. Remaining in B08: color-correction
  and attention-mode choice, Forge progress bar, interrupt inside the DiT step, title without "(24G)", GPU profiles. Keep
  weight 3 unless the owner agrees to 2.

## Acceptance criteria
- Met by tests / live runs: ×2 and long-edge sizes, per-size resolution and pass budget, old argument list, hostile
  values, 3840 cap, infotext, README numbers.
- Open at the time of the check: the owner's UI check (since confirmed live through the UI config of a running Forge) and
  the full test run + CI after push.

## Gaps
1. `03-ROADMAP.md`: B11 row weight/status, summary and the sum lines.
2. `03-ROADMAP.md`: trim the B08 row.
3. `04-STATE.md`: stale (B10 last closed, test count, percentage, next steps).
4. B11 card: "moving controls into Advanced — B08" is now done here; steps do not mention the preview. The card is
   deleted at block-done; the preview facts are already in `02-ARCHITECTURE.md`.
5. `agent-reports/idea-integrator-scale-presets.md`: delete; its conclusions are in `05-DECISIONS.md`.
6. `06-IDEAS.md`: nothing left to remove. No doc mentions a removed function.
7. `02-ARCHITECTURE.md` covers the new modules; no new requirement in `01-VISION.md`.
8. The extra work is owner-requested scope inside B11: record it by the weight, not as a new block.
