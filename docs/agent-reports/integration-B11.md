# integration-checker — B11 (saved by the main chat: the agent's rules forbid it to write report files)

No blockers, 2 IMPORTANT, 5 MINOR. Committed suite: 1793 passed, 1 skipped, 3 xfailed.

## IMPORTANT
- **I1 (S1×L3 = 3) txt2img size line is wrong at page load** — `scripts/seedvr2_script.py:222-225`. The initial text
  comes from code defaults inside `ui()`; Neo applies `ui-config.json` later (`loadsave.add_block`, `apply_field`), and
  nothing fires on load. img2img self-corrects (first JS poll). Fix: a hidden txt2img refresh button clicked once from
  `onUiLoaded`.
- **I2 (S2×L2 = 4) panoramas wider than 4:1 come out smaller than asked** — `seedvr2_neo/upscale_settings.py:118`: the
  4:1 cap applies on top of the 3840 cap in Scale factor / Long edge: ×2 of 4096×512 → 2048×256; ×1 of 3000×600 →
  2400×480; the line still says "raised to the 512 px minimum". Long edge can exceed the chosen value when the 512 floor
  applies (4081×850 at 1664 → 2048×426). Fix: `max_resolution = MAX_OUTPUT_LONG_EDGE` in those modes, long-edge cap wins
  over the 512 floor, flag the raise only when the shortest edge really reaches 512, correct README.

## MINOR
- M1 (2) a side can become 0: 2048×1 at ×2 → 2048×0 (also Shortest edge 4096×1, pre-existing).
- M2 (1) `res_step` read live; Neo's `sRound` uses the value fixed at import (option needs a restart).
- M3 (1) `finish_images` zips three sequences silently; suggest `zip(..., strict=True)`.
- M4 (1) `after_component`: a later component with the same `elem_id` would replace Forge's (none found).
- M5 (1) `seedvr2.js` falls back to `img.width` when `naturalWidth` is 0 (same as Neo's own function); not checked live.

## Verified OK
1. `ui()` returns 19 controls in `run()`'s order; README API order matches; 13 / 16-argument calls work; a missing
   `size_mode` falls back to Shortest edge.
2. txt2img `elem_id`s match Neo `modules/ui.py`; the InputAccordion checkbox is `txt2img_hr-checkbox`; the Accordion is
   skipped; selectable scripts receive `after_component`; a reordered UI falls back to the generic line.
3. `_js` → `js` conversion (`gradio_extensions.py`); the JS returns 2+4 values for 2+4 inputs; button id matches;
   `javascript/*.js` and the root `style.css` are loaded; `onUiLoaded` exists.
4. Gradio 4.40: all keywords used exist; `.change` uses `trigger_mode="always_last"`; `do_not_save_to_config` honoured.
5. `OutputTarget` vs torchvision + `SideResize` + even rounding: 7943 cases, 0 mismatches; `ctx` rebuilt per pass.
6. `finish_images` / `seedvr2_generation_params`: single callers, updated, covered by tests.
