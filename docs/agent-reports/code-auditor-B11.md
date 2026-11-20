# code-auditor — B11 (output size modes)

Scope: `git diff main...HEAD` on `feature/b11-output-size` (7 commits). Card `docs/roadmap-blocks/B11-output-size.md`,
rules `docs/process/code-style.md`.

Tests: `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q` →
**1793 passed, 1 skipped, 3 xfailed** (green).

Extra verification done: `OutputTarget.output_size` was compared with the vendored `SideResize` logic (torchvision
`resize` + `max_size` scaling + even rounding) on 300 random sizes / modes / factors — **0 mismatches**. So the preview,
the console line, the pass budget and the pipeline agree on the size for every input the run accepts.

Totals: **BLOCKER 0, IMPORTANT 2, MINOR 7.**

---

## IMPORTANT

### I-1 🟡 R4 (S2×L2) — Scale factor / Long edge are cut by the old 4:1 cap: inputs wider than 4:1 come out smaller than asked, even downscaled
`seedvr2_neo/upscale_settings.py:118` — `max_resolution = min(resolution * MAX_LONG_EDGE_FACTOR, MAX_OUTPUT_LONG_EDGE)`.

The 4:1 cap (`MAX_LONG_EDGE_FACTOR`) protects the shortest-edge mode from gigapixel outputs. In the new modes the
long edge is already bounded by `MAX_OUTPUT_LONG_EDGE` (3840), so the extra cap only shrinks panoramas. Reproduced:

| Mode, input | Expected | Actual `target_for(...).output_size(...)` |
|---|---|---|
| Scale factor ×1, 600×3600 (6:1 phone panorama) | 600×3600 (or ≤3840) | **400×2400 — a downscale** |
| Scale factor ×2, 600×3600 | 640×3840 | 426×2560 — smaller than the input's long edge |
| Scale factor ×2, txt2img 2048×448 | 3840×840 | 3360×734 |
| Long edge 3840, 768×3840 | 768×3840 | 614×3072 |

This contradicts the UI info text (`scripts/seedvr2_script.py:202`, "no input is upscaled below 512 px"), README
(`README.md:56`, "no image is made smaller than 512 px on its shortest edge, so ×1 of a small image is still an
upscale") and the comment at `upscale_settings.py:37` ("Up to 4:1 panoramas keep the requested resolution" — in the new
modes they do not get the requested long edge either). The infotext then writes `SeedVR2 Resolution: 640` for an image
whose shortest edge is 426. The preview shows the same wrong number, so the user at least sees it — but has no way to
get ×1 / ×2 of such an image except the Shortest edge mode.
Tests miss it: `test_no_new_mode_gives_a_long_edge_over_3840` only checks the upper bound.

Fix: in the Scale factor / Long edge branches use `max_resolution = MAX_OUTPUT_LONG_EDGE` (the long edge is already
bounded there), and add a test like `factor(1).target_for((600, 3600)).output_size(...) == (600, 3600)` plus one for
a >4:1 input in Long edge mode. Rewrite README/UI text to say that a very thin image whose 512 px short edge would push
the long edge over 3840 gets a shorter edge (see M-1).

### I-2 🟡 R4 (S2×L2) — txt2img size line is computed from code defaults, not from the values the page actually shows
`scripts/seedvr2_script.py:222-225` — the initial `gr.Markdown` value is
`preview_txt2img_output_size(*(component.value for component in txt2img_inputs))`.

Per `docs/external-apis/forge-neo.md` (ui-config section) Forge creates the script UI **before** `UiLoadsave.add_block`
`setattr`s the saved `ui-config.json` values onto the components, and a value applied that way fires no `.change`.
So at build time `component.value` is the code default of Forge's width / height / Hires. fix controls and of our
own Scale Factor / Long Edge / resolution sliders.

Scenario: a user saved txt2img defaults 832×1216 (Settings → Defaults, or an edited ui-config.json) or saved Scale
Factor 1.5. Page opens with width 832, height 1216, Scale Factor 1.5 on screen; the line says
"Output size: **2048×2048** from 1024×1024"; Generate gives 1248×1824. The line stays wrong until the user touches one
of the watched controls. img2img does not have this problem: `javascript/seedvr2.js` clicks the refresh button on its
first tick (`lastSize` starts `null`). The test `test_txt2img_preview_starts_with_the_size_of_forge_s_controls`
passes because the stub has no ui-config step.

Fix: refresh the txt2img line once after load the same way img2img does (a hidden txt2img refresh button clicked by
`seedvr2.js` in `onUiLoaded`, wired to `preview_txt2img_output_size` with the same inputs), or start with the generic
note until the first event.

---

## MINOR

### M-1 🟢 R2 (S1×L2) — "raised to the 512 px minimum" is printed for outputs whose shortest edge ends far below 512
`seedvr2_neo/size_preview.py:30`, `seedvr2_neo/upscale_run.py:104`, flag set at `upscale_settings.py:119`.
`raised_to_minimum` means "the requested resolution was under 512", not "the output is 512". Input 100×1000, ×2 →
preview "Output size: **204×2048** from 100×1000 (raised to the 512 px minimum)"; console says the same. After I-1 is
fixed the case remains for inputs thinner than 7.5:1 (512 × ratio > 3840). Fix: word it from the actual result (e.g.
only add the note when `min(output_size) >= 512`, otherwise "(long edge capped at 3840)").

### M-2 🟢 R2 (S1×L2) — The 512 minimum is written out as a literal in three places
`seedvr2_neo/size_preview.py:30` (`"512 px minimum"`), `scripts/seedvr2_script.py:202` (`"below 512 px"`),
`seedvr2_neo/upscale_settings.py:66` comment. The knowledge is `RESOLUTION.minimum`; `upscale_run.py:104` already uses
`target.resolution`. If the minimum is lowered (listed as an out-of-scope future step in the card), the preview and the
UI info silently lie. Fix: format from `RESOLUTION.minimum`; better, one shared formatter for the "raised" suffix used
by both `describe_output_size` and `describe_target` (they currently also differ in `×` vs `x`).

### M-3 🟢 R1 (S1×L1) — Dead mapping entry `"txt2img_hr": "enable_hr"`
`scripts/seedvr2_script.py:50` plus the `isinstance(component, gr.Accordion)` guard in `after_component` (line 279).
Forge's `InputAccordion` gives its checkbox `elem_id="txt2img_hr-checkbox"` and its `gr.Accordion` `"txt2img_hr"`
(the script's own comment at lines 47-48 and forge-neo.md). The only component with `"txt2img_hr"` is the Accordion,
which the guard rejects, so the entry can never take effect; removing it and the guard changes nothing (the stub test
at `tests/scripts/test_seedvr2_script.py:506` would still pass). Fix: drop the entry and the guard, or keep the guard
and drop only the entry with a one-line reason.

### M-4 🟢 R1 (S1×L4) — State created lazily through `hasattr` instead of a constructor
`scripts/seedvr2_script.py:283-286` `_size_sources()`. code-style: "a class's state changes only through its methods";
initialising an attribute on first access hides it from readers and from type hints. Forge instantiates the script per
tab with no arguments, so a plain `__init__` setting `self._txt2img_size_sources: Dict[str, Any] = {}` works. Fix:
add `__init__` (calling `super().__init__()`), remove `_size_sources`.

### M-5 🟢 R1 (S1×L4) — `ui()` and the script module grew past the size guidelines
`scripts/seedvr2_script.py:180-273` — `ui()` is ~93 lines (guideline 30-40), with the size group, mode switching and
two preview wirings inline; the module is 437 lines (guideline ~300). Fix: extract `_build_size_group(is_img2img)`
returning `(size_controls, note)` and move the txt2img preview wiring next to `_preview_img2img_output_size`.

### M-6 🟢 R1 (S1×L1) — Comment states history
`scripts/seedvr2_script.py:336-338`: "Parameters with defaults were added after the first API callers." Protocol §11:
comments explain why, not history. Fix: "Forge passes a shorter script_args list as is; the defaults are what a caller
that does not send these arguments gets: a shortest edge in px."

### M-7 🟢 R1 (S1×L1) — Block card out of date
`docs/roadmap-blocks/B11-output-size.md:19` still says `estimate_output_pixels` takes the per-size resolution (the
function was removed; `split_into_passes` takes an `output_pixels` callable), and line 47 says "1751 passed" (the run
now gives 1793). The card is deleted on `/block-done`, but the acceptance checklist should match before that.

---

## Checked, no finding
- `OutputTarget.output_size` vs the vendored `SideResize` + even rounding: identical on 300 random cases.
- `split_into_passes` with the per-size `output_pixels` callable; `targets` keyed by size, groups keyed by
  (size, mode) — every pass is one size, so `targets[pass_images[0].size]` is right.
- Infotext per image uses the same `target_for(image.size)` as the run; non-upscaled images still get the failure
  reason (`finish_images`), lists zipped in input order.
- Old API callers (16 args): `run()` defaults → Shortest edge; positional-order and hostile-value tests exist and pass.
- `parse_size_mode` / `clamp_to_bounds` for NaN, inf, bool, strings, lists; `txt2img_final_size` for NaN/inf/None and a
  broken `res_step`.
- No stale reference to `estimate_output_pixels` or `UpscaleSettings.max_resolution` in code or permanent docs;
  "Scale presets" removed from `06-IDEAS.md`.
- No debug output, commented-out code or TODOs added.
