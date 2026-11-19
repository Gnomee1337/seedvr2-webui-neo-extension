# Code audit — B07 (correct results)

Scope: `git diff main...HEAD` on `feature/b07-correct-results` (commits e132827, 568208e, 8b44152, efcf507), block card,
`docs/process/code-style.md`. The script itself has no unit tests (Forge stubs are B09); that gap is not counted as a
defect of this block.

Tests: `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q` →
**1069 passed, 1 skipped, 2 xfailed**. Edge cases below were reproduced with a scratch script against the helpers
(Pillow 3.10 env), not by editing the project.

Checked and found correct: index alignment inputs ↔ details ↔ `merge_run_images` ↔ infotexts (all built from the same
`inputs` list, `upscaled` keyed by input index, groups mapped back by index); grid skipping for images and infotexts
uses the same `index_of_first_image`; `finish_run` runs in `finally` on success / interrupt / exception after
`prepare_run`; early returns (no inputs, interrupt) happen before any model is loaded; no unused imports left after
removing `create_infotext` / `random`; `RESULT_SUFFIX` / `ORIGINAL_SUFFIX` keep the old file names byte for byte;
`SeedVR2 Model` / `SeedVR2 Resolution` keys kept.

Summary: BLOCKER 0, IMPORTANT 1, MINOR 9.

---

## IMPORTANT

### I1. Skip in the last txt2img batch: no `-seedvr2` file at all, and `-original` duplicates Forge's files — 🟡 R4 (S2×L2)
`scripts/seedvr2_script.py:312`, `seedvr2_neo/run_results.py:49`

`plan_saving` gets `p.save_samples()`, evaluated once **after** `process_images` returned. `save_samples()` includes
`not state.skipped` (forge-neo.md:47), and `process_images_inner` resets `state.skipped` only at the top of each batch
(forge-neo.md:228-229), so a Skip pressed during the **last** batch stays set when `run()` reads it.

Scenario: txt2img, batch count 2, "Always save" on, "Save Extra Copy of Original" on; the user presses Skip during
batch 2. Forge saved batch 1's base images (skipped was False then). Now `samples_allowed=False` →
`save_results=False` (no `-seedvr2` file for any image, including batch 1's complete ones) and
`base_saved_by_forge=False` → `save_originals=True` → batch 1's images are written a second time as `-original`.
This breaks the acceptance criterion "never writes a second copy of a file Forge already saved".

Fix: for the plan use the static part of the rule (`opts.samples_save and not p.do_not_save_samples`, plus
`state.interrupted` unless `save_incomplete_images`) rather than `p.save_samples()` with the stale `skipped` flag.

---

## MINOR

### M1. Request conversion moved outside the `try`: bad API args now escape `run()` — 🟢 R2 (S2×L1)
`scripts/seedvr2_script.py:322-327`

Before B07, `UpscaleRequest(... int(resolution), float(input_noise), int(tile_size) ...)` was built inside the
`try/except Exception` and a bad value gave `empty_result("Runtime Error: …")`. Now it is built in `run()` outside
any `try`. Scenario: API txt2img with `script_args` resolution `null` → `TypeError` escapes after the base images were
generated, (optionally) saved as `-original`, and the SD checkpoint was unloaded → HTTP 500, generated images not
returned. Catching behavior changed (code-style "Exceptions"). Fix: build/validate the request before
`_generate_base_images` (as the memory plan used to be), so a bad value costs no generation.

### M2. 16-bit grayscale with transparency becomes pure white — 🟢 R2 (S2×L1)
`seedvr2_neo/run_inputs.py:61-67`

The alpha branch (`"transparency" in image.info`) runs before the wide-grayscale branch, so `I;16` / `I` with a tRNS
value goes through a plain `convert("RGBA")`, which clips at 255 — exactly what the wide branch exists to avoid.
Reproduced: `Image.new("I;16", (1,1), 32768)` with `info["transparency"]=0` → `(255, 255, 255, 255)`. Scenario: a
16-bit grayscale PNG with a tRNS chunk via the API → white output. Fix: scale wide grayscale to `L` first, then apply
the alpha rule (`L` + transparency → RGBA); add the test.

### M3. Not-upscaled images and `-original` copies keep the raw input mode — 🟢 R2 (S2×L1)
`scripts/seedvr2_script.py:314, 331` (`merge_run_images(inputs, …)`, `save_originals(p, inputs, …)`)

Failed images are returned as the raw `inputs`, not as `to_pipeline_image(...)`. Pillow cannot write CMYK / F / YCbCr
as PNG (`OSError: cannot write mode CMYK as PNG`, reproduced). Scenario: API img2img with a CMYK JPEG, interrupted or
failing upscale → the returned `Processed` holds a CMYK image; the API's PNG base64 encoding (and Gradio's gallery
export) fails on it. Same input with "Save Extra Copy of Original" → `-original` save fails (printed, skipped). Fix:
convert inputs once with `to_pipeline_image` before the run and use the converted list for originals, failures and
the pipeline.

### M4. UI seed outside [0, 2³²−1] silently replaced — 🟢 R2 (S1×L2)
`seedvr2_neo/run_inputs.py:101`; doc `docs/09-PROCESSES.md` process 5 step 1

Old code used any non-`-1` UI seed as is. Now a typed seed like `9999999999` or `-5` counts as "not chosen" and the
run silently follows the image seed / `p.seed` / random (reproduced: `resolve_seed(2**40, None, 5) == 5`). The UI
`gr.Number` has no bounds, so the user sees a different `SeedVR2 Seed` than typed, without a message. 09-PROCESSES
says only "an API value that is not a number counts as -1". Fix: print a console note when a given seed is rejected
(or clamp/mod into range), and document the range.

### M5. txt2img API `override_settings` reach the base images but not the upscaled ones — 🟢 R2 (S1×L2)
`scripts/seedvr2_script.py:166, 312`; `docs/07-RISKS.md:120`

`process_images` applies `p.override_settings` and restores them when it returns; `run()` then calls
`p.save_samples()` and `save_image` with the user's own opts (only `save_to_dirs` is passed on). Scenario: API txt2img
with `override_settings={"samples_save": false}` and `save_images: true` → Forge does not save the base images, the
script saves `-seedvr2` files and (with "original" on) skips the original because it thinks Forge saved it. 07-RISKS
§4 describes this limitation for img2img only. Fix: extend the risk entry to txt2img (or read the overrides for
`samples_save` / `samples_format` too).

### M6. `run()` too long, two parallel `if is_txt2img` branches; module 467 lines — 🟢 R2 (S1×L2)
`scripts/seedvr2_script.py:262-332`

`run()` is ~70 lines (guideline 30–40) and mixes setup checks, input collection (branch at 281), seed/details
(second branch at 303), saving plan, SD unload, request building and result. Fix: extract
`_collect_txt2img_inputs` / `_collect_img2img_inputs` returning `(inputs, details, result_seed)`; the module is past the
~300-line guideline (pre-existing, grown here).

### M7. Vestigial parameter and magic strings — 🟢 R1 (S1×L1)
`scripts/seedvr2_script.py:391, 398, 178, 295, 376`

`_upscale_tensor(..., check_interruption)` takes a parameter shadowing the module-level `check_interruption`, and
`_upscale_group` passes that same global in — a leftover from the removed closure. `"Interrupted"` is written twice
(295, 376) and `"not upscaled"` (178) is an unnamed fallback text. Fix: drop the parameter and use the module
function; name the failure texts as constants.

### M8. Comment justifies the img2img seed choice with a false premise — 🟢 R1 (S1×L1)
`scripts/seedvr2_script.py:301`

"`p.all_seeds` there may be left from the previous Batch tab file": in img2img `all_seeds` is set only inside
`process_images` (forge-neo.md:193-196), and the Batch tab calls it only when `run()` returns None, which this script
never does — so it is always unset there. The code is fine; the "why" is wrong. Fix: "img2img inputs have no seeds of
their own; `p.all_seeds` is not set without `process_images`".

### M9. Docs/tests diverging from code — 🟢 R2 (S1×L2)
- `docs/07-RISKS.md:123` "img2img infotext 'Seed' is the SeedVR2 seed … 'Send to' puts it into the seed field": the
  img2img infotext has **no** `Seed:` key (`BaseInfotext.from_prompt` + only `SeedVR2 …` keys, `run_results.py:96-104`);
  only `Processed.seed`/`all_seeds` (the gen-info JSON, ♻️ button) carry the SeedVR2 seed. Rewrite the entry.
- Bug-fix commit efcf507 (`save_to_dirs` override, `seedvr2_script.py:166`) has no test; the rule "read
  `override_settings['save_to_dirs']`, None otherwise" can be a pure helper in `seedvr2_neo/` with a test now, without
  waiting for B09 (code-style: "Fixed a bug: first a test that catches it").
- Card's hostile-input list asks for "huge index": `select_generated_images` with a huge index/count, `select_distinct_images([])`,
  and `resolve_seed` with out-of-range / negative / bool values have no tests in `tests/seedvr2_neo/test_run_inputs.py`
  (they behave correctly when probed; the tests are missing).
- Card step 3 and "Hidden assumptions" still describe filling `create_infotext`'s lists in img2img; the code no longer
  does that (card is temporary, update or drop at `/block-done`).

---

## Notes (not findings)
- Infotext/grid alignment relies on Forge inserting the grid's infotext at index 0 together with the grid image
  (A1111 does `infotexts.insert(0, …)`); `forge-neo.md:25` records only `output_images.insert(0, grid)`. The live 2×2
  run passed; worth adding the infotexts fact to `forge-neo.md`.
- `prepare_run` (line 359) is still outside the `try` — unchanged from B06.
- Float (`F`) images in [0, 1] convert to black (Pillow clips without scaling); very unlikely input, not raised as a finding.
